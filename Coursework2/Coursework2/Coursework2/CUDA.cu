#define _USE_MATH_DEFINES

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <fstream>
#include <chrono>
#include <math.h>

#define BLOCK_SIZE 1024
#define EPS 3e4f // Softening factor
#define G 6.673e-11f // Gravitational constant

using namespace std;
using namespace std::chrono;

// Output file
ofstream dataFileOutput("data.csv", ofstream::out);

// Number of bodies
const int N = 1000;

// N-Body constants
const float solarmass = 1.98892e30f;
const float radiusOfUniverse = 1e18;

// create and initiliase a new body
struct Body
{
	float rx, ry;//cartesian positions
	float vx, vy;//velocity components
	float fx, fy;//force components
	float mass;//mass
};

// Global Bodies
//Body bodies[N];

// update velocity and position using timestamp dt
Body Update(Body body, float dt)
{
	body.vx += dt * body.fx / body.mass;
	body.vy += dt * body.fy / body.mass;
	body.rx += dt * body.vx;
	body.ry += dt * body.vy;
	return body;
}

// returns distace between two bodies
float distanceTo(Body a, Body b)
{
	float dx = a.rx - b.rx;
	float dy = a.ry - b.ry;
	return sqrt(dx * dx + dy * dy);
}

// set force to 0 for the next iteration
Body ResetForce(Body body)
{
	body.fx = 0.0;
	body.fy = 0.0;
	return body;
}

// convert to string representation formatted nicely
void PrintBody(Body body)
{
	printf("rx: %f, ry: %f, vx: %f, vy:%f, mass:%f \n", body.rx, body.ry, body.vx, body.vy, body.mass);
}


__global__ void bodyForce(Body *a, Body *b, int n, float dt)
{
	// Get block index
	unsigned int block_idx = blockIdx.x;
	// Get thread index
	unsigned int thread_idx = threadIdx.x;
	// Get the number of threads per block
	unsigned int block_dim = blockDim.x;
	// Get the thread's unique ID
	unsigned int idx = (block_idx * block_dim) + thread_idx;

	if (idx < n){
		float fx = 0.0f; float fy = 0.0f;
		for (int i = 0; i < n; ++i){
			if (idx != i){
				float dx = a[i].rx - a[idx].rx;
				float dy = a[i].ry - a[idx].ry;
				float distance = sqrt(dx*dx + dy*dy);
				float F = (G * a[idx].mass * a[i].mass) / (distance*distance + EPS*EPS);
				fx += F * dx / distance;
				fy += F * dy / distance;
			}
		}
		b[idx].vx += dt * fx / a[idx].mass;
		b[idx].vy += dt * fy / a[idx].mass;
		b[idx].rx += dt * a[idx].vx;
		b[idx].ry += dt * a[idx].vy;
	}
}
// compute the net force acting between the b a and b, and add to the net force acting on a
Body AddForce(Body a, Body b)
{
	//float EPS = 3E4; // softening parameter (just to avoid infinities)
	float dx = b.rx - a.rx;
	float dy = b.ry - a.ry;
	float dist = sqrt(dx * dx + dy * dy);
	float F = (G * a.mass * b.mass) / (dist * dist + EPS * EPS);
	a.fx += F * dx / dist;
	a.fy += F * dy / dist;
	return a;
}

// Random number generator
float randomGenerator(float min, float max)
{
	return ((float(rand()) / float(RAND_MAX)) * (max - min)) + min;
}

// bodies are initliased in circular orbits around the central mass. This is the physics to do that
float circlev(float rx, float ry)
{
	float r2 = sqrt(rx * rx + ry * ry);
	float numerator = (6.67e-11) * 1e6 * solarmass;
	return sqrt(numerator / r2);
}

// template for the signum function contained within Java's Math library
template <typename T>
int signum(T val)
{
	return (T(0) < val) - (val < T(0));
}

// Initialise N bodies with random positions and circular velocities
void startTheBodies(Body *bodies)
{
	bodies[0].rx = 0.0f;
	bodies[0].ry = 0.0f;
	bodies[0].vx = 0.0f;
	bodies[0].vy = 0.0f;
	bodies[0].mass = solarmass;

	for (int i = 0; i < N; ++i)
	{
		float px = 1e18 * exp(-1.8) * (.5 - randomGenerator(0.0, 1.0));
		float py = 1e18 * exp(-1.8) * (.5 - randomGenerator(0.0, 1.0));
		float magv = circlev(px, py);

		float absangle = atan(abs(py / px));
		float thetav = M_PI / 2 - absangle;
		float vx = -1 * signum(py) * cos(thetav) * magv;
		float vy = signum(px) * sin(thetav) * magv;

		// Orientate a random 2D cirular orbit
		if (randomGenerator(0.0, 1.0) <= 0.5)
		{
			vx = -vx;
			vy = -vy;
		}

		// Calculate mass
		float mass = solarmass * randomGenerator(0.0, 1.0) * 10 +1e20;
		// Assign variables to a body struct
		bodies[i].rx = px;
		bodies[i].ry = py;
		bodies[i].vx = vx;
		bodies[i].vy = vy;
		bodies[i].mass = mass;
		//printf("rx: %f, ry: %f, vx: %f, vy:%f, mass:%f \n", bodies[i].rx, bodies[i].ry, bodies[i].vx, bodies[i].vy, bodies[i].mass);
	}
};

//void addForces()
//{
//	for (int i = 0; i < N; ++i)
//	{
//		bodies[i] = ResetForce(bodies[i]);
//		for (int j = 0; j < N; ++j)
//		{
//			if (i != j)
//			{
//				bodies[i] = AddForce(bodies[i], bodies[j]);
//			}
//		}
//	}
//
//	// Loop again, update bodies with timestamp
//	for (int i = 0; i < N; ++i)
//	{
//		bodies[i] = Update(bodies[i], 1e11);
//	}
//}

void cudaInfo()
{
	int device;
	cudaGetDevice(&device);

	cudaDeviceProp properties;
	cudaGetDeviceProperties(&properties, device);

	cout << "Name: " << properties.name << endl;
	cout << "CUDA Capability: " << properties.major << "." << properties.minor << endl;
	cout << "Cores: " << properties.multiProcessorCount << endl;
	cout << "Memory: " << properties.totalGlobalMem / (1024 * 1024) << "MB" << endl;
	cout << "Clock Freq: " << properties.clockRate / 1000 << "MHz" << endl;
}

int main()
{
	// Initialise CUDA - select device
	cudaSetDevice(0);
	cudaInfo();

	float dt = 0.01f;
	// Host input struct
	Body *h_a;

	// Host ouput struct
	Body *h_b;

	// Device input struct
	Body *d_a;

	// Device output struct
	Body *d_b;

	// host memory size
	auto data_size = sizeof(Body) * N;

	// Allocate memory for each struct on host
	h_a = (Body*)malloc(data_size);
	h_b = (Body*)malloc(data_size);

	// Allocate memory for each struct on GPU
	cudaMalloc(&d_a, data_size);
	cudaMalloc(&d_b, data_size);

	startTheBodies(h_a);

	auto start = system_clock::now();
	cudaMemcpy(d_a, h_a, data_size, cudaMemcpyHostToDevice);

	int gridSize;

	// Number of thread blocks in grid
	gridSize = (int)ceil((float)N / BLOCK_SIZE);

	// Execute the kernel
	bodyForce << <gridSize, BLOCK_SIZE >> >(d_a, d_b, N, dt);

	// Copy back to host
	cudaMemcpy(h_b, d_b, data_size, cudaMemcpyDeviceToHost);
	cout << data_size << endl;
	// Release device memory
	cudaFree(d_a);
	cudaFree(d_b);

	// Release host memory
	free(h_a);
	free(h_b);

	//addForces();

	//for (int i = 0; i < N; i++)
	//{
	//	PrintBody(bodies[i]);
	//}

	auto end = system_clock::now();
	auto total = end - start;
	cout << "Number of Bodies = " << N << endl;
	cout << "Main Application time = " << duration_cast<milliseconds>(total).count() << "ms" << endl;
	//dataFileOutput << duration_cast<milliseconds>(total).count() << endl;

	return 0;
}
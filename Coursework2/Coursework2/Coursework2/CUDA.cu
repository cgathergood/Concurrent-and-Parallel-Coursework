#define _USE_MATH_DEFINES

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <fstream>
#include <chrono>
#include <math.h>

using namespace std;
using namespace std::chrono;

#define BLOCK_SIZE 1024
#define EPS 3e4f // Softening factor
#define G 6.673e-11f // Gravitational constant

// Output file
ofstream dataFileOutput("data.csv", ofstream::out);

// Number of bodies
const int N = 5000;

const float solarmass = 1.98892e30f;

// structure of a body (particle)
struct Body
{
	float rx, ry;//cartesian positions
	float vx, vy;//velocity components
	float fx, fy;//force components
	float mass;//mass
};

// convert to string representation formatted nicely
void PrintBody(Body body)
{
	printf("rx: %f, ry: %f, vx: %f, vy:%f, mass:%f \n", body.rx, body.ry, body.vx, body.vy, body.mass);
}

// compute the net force acting between the b a and b, and add to the net force acting on a
__global__ void bodyForce(Body* a, Body* b, int n, float dt)
{
	// Get block index
	auto block_idx = blockIdx.x;
	// Get thread index
	auto thread_idx = threadIdx.x;
	// Get the number of threads per block
	auto block_dim = blockDim.x;
	// Get the thread's unique ID
	auto idx = (block_idx * block_dim) + thread_idx;

	if (idx < n)
	{
		// set force to 0
		auto fx = 0.0f;
		auto fy = 0.0f;
		for (auto i = 0; i < n; ++i)
		{
			if (idx != i)
			{
				auto dx = a[i].rx - a[idx].rx;
				auto dy = a[i].ry - a[idx].ry;
				auto distance = sqrt(dx * dx + dy * dy);

				auto F = (G * a[idx].mass * a[i].mass) / (distance * distance + EPS * EPS);
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

// Random number generator
float randomGenerator(float min, float max)
{
	return ((float(rand()) / float(RAND_MAX)) * (max - min)) + min;
}

// bodies are initliased in circular orbits around the central mass. This is the physics to do that
float circlev(float rx, float ry)
{
	auto r2 = sqrt(rx * rx + ry * ry);
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
void startTheBodies(Body* bodies)
{
	bodies[0].rx = 0.0f;
	bodies[0].ry = 0.0f;
	bodies[0].vx = 0.0f;
	bodies[0].vy = 0.0f;
	bodies[0].mass = solarmass;

	for (auto i = 0; i < N; ++i)
	{
		float px = 1e18 * exp(-1.8) * (.5 - randomGenerator(0.0, 1.0));
		float py = 1e18 * exp(-1.8) * (.5 - randomGenerator(0.0, 1.0));
		auto magv = circlev(px, py);

		auto absangle = atan(abs(py / px));
		float thetav = M_PI / 2 - absangle;
		auto vx = -1 * signum(py) * cos(thetav) * magv;
		auto vy = signum(px) * sin(thetav) * magv;

		// Orientate a random 2D cirular orbit
		if (randomGenerator(0.0, 1.0) <= 0.5)
		{
			vx = -vx;
			vy = -vy;
		}

		// Calculate mass
		float mass = solarmass * randomGenerator(0.0, 1.0) * 10 + 1e20;
		// Assign variables to a body struct
		bodies[i].rx = px;
		bodies[i].ry = py;
		bodies[i].vx = vx;
		bodies[i].vy = vy;
		bodies[i].mass = mass;
	}
};

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
	cout << "Clock Freq: " << properties.clockRate / 1000 << "MHz \n" << endl;
}

int main()
{
	// Initialise CUDA - select device
	cudaSetDevice(0);
	cudaInfo();

	auto dt = 0.01f;

	// Host Buffers
	Body* buffer_host_A;
	Body* buffer_host_B;

	// Device Buffers
	Body* buffer_Device_A;
	Body* buffer_Device_B;

	// host memory size
	auto data_size = sizeof(Body) * N;

	// Allocate memory for each struct on host
	buffer_host_A = static_cast<Body*>(malloc(data_size));
	buffer_host_B = static_cast<Body*>(malloc(data_size));

	// Allocate memory for each struct on GPU
	cudaMalloc(&buffer_Device_A, data_size);
	cudaMalloc(&buffer_Device_B, data_size);

	auto start = system_clock::now();

	startTheBodies(buffer_host_A);

	cudaMemcpy(buffer_Device_A, buffer_host_A, data_size, cudaMemcpyHostToDevice);

	// Number of thread blocks in grid
	auto gridSize = static_cast<int>(ceil(static_cast<float>(N) / BLOCK_SIZE));

	// Execute the kernel
	bodyForce <<<gridSize, BLOCK_SIZE >>>(buffer_Device_A, buffer_Device_B, N, dt);

	// Copy to host
	cudaMemcpy(buffer_host_B, buffer_Device_B, data_size, cudaMemcpyDeviceToHost);

	// Release device memory
	cudaFree(buffer_Device_A);
	cudaFree(buffer_Device_B);

	// Release host memory
	free(buffer_host_A);
	free(buffer_host_B);

	auto end = system_clock::now();
	auto total = end - start;
	cout << "Number of Bodies = " << N << endl;
	cout << "Main Application time = " << duration_cast<milliseconds>(total).count() << "ms" << endl;
	dataFileOutput << duration_cast<milliseconds>(total).count() << endl;

	return 0;
}


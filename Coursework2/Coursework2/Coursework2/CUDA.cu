#define _USE_MATH_DEFINES

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <fstream>
#include <chrono>
#include <math.h>
#include <sstream>
#include "FreeImage/FreeImage.h"

using namespace std;
using namespace std::chrono;

// Memory block size
#define BLOCK_SIZE 1024

#define EPS 3e4f // Softening factor
#define G 6.673e-11f // Gravitational constant

// Output file
ofstream dataFileOutput("data.csv", ofstream::out);
// Number of bodies
const int N = 100;
// Number of iterations
const int iterations = 100;

// N-Body constants
#define G 6.673e-11f; // Gravitational Constant
const float radiusOfUniverse = 1000.0f;

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

// Random number generator
float randomGenerator(float min, float max)
{
	return ((float(rand()) / float(RAND_MAX)) * (max - min)) + min;
}

// compute the net force acting between bodies
__global__ void bodyForce(Body* bodies, int n, float dt)
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
				// Calcululate new forces
				auto dx = bodies[i].rx - bodies[idx].rx;
				auto dy = bodies[i].ry - bodies[idx].ry;
				auto distsqr = dx * dx + dy * dy + 1e-9f; //softening
				auto distSixth = distsqr * distsqr;
				auto invDist = 1.0f / sqrtf(distSixth);
				auto F = bodies[i].mass * invDist;
				fx += F * dx;
				fy += F * dy;
			}
		}
		bodies[idx].vx += dt * fx;
		bodies[idx].vy += dt * fy;
	}

}

// Calculate new position (based on new velocity)
__global__ void calculatePosition(Body *a, int n, float dt){

	// Get block index
	auto block_idx = blockIdx.x;
	// Get thread index
	auto thread_idx = threadIdx.x;
	// Get the number of threads per block
	auto block_dim = blockDim.x;
	// Get the thread's unique ID
	auto idx = (block_idx * block_dim) + thread_idx;

	if (idx < n){
		a[idx].rx += a[idx].vx * dt;
		a[idx].ry += a[idx].vy * dt;
	}
}

// Initialise N bodies with random positions and circular velocities
void startTheBodies(Body* bodies)
{
	for (auto i = 0; i < N; ++i)
	{
		auto px = randomGenerator(0, radiusOfUniverse);
		auto py = randomGenerator(0, radiusOfUniverse);
		auto m = randomGenerator(0, 10e4);
		bodies[i].rx = px;
		bodies[i].ry = py;
		bodies[i].vx = 0.0f;
		bodies[i].vy = 0.0f;
		bodies[i].mass = m;
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

void drawImage(Body bodies[N], int name)
{
	FreeImage_Initialise();
	auto bitmap = FreeImage_Allocate(radiusOfUniverse, radiusOfUniverse, 24);
	RGBQUAD color;

	for (auto i = 0; i < N; i++)
	{
		color.rgbGreen = 255;
		color.rgbBlue = 255;
		color.rgbRed = 255;
		FreeImage_SetPixelColor(bitmap, bodies[i].rx, bodies[i].ry, &color);
	}

	// Creates a numbered file name
	stringstream fileName;
	fileName << name << "test.png";
	char file[1024];
	strcpy(file, fileName.str().c_str());

	// Save the file
	if (FreeImage_Save(FIF_PNG, bitmap, file, 0))
	{
		cout << "Image saved - " << fileName.str() << endl;
	}

	FreeImage_DeInitialise();
}

int main()
{
	// Random Seed
	srand(time(nullptr));

	// Initialise CUDA - select device
	cudaSetDevice(0);
	cudaInfo();

	// Timestamp
	auto dt = 0.01f;

	// Host Buffer
	Body* buffer_host_A;

	// Device Buffer
	Body* buffer_Device_A;

	// host memory size
	auto data_size = sizeof(Body) * N;

	// Allocate memory for each struct on host
	buffer_host_A = static_cast<Body*>(malloc(data_size));

	// Allocate memory for each struct on GPU
	cudaMalloc(&buffer_Device_A, data_size);

	// Number of thread blocks in grid
	auto gridSize = static_cast<int>(ceil(static_cast<float>(N) / BLOCK_SIZE));

	//auto start = system_clock::now();

	// Initiliase the universe
	startTheBodies(buffer_host_A);
	for (auto i = 0; i < iterations; ++i)
	{
		cudaMemcpy(buffer_Device_A, buffer_host_A, data_size, cudaMemcpyHostToDevice);
		// Execute kernels
		bodyForce << <gridSize, BLOCK_SIZE >> >(buffer_Device_A, N, dt);
		calculatePosition << <gridSize, BLOCK_SIZE >> >(buffer_Device_A, N, dt);

		// Copy to host
		cudaMemcpy(buffer_host_A, buffer_Device_A, data_size, cudaMemcpyDeviceToHost);

		drawImage(buffer_host_A, i);
	}
	
	for (auto i = 0; i < N; ++i)
	{
		PrintBody(buffer_host_A[i]);
	}

	// Release device memory
	cudaFree(buffer_Device_A);

	// Release host memory
	free(buffer_host_A);
	//auto end = system_clock::now();
	//auto total = end - start;
	//cout << "Number of Bodies = " << N << endl;
	//cout << "Main Application time = " << duration_cast<milliseconds>(total).count() << "ms" << endl;
	//dataFileOutput << duration_cast<milliseconds>(total).count() << endl;

	return 0;
}


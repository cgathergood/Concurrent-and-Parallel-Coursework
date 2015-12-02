#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>

using namespace std;

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

	return 0;
}
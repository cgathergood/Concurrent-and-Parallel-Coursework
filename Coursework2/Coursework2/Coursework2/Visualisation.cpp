#define _USE_MATH_DEFINES

#include <iostream>
#include <fstream>
#include <chrono>
#include <math.h>
#include <sstream>
#include "FreeImage/FreeImage.h"

using namespace std;
using namespace std::chrono;

// Output file
ofstream dataFileOutput("data.csv", ofstream::out);
// Number of bodies
const int N = 200;
// Number of iterations
const int iterations = 100;

// N-Body constants
const float G = 6.673e-11; // Gravitational Constant
const float radiusOfUniverse = 1000.0f;

// create and initiliase a new body
struct Body
{
	double rx, ry;//cartesian positions
	double vx, vy;//velocity components
	double fx, fy;//force components
	double mass;//mass
};

// convert to string representation formatted nicely
string PrintBody(Body body)
{
	stringstream ss;
	ss << "rx:" << body.rx << ", ry:" << body.ry << ", vx:" << body.vx << ", vy:" << body.vy << ", mass:" << body.mass << ", fx:" << body.fx << ", fy:" << body.fy << "\n";
	return ss.str();
}

// Random number generator
float randomGenerator(float min, float max)
{
	return ((float(rand()) / float(RAND_MAX)) * (max - min)) + min;
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
}

void updateForces(Body* bodies, float dt)
{
	for (auto i = 0; i < N; ++i)
	{
		// Reset Forces
		auto fx = 0.0f;
		auto fy = 0.0f;

		for (auto j = 0; j < N; ++j)
		{
			if (i != j)
			{
				// Calcululate new forces
				float dx = bodies[j].rx - bodies[i].rx;
				float dy = bodies[j].ry - bodies[i].ry;
				auto distsqr = dx * dx + dy * dy + 1e-9f; //softening
				auto distSixth = distsqr * distsqr;
				auto invDist = 1.0f / sqrtf(distSixth);
				float F = bodies[j].mass * invDist;
				fx += F * dx;
				fy += F * dy;
			}
		}
		// Calculate new velcocity (based on new force)
		bodies[i].vx += dt * fx;
		bodies[i].vy += dt * fy;
	}
	// Calculate new position (based on new velocity)
	for (auto i = 0; i < N; ++i)
	{
		bodies[i].rx += bodies[i].vx * dt;
		bodies[i].ry += bodies[i].vy * dt;
	}
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

	//Time Stamp
	auto dt = 0.01f;
	// Collection of bodies (particles)
	Body* universe = new Body[N];

	auto start = system_clock::now();
	// set up simulation
	startTheBodies(universe);
	// Iterate and update the forces
	for (auto i = 0; i < iterations; ++i)
	{
		updateForces(universe, dt);
		//PrintBody(universe[i]);
		drawImage(universe, i);
	}

	auto end = system_clock::now();
	auto total = end - start;
	//cout << "Number of Bodies = " << N << endl;
	//cout << "Main Application time = " << duration_cast<milliseconds>(total).count() << "ms" << endl;
	dataFileOutput << duration_cast<milliseconds>(total).count() << endl;

	return 0;
}
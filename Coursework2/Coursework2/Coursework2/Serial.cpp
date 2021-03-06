#define _USE_MATH_DEFINES

#include <iostream>
#include <fstream>
#include <chrono>
#include <math.h>
#include <sstream>

using namespace std;
using namespace std::chrono;

// Output file
ofstream dataFileOutput("data.csv", ofstream::out);
// Number of bodies
const int N = 500;
// Number of iterations
const int iterations = 1000;

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

int main()
{
	for (auto testing = 0; testing < 10; ++testing)
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
			//drawImage(universe, i);
		}

		auto end = system_clock::now();
		auto total = end - start;
		//cout << "Number of Bodies = " << N << endl;
		//cout << "Main Application time = " << duration_cast<milliseconds>(total).count() << "ms" << endl;
		//drawImage(bodies);
		dataFileOutput << duration_cast<milliseconds>(total).count() << endl;
	}
	return 0;
}


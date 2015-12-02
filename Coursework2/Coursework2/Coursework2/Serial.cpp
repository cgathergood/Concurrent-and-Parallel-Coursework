#define _USE_MATH_DEFINES

#include <iostream>
#include <fstream>
#include <chrono>
#include <math.h>

using namespace std;
using namespace std::chrono;

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
void bodyForce(Body* bodies, float dt)
{
	for (int i = 0; i < N; ++i)
	{
		// set force to 0
		auto fx = 0.0f;
		auto fy = 0.0f;

		for (int j = 0; j < N; ++j)
		{
			if (i != j)
			{
				auto dx = bodies[i].rx - bodies[j].rx;
				auto dy = bodies[i].ry - bodies[j].ry;
				auto distance = sqrt(dx * dx + dy * dy);

				auto F = (G * bodies[j].mass * bodies[i].mass) / (distance * distance + EPS * EPS);
				fx += F * dx / distance;
				fy += F * dy / distance;
			}
		}

		bodies[i].vx += dt * fx / bodies[i].mass;
		bodies[i].vy += dt * fy / bodies[i].mass;
		bodies[i].rx += dt * bodies[i].vx;
		bodies[i].ry += dt * bodies[i].vy;
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

int main()
{
	auto dt = 0.01f;
	Body *bodies = new Body[N];

	auto start = system_clock::now();
	startTheBodies(bodies);
	bodyForce(bodies, dt);
	auto end = system_clock::now();
	auto total = end - start;

	cout << "Number of Bodies = " << N << endl;
	cout << "Main Application time = " << duration_cast<milliseconds>(total).count() << "ms" << endl;
	dataFileOutput << duration_cast<milliseconds>(total).count() << endl;

	return 0;
}
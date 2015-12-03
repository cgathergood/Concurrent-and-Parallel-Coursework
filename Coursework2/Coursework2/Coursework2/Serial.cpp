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
const int N = 1000;

// N-Body constants
const double G = 6.673e-11; // Gravitational Constant
const double solarmass = 1.98892e30;
const double radiusOfUniverse = 1e18;

// create and initiliase a new body
struct Body
{
	double rx, ry;//cartesian positions
	double vx, vy;//velocity components
	double fx, fy;//force components
	double mass;//mass
};

// Global Bodies
Body bodies[N];

// convert to string representation formatted nicely
string PrintBody(Body body)
{
	stringstream ss;
	ss << "rx:" << body.rx << ", ry:" << body.ry << ", vx:" << body.vx << ", vy:" << body.vy << ", mass:" << body.mass << "\n";
	return ss.str();
}

// update velocity and position using timestamp dt
Body Update(Body body, double dt)
{
	body.vx += dt * body.fx / body.mass;
	body.vy += dt * body.fy / body.mass;
	body.rx += dt * body.vx;
	body.ry += dt * body.vy;
	return body;
}

// returns distace between two bodies
double distanceTo(Body a, Body b)
{
	double dx = a.rx - b.rx;
	double dy = a.ry - b.ry;
	return sqrt(dx * dx + dy * dy);
}

// set force to 0 for the next iteration
Body ResetForce(Body body)
{
	body.fx = 0.0;
	body.fy = 0.0;
	return body;
}

// compute the net force acting between the body a and b, and add to the net force acting on a
Body AddForce(Body a, Body b)
{
	double EPS = 3E4; // softening parameter (just to avoid infinities)
	double dx = b.rx - a.rx;
	double dy = b.ry - a.ry;
	double dist = sqrt(dx * dx + dy * dy);
	double F = (G * a.mass * b.mass) / (dist * dist + EPS * EPS);
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
double circlev(double rx, double ry)
{
	double r2 = sqrt(rx * rx + ry * ry);
	double numerator = (6.67e-11) * 1e6 * solarmass;
	return sqrt(numerator / r2);
}

// template for the signum function contained within Java's Math library
template <typename T>
int signum(T val)
{
	return (T(0) < val) - (val < T(0));
}

// Initialise N bodies with random positions and circular velocities
void startTheBodies()
{
	for (int i = 0; i < N; ++i)
	{
		double px = 1e18 * exp(-1.8) * (.5 - randomGenerator(0.0, 1.0));
		double py = 1e18 * exp(-1.8) * (.5 - randomGenerator(0.0, 1.0));
		double magv = circlev(px, py);

		double absangle = atan(abs(py / px));
		double thetav = M_PI / 2 - absangle;
		double vx = -1 * signum(py) * cos(thetav) * magv;
		double vy = signum(px) * sin(thetav) * magv;

		// Orientate a random 2D cirular orbit
		if (randomGenerator(0.0, 1.0) <= 0.5)
		{
			vx = -vx;
			vy = -vy;
		}

		// Calculate mass
		double mass = solarmass * randomGenerator(0.0, 1.0) * +1e20;
		// Assign variables to a body struct
		bodies[i].rx = px;
		bodies[i].ry = py;
		bodies[i].vx = vx;
		bodies[i].vy = vy;
		bodies[i].mass = mass;
	}
};

void addForces()
{
	for (int i = 0; i < N; ++i)
	{
		bodies[i] = ResetForce(bodies[i]);
		for (int j = 0; j < N; ++j)
		{
			if (i != j)
			{
				bodies[i] = AddForce(bodies[i], bodies[j]);
			}
		}
	}

	// Loop again, update bodies with timestamp
	for (int i = 0; i < N; ++i)
	{
		bodies[i] = Update(bodies[i], 1e11);
	}
}

void drawImage(Body bodies[N], int name)
{
	FreeImage_Initialise();
	FIBITMAP* bitmap = FreeImage_Allocate(N*2, N*2, 24);
	RGBQUAD color;

	for (int i = 0; i < N; i++)
	{
		color.rgbGreen = 255;
		color.rgbBlue = 255;
		color.rgbRed = 255;
		FreeImage_SetPixelColor(bitmap, round(bodies[i].rx * 10 / 1e30+N), round(bodies[i].ry * 10 / 1e30+N), &color);
	}

	// Creates a numbered file name
	stringstream fileName;
	fileName << name << "test.png";
	cout << fileName.str() << endl;
	char file[1024];
	strcpy(file, fileName.str().c_str());

	// Save the file
	if (FreeImage_Save(FIF_PNG, bitmap, file, 0))
	{
		cout << "Image successfully saved!" << endl;
	}

	FreeImage_DeInitialise();
}

int main()
{
	auto start = system_clock::now();
	startTheBodies();

	for (int i = 0; i < 100; ++i)
	{
		addForces();
		if (i%10 == 0)
		{
			drawImage(bodies, i);
		}
	}
	

	auto end = system_clock::now();
	auto total = end - start;
	cout << "Number of Bodies = " << N << endl;
	cout << "Main Application time = " << duration_cast<milliseconds>(total).count() << "ms" << endl;
	//drawImage(bodies);
	//dataFileOutput << duration_cast<milliseconds>(total).count() << endl;
	return 0;
}

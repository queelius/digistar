#ifndef PARTS_H
#define PARTS_H

#include <vector>
#include <limits>
#include "glut.h"
#include "glColor.h"
#include "timer.h"
using namespace std;

namespace {
    const GLfloat GAS_CONSTANT = 8.314472f;
};

GLfloat defaultLeakFunc(GLfloat capacity, GLfloat amount, GLfloat coeff);

class Jet {
private:
    // force impulses per second

    // force = mass (kg) * acceleration (meters * sec^-2)
    // work (joules) = force * distance (meters)

    GLfloat theta;

    GLfloat tank; // L (liters)
    GLfloat fuel; // L (liters)
};

class Tube {
public:
private:
};

class Container {
// (pressure)*(volume) = (moles)*R*T
public:
    GLfloat getPressure() {
        return (moles * GAS_CONSTANT * getTemperature() / volume);
    };

    unsigned int getMoles() {
        return moles;
    };

    GLfloat getVolume() {
        return volume;
    };

    GLfloat getTemperature() {
        return temperature;
    };

private:
    unsigned int moles; // number of particles
    double temperature; // average particle momentum
    double volume; // volume of container
};


class Tank: public Container {
public:
    Tank(GLfloat capacity, GLfloat amount, GLfloat (*leakFunc)(GLfloat, GLfloat, GLfloat) = NULL) {
        this->capacity = capacity;
        this->amount = amount;

        if (leakFunc)
            this->leakFunc = leakFunc;
        else
            this->leakFunc = defaultLeakFunc;
    };

    void plug() {
        rateCoeff = 0.0f;
    };

    void unplug() {
        rateCoeff = 1.0f;
    };

    void update() {
        amount = leakFunc(capacity, amount, theta);
    };

private:
    GLfloat (*leakFunc)(GLfloat capacity, GLfloat amount, GLfloat coeff);

    GLfloat capacity;
    GLfloat amount;
    GLfloat rateCoeff;

    GLfloat theta;
};

// connect parts to other parts
class Connector {
};

class Spring {
};

GLfloat defaultLeakFunc(GLfloat capacity, GLfloat amount, GLfloat coeff) {

}

#endif
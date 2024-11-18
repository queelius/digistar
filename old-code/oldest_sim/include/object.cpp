#include "object.h"
using namespace std;

const GLfloat *Object::getPosition() const {
    return pos;
}

const GLfloat *Object::getVelocity() const {
    return vel;
}

GLfloat Object::getRadius() const {
    return radius;
}

bool Object::isExpired() const {
    return state == Expire;
}

GLfloat Object::getSpeed() const {
    return sqrt(vel[0] * vel[0] + vel[1] * vel[1]);
}

State Object::getState() const {
    return state;
}

GLfloat Object::getAngle() const {
    return ang;
}

GLfloat Object::getMass() const {
    return mass;
}

GLuint Object::getAge() const {
    return getTime() - t0;
}

GLfloat Object::getAngularVelocity() const {
    return angVel;
}

const glColor &Object::getColor() const {
    return color;
}

bool Object::isFixed() const {
    return fixed;
}

// mutators
void Object::setPosition(GLfloat pos[2]) {
    this->pos[0] = pos[0];
    this->pos[1] = pos[1];
}

void Object::setAngularVelocity(GLfloat velocity) {
    this->angVel = velocity;
}

void Object::setVelocity(GLfloat vel[2]) {
    this->vel[0] = vel[0];
    this->vel[1] = vel[1];
}

void Object::setRadius(GLfloat radius) {
    this->radius = radius;
}

void Object::setMass(GLfloat mass) {
    this->mass = mass;
}

void Object::setFixed(bool fixed) {
    this->fixed = fixed;
}

// force applications
void Object::transForce(GLfloat theta, GLfloat mag) {
    if (fixed) {
        // store total energy
    }
    else {
        netForce[0] += abs(mag) * cos(theta);
        netForce[1] += abs(mag) * sin(theta);
    }
}

void Object::torqueForce(GLfloat F) {
    if (fixed) {
        // store total energy
    }
    else
        netTorque += F;
}

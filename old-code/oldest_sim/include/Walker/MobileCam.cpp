////////////////////////////////////////////////////
// MobileCam.cpp - MobileCam Class Implementation //
////////////////////////////////////////////////////

#include "MobileCam.h"

void MobileCam::init(const GLfloat pos[3], GLfloat accel, GLfloat theta, GLfloat thetaAccel, GLfloat friction) {
    thetaVel = vel = 0.0f;

    this->theta      = theta;
    this->thetaAccel = thetaAccel;
    this->pos[0]     = pos[0];
    this->pos[1]     = pos[1]; // not used -- only x and z used
    this->pos[2]     = pos[2];
    this->accel      = accel;
    this->friction   = friction;
}

void MobileCam::reset() {
    // do nothing yet
}

void MobileCam::update() {
    // rate of change of the angles determined by their respective angular velocities
    theta += thetaVel;

    if (theta >= TWO_PI) { theta -= TWO_PI; }
    if (theta <= 0.0f)   { theta += TWO_PI; }

    pos[0] += cos(theta) * vel;
    pos[2] += sin(theta) * vel;

    // apply friction to angular velocities
    thetaVel /= friction;
    vel      /= friction;
}

void MobileCam::orient() {
    gluLookAt(pos[0], pos[1], pos[2], pos[0] + cos(theta), pos[1], pos[2] + sin(theta), 0, 1, 0);
}

void MobileCam::accelerateForward() {
    vel += accel;
}

void MobileCam::accelerateBackward() {
    vel -= accel;
}

void MobileCam::accelerateLeft() {
    thetaVel += thetaAccel;
}

void MobileCam::accelerateRight() {
    thetaVel -= thetaAccel;
}
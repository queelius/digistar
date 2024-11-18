////////////////////////////////////////////////////
// SphereCam.cpp - SphereCam Class Implementation //
////////////////////////////////////////////////////

#include "SphereCam.h"

SphereCam::SphereCam() {}

void SphereCam::init(const GLfloat focal[3], GLfloat distance, GLfloat theta, GLfloat phi, GLfloat accel, GLfloat angularAccel, GLfloat friction) {
    thetaVel = phiVel = vel = 0;

    this->theta        = theta;
    this->phi          = phi;
    this->angularAccel = angularAccel;
    this->accel        = accel;
    this->friction     = friction;
    this->distance     = distance;
    this->focal[0]     = focal[0];
    this->focal[1]     = focal[1]; 
    this->focal[2]     = focal[2];
}

void SphereCam::lookAt() {
    gluLookAt(focal[0] + distance * sin(phi) * sin(theta), 
              focal[1] + distance * cos(theta),
              focal[2] + distance * cos(phi) * sin(theta),
              focal[0],
              focal[1],
              focal[2],
              0.0, 1.0, 0.0);
}

void SphereCam::reset() {
    // do nothing yet
}

void SphereCam::update() {
    // rate of change of the angles determined by their respective angular velocities
    theta    += thetaVel;
    phi      += phiVel;

    // rate of chance of the distance from focus is determined by the velocity
    distance += vel;

    if (theta >= PI)  { thetaVel = 0.0; theta = PI - EPSILON; }
    if (theta <= 0)   { thetaVel = 0.0; theta = EPSILON; }
    if (phi > TWO_PI) { phi -= TWO_PI; }
    if (phi < 0.0)    { phi += TWO_PI; }

    // apply friction to angular velocities
    phiVel   /= friction;
    thetaVel /= friction;
    vel      /= friction;
}

void SphereCam::update(GLuint key) {
    switch (key) { // increase / decrease angular velocities
        case GLUT_KEY_UP:    {  thetaVel += angularAccel; break; }
        case GLUT_KEY_DOWN:  {  thetaVel -= angularAccel; break; }
		case GLUT_KEY_RIGHT: {  phiVel   += angularAccel; break; }
        case GLUT_KEY_LEFT:  {  phiVel   -= angularAccel; break; }
    }
}

void SphereCam::increaseVel() {
    vel += accel;
}

void SphereCam::decreaseVel() {
    vel -= accel;
}
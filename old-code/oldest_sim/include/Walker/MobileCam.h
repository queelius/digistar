//////////////////////////////////////////////
// MobileCam.h - MobileCam Class Definition //
//////////////////////////////////////////////

#ifndef MOBILE_CAM_H
#define MOBILE_CAM_H

#include <cmath>
#include <iostream>
#include "glut.h"
#include "Globals.h"

class MobileCam {
public:
    void init(const GLfloat pos[3], GLfloat accel, GLfloat theta, GLfloat thetaAccel, GLfloat friction);
    void reset();
    void update();
    void orient();
    void accelerateForward();
    void accelerateBackward();
    void accelerateLeft();
    void accelerateRight();

    GLfloat theta;
    GLfloat thetaVel;
    GLfloat thetaAccel;
    GLfloat pos[3];
    GLfloat vel;
    GLfloat accel;
    GLfloat friction;
};

#endif
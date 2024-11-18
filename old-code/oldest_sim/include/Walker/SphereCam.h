//////////////////////////////////////////////
// SphereCam.h - SphereCam Class Definition //
//////////////////////////////////////////////

#ifndef SPHERE_CAM_H
#define SPHERE_CAM_H

#include <cmath>
#include <iostream>
#include "glut.h"
#include "Globals.h"

class SphereCam {
public:
	SphereCam();
    void init(const GLfloat focal[3], GLfloat distance, GLfloat theta, GLfloat phi, GLfloat accel, GLfloat angularAccel, GLfloat friction);
    void reset();

    void lookAt();
    void update(GLuint key);
    void update();

    void increaseVel();
    void decreaseVel();

    GLfloat focal[3];     // focal point
    GLfloat distance;     // camera distance from focal point
    GLfloat theta,        // left and right camera angle
            phi;          // up and down camera angle

    GLfloat thetaVel,     // left and right angular velocity
            phiVel;       // up and down angular velocity

    GLfloat angularAccel; // angular acceleration

    GLfloat vel;          // velocity
    GLfloat accel;        // acceleration

    GLfloat friction;     // slow-down factor
};

#endif
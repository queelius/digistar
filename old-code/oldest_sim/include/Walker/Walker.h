////////////////////////////////////////
// Walker.h - Walker Class Definition //
////////////////////////////////////////

#ifndef WALKER_H
#define WALKER_H

#include <cmath>
#include <iostream>
#include "glut.h"
#include "Globals.h"
#include "Joint.h"
#include "CrossJoint.h"

class Walker {
public:
	Walker();                             // default constructor
	void draw();                          // draw walker to screen
	void update(float animFactor = 1.0f); // update joint positions
    void reset();                         // set walker's parameters to default values

    CrossJoint Shoulders;
	CrossJoint Hips;
    Joint      TorsoAndHead;
    Joint      LeftArm[3];         // 0=Humerus, 1=Forearm, 2=Hand
    Joint      RightArm[3];
	Joint      LeftLeg[4];         // 0=Thigh, 1=Shin, 2=Heel, 3=Toe
	Joint      RightLeg[4];

    GLfloat    pos[3];             // position (x, y, z)
    GLfloat    v[3];               // velocity
    GLfloat    a[3];               // acceleration
    GLfloat    origin[3];          // point to uniformly accelerate around
    GLfloat    theta;              // angle walker is pointed at
};

#endif

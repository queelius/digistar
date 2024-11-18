#ifndef GLOBALS_H
#define GLOBALS_H

#include <iostream>
#include <vector>
#include <string>
#include <sstream>
#include <cmath>
#include "glut.h"

const GLfloat PI = 3.14159265;
const GLfloat TWO_PI = 2*PI;
const GLfloat HALF_PI = PI/2.0;
const GLfloat GRAVITY_CONSTANT = 0.00075;

const unsigned int UPDATES_PER_SEC = 40;
const GLfloat      SEC_PER_UPDATE = 1 / UPDATES_PER_SEC;
const unsigned int MSEC_PER_UPDATE = 1000 / UPDATES_PER_SEC;

#endif
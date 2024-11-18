///////////////////////////////////////////////
//    Filename: Common.h                     //
//        Name: Alex Towell                  //
// Description: Common functions reside here //
///////////////////////////////////////////////

#ifndef COMMON_H
#define COMMON_H

#include <ctime>
#include <cstdlib>
#include "glut.h"
#include "RGBpixmap.h"
#include "Globals.h"

void    seed();
GLint   getRandInt(GLint low, GLint high);
GLfloat getRand(GLfloat low, GLfloat high);

template <class T> void cp(T *dest, const T *src, size_t size);

#endif
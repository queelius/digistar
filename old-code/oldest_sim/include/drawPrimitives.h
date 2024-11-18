#ifndef DRAW_PRIMITIVES_H
#define DRAW_PRIMITIVES_H

#include "globals.h"
#include "glColor.h"

void drawCircle(const GLfloat pos[2], GLfloat radius, GLfloat slices, const glColor &color);
void drawTriangle(const GLfloat pos[2], GLfloat base, GLfloat height, GLfloat angle, const glColor &color);
void drawRectangle(const GLfloat pos[2], GLfloat width, GLfloat height, GLfloat angle, const glColor &color);

#endif
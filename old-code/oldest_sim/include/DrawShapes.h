#ifndef DRAW_SHAPES_H
#define DRAW_SHAPES_H

#include "Globals.h"
#include "VectorSpace.h"

namespace PhysicsTest {

void DrawCircle(const Point &pt, GLfloat radius, const GLfloat color[2], GLuint slices = 0);
void DrawTriangle(const Point &pt, GLfloat base, GLfloat height, GLfloat angle, const GLfloat color[2]);
void DrawRectangle(const Point &pt, GLfloat width, GLfloat height, GLfloat angle, const GLfloat color[2]);

}

#endif
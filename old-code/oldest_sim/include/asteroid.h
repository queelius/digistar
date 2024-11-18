#ifndef ASTEROID_H
#define ASTEROID_H

#include "globals.h"
#include "object.h"
#include "drawPrimitives.h"

extern std::vector<Object*> o;

class Asteroid: public Object {
    friend Object;
public:
    Asteroid(const GLfloat initialPos[2], GLfloat initialAngle, GLfloat initialMass, glColor intialColor, const GLfloat initialVel[2] = NULL, const GLfloat vertices[5][2] = NULL, GLfloat radius = 10);
    ObjectType what() const;

    void update();
    void draw() const;

private:
    GLfloat vertices[5][2];
};

#endif
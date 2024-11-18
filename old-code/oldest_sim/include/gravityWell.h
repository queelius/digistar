#ifndef GRAVITY_WELL_H
#define GRAVITY_WELL_H

#include "globals.h"
#include "object.h"

extern std::vector<Object*> o;

class GravityWell: public Object {
    friend Object;
public:
    GravityWell(const GLfloat initialPos[2], GLfloat initialMass, const GLfloat initialVel[2] = NULL, bool fixed = true, GLfloat radius = 120);
    ObjectType what() const;
    void update();
    void draw() const;

private:
    GLint slices;
};

#endif
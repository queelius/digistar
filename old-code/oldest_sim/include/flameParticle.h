#ifndef FLAME_PARTICLE_H
#define FLAME_PARTICLE_H

#include "globals.h"
#include "object.h"

extern std::list<Object*> o;

class FlameParticle: public Object {
    friend Object;
public:
    FlameParticle(const GLfloat initialPos[2], GLfloat initialAngle, GLfloat initialMass, const GLfloat intialColor[3], const GLfloat initialVel[2], GLfloat radius = 10);
    ObjectType what() const;

    void update();
    void draw() const;
};

#endif
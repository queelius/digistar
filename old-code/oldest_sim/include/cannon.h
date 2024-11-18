#ifndef CANNON_H
#define CANNON_H

#include "object.h"
#include <vector>

extern std::vector<Object*> o;
extern std::vector<Object*>::const_iterator selObject;

class Spaceship: public Object {
    friend Object;
public:
    Spaceship(const GLfloat initialPos[2], GLfloat initialAngle, GLfloat initialMass, glColor intialColor, const GLfloat initialVel[2] = NULL, GLfloat radius = 10);
    ObjectType what() const;

    void update();

    void shoot(GLfloat initialRelativeVelocity = 15.0);
    void shootMissile(GLfloat initialRelativeVelocity = 0.0);
    void afterBurn();

    void go();
    void setTarget(Object *o);
    void orbit(Object *o, GLfloat radius);

    void draw() const;

private:
    Object *target;
};

#endif
#ifndef PROJECTILE_H
#define PROJECTILE_H

#include "object.h"

extern std::vector<Object*> o;

class Projectile: public Object {
    friend Object;
public:
    Projectile(const GLfloat initialPos[2], GLfloat initialMass, glColor intialColor, const GLfloat initialVel[2], GLfloat initialRadius = 10.0);
    ObjectType what() const;
    void update();
    void draw() const;
};

class Missile: public Object {
    friend Object;
public:
    Missile(const GLfloat initialPos[2], GLfloat initialAngle, GLfloat initialMass, glColor initialColor,
            GLfloat propulsionForce, GLuint propulsionDuration, const GLfloat initialVel[2], GLfloat initialRadius = 3.0);
    ObjectType what() const;
    void update();
    void draw() const;

private:
    GLuint propulsionDuration;
    GLfloat propulsionForce;
};


#endif
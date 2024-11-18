#ifndef SPACESHIP_H
#define SPACESHIP_H

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

    void Spaceship::shoot(GLfloat initialRelativeVelocity, GLfloat initialMass, GLfloat initialRadius = NULL);
    void shootMissile(GLfloat initialRelativeVelocity = 0.0);

    void applyWork(GLfloat joules, const GLfloat vector[2]);
    void applyPower(GLfloat power, const GLfloat vector[2]); // power == anything (e.g., power from star --> light sail propulsion)

    bool setThrustVector(const GLfloat vector[2]);
    bool setThrustVector(GLfloat angle);
    void thrust(GLfloat power); // power <= maxPower
    void afterBurn(); // ?

    GLfloat getKineticEnergyRelativeTo(const Object *object);
    GLfloat getPotentialEnergyRelativeTo(const Object *object);
    GLfloat getMaxPower(); // a = F/m = P/(v*m)

    void go();
    void setTarget(Object *o);
    void orbit(Object *o, GLfloat radius);

    void draw() const;

private:
    Object *target;

    GLfloat maxPower;
};

#endif
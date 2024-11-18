#ifndef PLANETOID_H
#define PLANETOID_H

#include "Object.h"
#include "VectorSpace.h"
#include "DrawShapes.h"

namespace PhysicsTest {

class Planetoid: public Object {
public:
    Planetoid(const Point &position, const Vector &velocity, GLfloat radius, GLfloat mass, GLfloat angle, GLfloat angularVelocity, const GLfloat color[]) {
        this->angle = angle;
        this->angularVelocity = angularVelocity;
        this->position = position;
        this->mass = mass;
        this->velocity = velocity;
        this->radius = radius;
        this->age = 0;

        this->color[0] = color[0];
        this->color[1] = color[1];
        this->color[2] = color[2];

        this->netTorque = 0;
        this->netForce = Vector(0, 0);
        this->momentOfInertia = 0.4 * mass * radius * radius;
    };

    ObjectType what() const;
    void update();
    void draw() const;
    const char *id() const;
};

}

#endif
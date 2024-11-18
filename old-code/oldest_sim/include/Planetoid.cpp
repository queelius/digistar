#include "Planetoid.h"
#include <iostream>

namespace PhysicsTest {

ObjectType Planetoid::what() const {
    return PlanetoidType;
}

void Planetoid::update() {
    for (auto i = world.begin(); i != world.end(); ++i) {
        
    }

    velocity = velocity + netForce / mass;
    position = position + velocity;
    angularVelocity = angularVelocity + netTorque / momentOfInertia;
    angle = angle + angularVelocity;
}

void Planetoid::draw() const {
    DrawCircle(position, radius, color, 8);
}

const char *Planetoid::id() const {
    return "Placeholder";
}

}
#include "gravityWell.h"
#include "drawPrimitives.h"

GravityWell::GravityWell(const GLfloat initialPos[2], GLfloat initialMass, const GLfloat initialVel[2], bool fixed, GLfloat radius) {
    this->pos[0] = initialPos[0];
    this->pos[1] = initialPos[1];

    this->mass = initialMass;
    this->state = Resting;
    this->radius = radius;

    if (initialVel) {
        this->vel[0] = initialVel[0];
        this->vel[1] = initialVel[1];
    }
    else {
        this->vel[0] = 0;
        this->vel[1] = 0;
    }

    this->color = Yellow;

    this->netForce[0] = 0;
    this->netForce[1] = 0;

    this->netTorque = 0;
    this->fixed = fixed;
    this->slices = (GLint)(radius/4.0);

    this->t0 = getTime();
}

ObjectType GravityWell::what() const {
    return GravityWellType;
}

void GravityWell::update() {
    GLfloat tmp = GRAVITY_CONSTANT * mass;

    for (auto i = o.begin(); i != o.end(); ++i) {
        const GLfloat *p = (*i)->getPosition();
        GLfloat dx = pos[0] - p[0];
        GLfloat dy = pos[1] - p[1];
        GLfloat r2 = dx * dx + dy * dy;

        if (r2)
            (*i)->transForce(atan2(dy, dx), tmp * (*i)->getMass() / r2);
    }

    if (!fixed) {
        vel[0] += netForce[0] / mass;
        vel[1] += netForce[1] / mass;

        pos[0] += vel[0];
        pos[1] += vel[1];

        netForce[0] = 0;
        netForce[1] = 0;
    }
}

void GravityWell::draw() const {
    drawCircle(pos, radius, slices, color);
}

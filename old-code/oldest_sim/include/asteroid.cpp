#include "asteroid.h"

Asteroid::Asteroid(const GLfloat initialPos[2], GLfloat initialAngle, GLfloat initialMass, glColor initialColor, const GLfloat initialVel[2], const GLfloat vertices[5][2], GLfloat radius) {
    this->pos[0] = initialPos[0];
    this->pos[1] = initialPos[1];
    
    this->ang = initialAngle;
    this->mass = initialMass;
    this->radius = radius;
    this->color = initialColor;

    if (initialVel) {
        this->vel[0] = initialVel[0];
        this->vel[1] = initialVel[1];
    }
    else {
        this->vel[0] = 0;
        this->vel[1] = 0;
    }

    if (vertices) {
        for (unsigned int i = 0; i < 5; ++i) {
            this->vertices[i][0] = vertices[i][0];
            this->vertices[i][1] = vertices[i][1];
        }
    }
    else {
    }

    this->fixed = false;

    this->netForce[0] = 0;
    this->netForce[1] = 0;

    this->state = Resting;
    this->t0 = getTime();
}

ObjectType Asteroid::what() const {
    return AsteroidType;
}

void Asteroid::update() {
    vel[0] += netForce[0] / mass;
    vel[1] += netForce[1] / mass;

    pos[0] += vel[0];
    pos[1] += vel[1];

    netForce[0] = 0;
    netForce[1] = 0;
}

void Asteroid::draw() const {
    glPushMatrix();
        drawCircle(pos, radius, 5, color);
    glPopMatrix();
}
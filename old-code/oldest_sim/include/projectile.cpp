#include "projectile.h"
#include "drawPrimitives.h"

// projectile
Projectile::Projectile(const GLfloat initialPos[2], GLfloat initialMass, glColor initialColor, const GLfloat initialVel[2], GLfloat initialRadius) {
    this->pos[0] = initialPos[0];
    this->pos[1] = initialPos[1];
    this->mass = initialMass;
    this->color = initialColor;
    this->vel[0] = initialVel[0];
    this->vel[1] = initialVel[1];
    this->radius = initialRadius;
    this->fixed = false;
    this->netForce[0] = 0;
    this->netForce[1] = 0;
    this->state = Resting;
    this->t0 = getTime();
}

ObjectType Projectile::what() const {
    return ProjectileType;
}

void Projectile::update() {
    vel[0] += netForce[0] / mass;
    vel[1] += netForce[1] / mass;

    pos[0] += vel[0];
    pos[1] += vel[1];

    netForce[0] = 0;
    netForce[1] = 0;
}

void Projectile::draw() const {
    drawCircle(pos, radius, 6, color);
}

// missile
Missile::Missile(const GLfloat initialPos[2], GLfloat initialAngle, GLfloat initialMass, glColor initialColor,
                 GLfloat propulsionForce, GLuint propulsionDuration, const GLfloat initialVel[2], GLfloat initialRadius) {
    this->pos[0] = initialPos[0];
    this->pos[1] = initialPos[1];
    this->ang = initialAngle;
    this->mass = initialMass;
    this->color = initialColor;

    this->vel[0] = initialVel[0];
    this->vel[1] = initialVel[1];

    this->fixed = false;

    this->radius = initialRadius;
    this->fixed = false;
    this->netForce[0] = 0;
    this->netForce[1] = 0;
    this->state = Resting;
    this->t0 = getTime();

    this->propulsionDuration = propulsionDuration;
    this->propulsionForce = propulsionForce;
}

ObjectType Missile::what() const {
    return MissileType;
}

void Missile::update() {
    if (getAge() <= propulsionDuration) {
        netForce[0] += propulsionForce * cos(ang);
        netForce[1] += propulsionForce * sin(ang);
        ++propulsionDuration;
    }

    vel[0] += netForce[0] / mass;
    vel[1] += netForce[1] / mass;

    pos[0] += vel[0];
    pos[1] += vel[1];

    netForce[0] = 0;
    netForce[1] = 0;
}

void Missile::draw() const {
    drawRectangle(pos, radius/3.0, radius, ang, color);
}

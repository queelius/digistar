#include "flameParticle.h"

FlameParticle::FlameParticle(const GLfloat initialPos[2], GLfloat initialAngle, GLfloat initialMass, const GLfloat initialColor[3], const GLfloat initialVel[2], GLfloat radius) {
    memcpy(pos, initialPos, sizeof(pos));
    memcpy(color, initialColor, sizeof(color));    

    this->ang = initialAngle;
    this->mass = initialMass;
    this->radius = radius;

    memcpy(vel, initialVel, sizeof(vel));

    this->fixed = false;
    this->netForce[0] = 0;
    this->netForce[1] = 0;
    this->state = Resting;
    this->t0 = getTime();
}

ObjectType FlameParticle::what() const {
    return FlameParticleType;
}

void FlameParticle::update() {
    vel[0] += netForce[0] / mass;
    vel[1] += netForce[1] / mass;

    pos[0] += vel[0];
    pos[1] += vel[1];

    netForce[0] = 0;
    netForce[1] = 0;
}

void FlameParticle::draw() const {
    glEnable(GL_TEXTURE_2D);
    glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_MODULATE);
    glBindTexture(GL_TEXTURE_2D, SMOKE_PARTICLE);
    glDisable(GL_DEPTH_TEST);
    glEnable(GL_BLEND);
    glBlendFunc(GL_ONE_MINUS_SRC_COLOR, GL_ONE_MINUS_SRC_COLOR);
}

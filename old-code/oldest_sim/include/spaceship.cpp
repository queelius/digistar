#include "spaceship.h"
#include "drawPrimitives.h"
#include "projectile.h"
#include "utils.h"
#include <cmath>

Spaceship::Spaceship(const GLfloat initialPos[2], GLfloat initialAngle, GLfloat initialMass, glColor initialColor, const GLfloat initialVel[2], GLfloat radius) {
    this->pos[0] = initialPos[0];
    this->pos[1] = initialPos[1];
    this->ang = initialAngle;
    
    this->mass = initialMass;
    this->moment = initialMass * radius * radius * 0.3;

    this->radius = radius;

    this->angVel = 0;

    this->color = initialColor;

    if (initialVel) {
        this->vel[0] = initialVel[0];
        this->vel[1] = initialVel[1];
    }
    else {
        this->vel[0] = 0;
        this->vel[1] = 0;
    }

    this->fixed = false;
    this->target = NULL;

    this->netTorque = 0;
    this->netForce[0] = 0;
    this->netForce[1] = 0;

    state = Resting;
}

// implement couple diff versions
// 1) logical graph search (discrete movements)
// 2) this one
// 3) maybe others
void Spaceship::go() {
    return;

    if (target) {
        const GLfloat *p = target->getPosition();
        GLfloat dx = p[0] - pos[0];
        GLfloat dy = p[1] - pos[1];
        GLfloat r = sqrt(dx * dx + dy * dy);
        GLfloat theta = atan2(dy, dx);

        if (r < 1.1 * target->getRadius() + radius) {
            const GLfloat *targVel = target->getVelocity();
            GLfloat velocity = sqrt(GRAVITY_CONSTANT * target->getMass() / r);
            vel[0] = targVel[0] + velocity * cos(theta - PI/2);
            vel[1] = targVel[1] + velocity * sin(theta - PI/2);
            target = NULL;
            netForce[0] = 0;
            netForce[1] = 0;
        }
        else {
            GLfloat F = (1/log(r) > 0.1 ? 1/log(r) : 0.1);
            GLfloat Fx = F*cos(theta);
            GLfloat Fy = F*sin(theta);

            netForce[0] += Fx;
            netForce[1] += Fy;

            ang = atan2(Fy, Fx);
        }
    }
}

void Spaceship::setTarget(Object *target) {
    this->target = target;
}

ObjectType Spaceship::what() const {
    return SpaceshipType;
}

void Spaceship::update() {
    go();

    vel[0] += netForce[0] / mass;
    vel[1] += netForce[1] / mass;
    angVel += netTorque / moment;

    //if (target) {
    //    GLfloat v = sqrt(vel[0] * vel[0] + vel[1] * vel[1]);
    //    if (v > 233) {
    //        vel[0] = 233 * vel[0]/v;
    //        vel[1] = 233 * vel[1]/v;
    //    }
    //}
    
    /*
    static GLfloat totalWork = 0;
    GLfloat dist_x = (pos[0] + vel[0]) - pos[0];
    GLfloat dist_y = (pos[1] + vel[1]) - pos[1];
    GLfloat dist = sqrt(dist_x * dist_x + dist_y * dist_y);
    totalWork += sqrt(netForce[0] * netForce[0] + netForce[1] * netForce[1]) * dist;
    */

    pos[0] += vel[0];
    pos[1] += vel[1];

    if (target) {
        // placeholder
    }
    else {
        ang += angVel;
        if (ang > 2*PI)  ang -= 2*PI;
        if (ang < -2*PI) ang += 2*PI;
    }

    netForce[0] = 0;
    netForce[1] = 0;
    netTorque = 0;
}

void Spaceship::draw() const {
    drawTriangle(pos, std::sqrt(3.0)*radius, 2*radius, ang, color);
}

void Spaceship::shoot(GLfloat initialRelativeVelocity, GLfloat initialMass, GLfloat initialRadius) {
    if (initialRadius == NULL)
        initialRadius = initialMass / 5.0;

    const GLfloat cannonPos[] = {pos[0] + (radius + initialRadius) * cos(ang), pos[1] + (radius + initialRadius) * sin(ang)};
    const GLfloat v[] = {vel[0] + initialRelativeVelocity * cos(ang), vel[1] + initialRelativeVelocity * sin(ang) };
    o.push_back(new Projectile(cannonPos, mass, Green, v, initialRadius));
}

void Spaceship::shootMissile(GLfloat initialRelativeVelocity) {
    GLfloat v[] = {vel[0] + initialRelativeVelocity * cos(ang), vel[1] + initialRelativeVelocity * sin(ang) };
    o.push_back(new Missile(pos, ang, 0.5, Green, 0.875, 120, v, 4.0));
}

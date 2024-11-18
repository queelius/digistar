#include "PointParticle.h"

PointParticle::PointParticle()
{
}

PointParticle::PointParticle(const Vector2D &position, double mass, double charge, bool fixed)
{
}

PointParticle::PointParticle(const Vector2D &position, const Vector2D &velocity, double mass, double charge, bool fixed)
{
}

void PointParticle::addImpulse(const Vector2D &force, double duration)
{
}

void PointParticle::tick()
{
    velocity = velocity + impulse / mass;
    position = position + velocity; // incorrect
}

double PointParticle::getMass() const
{
    return mass;
}

double PointParticle::getCharge() const
{
    return charge;
}

double PointParticle::getKineticEnergy() const
{
    return 0.5 * mass * velocity * velocity; 
}

const Vector2D  &PointParticle::getCenterOfMassVelocity() const
{
    return velocity;
}

const Vector2D  &PointParticle::getCenterOfMass() const
{
    return position;
}

const Vector2D PointParticle::getMomentum() const
{
    return mass * velocity;
}

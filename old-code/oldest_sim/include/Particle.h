#ifndef PARTICLE_H
#define PARTICLE_H

#include "Object.h"
#include "vec2.h"
#include "Types.h"

class Particle: public Object
{
public:
    Particle(double time): Object(time)
    {
        addTypes(Types::PARTICLE);
    };

    Particle(double time, unsigned id): Object(time, id)
    {
        addTypes(Types::PARTICLE);
    };

    virtual double getMass() const = 0;
    virtual double getCharge() const = 0;
    virtual double getSpeed() const = 0;
    virtual double getTotalDistance() const = 0;
    virtual double getKineticEnergy() const = 0;
    virtual double getTotalKineticEnergy() const = 0;
    virtual double getInternalKineticEnergy() const = 0;
    virtual double getExternalKineticEnergy() const = 0;
    virtual bool getFixed() const = 0;

    virtual vec2 getAcceleration() const = 0;
    virtual vec2 getMomentum() const = 0;
    virtual vec2 getPosition() const = 0;
    virtual vec2 getVelocity() const = 0;
    virtual vec2 getAngularMomentum() const = 0;
    virtual vec2 getAngularVelocity() const = 0;
    virtual vec2 getAngularAcceleration() const = 0;
    virtual vec2 getChargeDipoleMoment() const = 0;

    virtual void setFixed(bool value = true) = 0;
    virtual void setPosition(const vec2& position) = 0;

    // for composite body, normalization + weighting stuff to set constituent velocity
    virtual void setVelocity(const vec2& velocity) = 0;
    virtual void setMass(double mass) = 0;
    virtual void setCharge(double charge) = 0;

    virtual void addImpulse(const vec2& force, double startTime, double endTime) = 0;
    virtual void addVelocity(const vec2& velocity) = 0;
    //virtual void addAngularVelocity() = 0;
    //virtual void addAngularImpulse() = 0;
    virtual void addDisplacement(const vec2& displacement) = 0;
    virtual void addThermalEnergy(double amount) = 0;
    virtual void addKineticEnergy(double amount, const vec2& direction = vec2::zero);

    // W = integrate F dl from 0 to distance
    virtual void addWork(const vec2& force, double distance) = 0;
    // add addWork that uses variable force function (wrt time or coordinates)


    //virtual void addAngularImpulse(...) = 0;
}

#endif
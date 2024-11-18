#ifndef COMPOSITE_PARTICLE_H
#define COMPOSITE_PARTICLE_H

#include "vec2.h"
#include "Object.h"
#include "Particle.h"
#include "Random.h"
#include "Constants.h"
#include <set>

class CompositeParticle: public Particle
{
public:
    void pin(const vec2& position)
    {

    };

    // heat capacity? how much of the thermal energy remains as kinetic energy when "heat"
    // (thermal energy) is transferred from another body to this one? the rest of the thermal
    // energy can be converted to other forms of internal energy.

    double getTemperature() const
    {
        vec2 v_cm = getVelocity();
        double num = 0;

        for (auto o = _components.begin(); o != _components.end(); o++)
        {
            vec2 v_rel = (*o)->getVelocity() - v_cm;
            num += (*o)->getMass() * (v_rel * v_rel);
        }

        return num / (double)(3 * getNumParticles() * BOLTZMANN_CONSTANT);
    };

    // note: as temperature increases, increase probability of body emitting particles;
    // for instance, for one, the spring bonds may be destroyed. however, other things
    // that could happen: a certain amount of energy is converted into a particle and
    // emitted, reducing thermal energy of the system.

    void addThermalEnergy(double amount)
    {
        using namespace queelius;

        // give kinetic energy to constituent particles randomly
    };

    // make each object have a spring force connection to 
    void setAngularVelocity(double angularVelocity, const vec2& axisPosition)
    {
        vec2 velocity = getVelocity();
        for (auto o = _components.begin(); o != _components.end(); o++)
            (*o)->setVelocity(velocity + angularVelocity * ((*o)->getPosition() - axisPosition).perp().normal());
    };

    void convertInternalEnergy()
    {
    };

    vec2 getPosition() const
    {
        vec2 pos(0, 0);
        for (auto o = _components.begin(); o != _components.end(); o++)
            pos += (*o)->getMass() * (*o)->getPosition();
        return (1 / getMass()) * pos;
    };

    vec2 getVelocity() const
    {
        return (1 / getMass()) * getMomentum();
    };

    unsigned getNumParticles() const
    {
        return _components.size();
    };

    vec2 getAngularMomentum() const
    {
        vec2 cm = getPosition();
        vec2 momentum(0, 0);

        for (auto o = _components.begin(); o != _components.end(); o++)
            momentum += crossProduct(((*o)->getPosition() - cm), (*o)->getMomentum());
        return momentum;
    };

    vec2 getChargeDipoleMoment() const
    {
        vec2 dipole(0, 0);
        for (auto o = _components.begin(); o != _components.end(); o++)
            dipole += (*o)->getCharge() * (*o)->getPosition();
        return dipole;
    };

    vec2 getMomentum() const
    {
        vec2 p(0, 0);
        for (auto o = _components.begin(); o != _components.end(); o++)
            p += (*o)->getMass() * (*o)->getVelocity();
        return p;
    };

    double getMass() const
    {
        double mass = 0;
        for (auto o = _components.begin(); o != _components.end(); o++)
            mass += (*o)->getMass();
        return mass;
    };

    double getCharge() const
    {
        double charge = 0;
        for (auto o = _components.begin(); o != _components.end(); o++)
            charge += (*o)->getCharge();
        return charge;
    };

    double getTotalKineticEnergy() const
    {
        double ke = 0;
        for (auto o = _components.begin(); o != _components.end(); o++)
            ke += (*o)->getMass() * (*o)->getVelocity().mag2();
        return 0.5 * ke;
    };

    double getInternalKineticEnergy() const
    {
        double ke = 0;
        for (auto o = _components.begin(); o != _components.end(); o++)
            ke += (*o)->getMass() * ((*o)->getVelocity() - getVelocity()).mag2();
        return 0.5 * ke;
    };

    double getExternalKineticEnergy() const
    {
        return 0.5 * getMass() * getVelocity().mag2();
    };

protected:
    std::set<Particle*> _components;

};

#endif
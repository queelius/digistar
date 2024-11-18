#ifndef PARTICLE_CREATOR_H
#define PARTICLE_CREATOR_H

#include <cmath>
#include <ctime>

#include "GlobalConstants.h"
#include "PointParticle.h"
#include "Timer.h"

// particles in a particle simulation
//
// notes: for interesting effects, consider
// 1) making each particle affect, gravitationally,
// elastically, or whatever, every other particle
// -- or perhaps only certain kinds of particles,
// -- or perhaps only randomly selected particles,.
// -- or perhaps make it more likely that particles
// in a column range will attract other particles
// in the same column range (or more strongly)
// 2) normal distribution: radius of emitter
// will relate to variance and cutoff point, center
// will generally be the mean. every other aspect
// of the particle will also, generally, be
// governed by a normal distribution, e.g.,
// lifetime, mass, particle radius, particle
// transparency, etc.
// -- also consider poisson distribution
// -- and binomial distribution (for quick
// generation of normal using discrete math)

class EntityCreator
{
public:
    virtual void        tick()          = 0;
    virtual Entity      *getEntity()    = 0;
};

class CombustionCreator: public EntityCreator
{
public:
    // initialProbability: initial probability of creating a flame particle (1 - initialProbability == initial probability of creating a smoke particle)
    // time, probability: at specified time, have specified probability 
    CombustionCreator(double initialProbability, double time, double probability)
    {
        Pi = initialProbability;
        k = -1 / time * log(initialProbability / probability);

        t.start();
    };

    void tick()
    {
        // start off with probability P for fire and probability (1 - P) for smoke,
        // and slowly decrease P to 0.

        P = Pi * std::pow(E, k * t.getElapsed());
    };

    Entity *getEntity(const Vector2D &position, const Vector2D &velocity)
    {
        // if random number < P then make fire particle
        // else make smoke particle
    };

protected:
    double k;
    double Pi;
    double P;

    Timer t;
};


#endif
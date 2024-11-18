#ifndef PARTICLE_H
#define PARTICLE_H

#include "GlobalConstants.h"
#include "PointMass.h"

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

class Particle: public PointMass {
public:
    EntityType getType() { return ParticleEntity; };
    virtual void draw()          =0; // draws the particle
    virtual void tick()          =0; // updates the state of the particle
};

#endif
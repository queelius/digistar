#ifndef GRAVITY_FORCE_H
#define GRAVITY_FORCE_H

#include "vec2.h"
#include "Force.h"
#include "Particle.h"
#include "Types.h"
//#include <set>


// derive from PairForce?
class GravityForce: public Force
{
public:
    GravityForce(double time): Force(time)
    {
        _types |= Types::GRAVITY_FORCE;
    };

    void updateImp(double time)
    {
    };

protected:
    //std::set<Particle*> _particles;
    Particle *end1;
    Particle *end2;
};

#endif

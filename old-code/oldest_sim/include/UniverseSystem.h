#ifndef UNIVERSE_SYSTEM
#define UNIVERSE_SYSTEM

#include "Timer.h"

// for relativistic effects, should we slow down time? relavistic mass different than rest mass in simulation?

class UniverseSystem
{
public:
    double  getMaxSpeed()                const;
    double  getGravitationalConstant()   const;

private:
    Timer age;                      // age of universe
    double gravitationalConstant;   // default gravitational constant; overridable
    double maxSpeed;                // like the speed of light; used for relativistic calculations
};


#endif
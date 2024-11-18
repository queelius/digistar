#ifndef SPRING_FORCE_H
#define SPRING_FORCE_H

#include <list>
#include "Object.h"
#include "Types.h"
#include "Particle.h"
#include "vec2.h"
#include "Events.h"
#include "Force.h"

class SpringForce: public Force
{
public:
    SpringForce(double time, Particle* end1, Particle* end2, double equilibriumDistance, double springConstant, double maxTension = 0)
        : Force(time, Types::IDEAL_SPRING_FORCE)
    {
        _lastUpdateTime = time;
        _end1 = end1;
        _end2 = end2;
        _springConstant = springConstant;
        _equilibriumDistance = equilibriumDistance;
        _maxTensionSquared = maxTension * maxTension;
    };

    void update(double time)
    {        
        vec2 diff = _end2->getPosition() - _end1->getPosition();
        vec2 force = _springConstant * (diff - _equilibriumDistance * diff.normal());

        if (_maxTensionSquared != 0 && force.mag2() >= _maxTensionSquared)
        {
            force = sqrt(_maxTensionSquared) * force.normal();
            _expire = true;
            alert(Events::EXPIRED);
        };

        _end1->addImpulse(-1 * force, 0, time - _lastUpdateTime);
        _end2->addImpulse(force, 0, time - _lastUpdateTime);
        _lastUpdateTime = time;
    };

private:
    double _equilibriumDistance;
    double _springConstant;
    double _maxTensionSquared;

    Particle* _end1;
    Particle* _end2;
};

#endif
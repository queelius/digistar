#ifndef PHYSICAL_ATOMIC_ENTITY_H
#define PHYSICAL_ATOMIC_ENTITY_H

#include "PhysicalEntity.h"

class PhysicalAtomicEntity: public PhysicalEntity
{
    void            addImpulse(
        const double force[2],
        double duration,
        const double point[2]);

    void            tick();

    double          getMass()                               const;
    double          getCharge()                             const;

    const double    *getPositionOfCenterOfMass()            const;
    const double    *getVelocityOfCenterOfMass()            const;
    double          getVelocityOfCenterOfMassMagnitude()    const;
    bool            isFixed()                               const;
};

#endif
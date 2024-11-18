#include "Entity.h"

class PhysicalEntity: public Entity
{
public:
    virtual void            addImpulse(
        const double force[2],
        double duration,
        const double point[2]);

    virtual void            tick();

    virtual double          getMass()                               const;
    virtual double          getCharge()                             const;
    virtual const double    *getPositionOfCenterOfMass()            const;
    virtual const double    *getVelocityOfCenterOfMass()            const;
    virtual double          getVelocityOfCenterOfMassMagnitude()    const;
    virtual bool            isFixed()                               const;
};

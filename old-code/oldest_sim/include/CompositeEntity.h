#ifndef COMPOSITE_ENTITY_H
#define COMPOSITE_ENTITY_H

#include <iostream>
#include <cmath>
#include <cstdlib>
#include "glut.h"
#include "Entity.h"
#include "Vector2D.h"
#include "CompositeEntityList.h"

class CompositeEntity: public Entity
{
public:
                    CompositeEntity();
    void            tick();

    // equilibrium for two nodes defaults to their distance apart that existed
    // during their initial connection
    void            makeSpringLink(double springConstant, unsigned int entityOneId, unsigned int entityTwoId);
    void            makeSpringLink(double springConstant, double equilibrium, unsigned int entityOneId, unsigned int entityTwoId);

    unsigned int    getComponentCount();
    double          getMass();
    double          getCharge();

    // provide it with linear and angular velocity equal to "center" values
    void            addComponentEntity(const Vector2D &relativePosition, double mass, double charge, bool fixed = false);
    void            addComponentEntity(const Vector2D &relativePosition, Entity *ent);
   
    const Vector2D  &getCenterOfMassVelocity();
    const Vector2D  &getCenterOfMass();
    const Vector2D  &getDipoleMoment(); // "center" of charge

    double          getKineticEnergy();
    double          getAngularKineticEnergy();
    double          getLinearKineticEnergy();

    const Vector2D  &getMomentum();
    const Vector2D  &getLinearMomentum();
    double          getAngularMomentum();

    // a = v_i^2 / r_i
    // use this to force rotation about a fixed axis;
    // create a radius force (pointing towards the center of mass) that applies this to every particle in the composite body
    void            setAngularVelocity(double w, const Vector2D &axisOfRotation);
    void            setCenterOfMassVelocity(const Vector2D &velocity);
    void            setCenterOfMass(const Vector2D &position);

    // give impulse to bodies (that make up this object) in a direction
    // perpendicular to axis. add impulse to each body such that
    // change in w, dw, is the same for every body with respect to
    // specified axis
    void            addAngularImpulse(const Vector2D &force, double duration, const Vector2D &axisOfRotation);

    // give impulse to bodies (that make up this object) in a direction
    // perpendicular to center of mass (as an axis of rotation)
    void            addAngularImpulse(const Vector2D &force, double duration);

    // add impulse to bodies (that make up this object) such that
    // every body gets same amount of change in momentum
    void            addImpulse(const Vector2D &force, double duration);

protected:
    CompositeEntityList entityList;

    Vector2D            centerOfMass;
    Vector2D            centerOfMassVelocity;
    double              totalMass;
    double              totalCharge;
};

#endif
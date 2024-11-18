#ifndef IDEAL_SPRING_H
#define IDEAL_SPRING_H

#include "Link.h"

class IdealSpring: public Link
{
public:
            IdealSpring(
                double stiffness,
                double equilibrium,
                double elasticLimit,
                Entity *ent1,
                Entity *ent2
            );

    void    tick();

protected:
    double  stiffness;
    double  equilibrium;
    double  elasticLimit;
    Entity  *ent1;
    Entity  *ent2;
};

class IdealSpringWithDampening: public IdealSpring
{
public:
            IdealSpringWithDampening(
                double stiffness,
                double equilibrium,
                double elasticLimit,
                double dampening,
                Entity *ent1,
                Entity *ent2
            );

    void    tick();

protected:
    double dampening;
};

#endif

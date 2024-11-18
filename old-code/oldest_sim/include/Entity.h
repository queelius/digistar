#ifndef ENTITY_H
#define ENTITY_H

#include "GlobalConstants.h"
#include "Timer.h"

enum EntityType
{
    EXTENDED_MASS,
    POINT_MASS,
    PARTICLE,
    EMITTER,
    LINK_FORCE,
    LIFE_FORM,
    PHYSICAL_ENTITY,
    IDEAL_SPRING,
    EMITTER,
    CONCEPT_ENTITY,
    ENTITY
};

class Entity {
public:
    virtual EntityType getEntityType()  = 0; // returns the type of the entity
    virtual void tick()                 = 0; // updates the state of the particle

    Entity()
    {
        _expire = false;
        _id = _count++;
        _age.start();
    };

    bool isExpired() const
    {
        return _expire;
    };

    void expire()
    {
        _expire = true;
    };

    unsigned int getId() const
    {
        return _id;
    };

    double getAge() const
    {
        return _age.getElapsed();
    };

private:
    static unsigned int _count;

    Timer _age;
    unsigned int _id;
    bool _expire;
};

#endif
#ifndef GAS_H
#define GAS_H

#include "Object.h"
#include "Types.h"
#include "Constants.h"
#include <list>

class Gas: public Object
{
public:
    Gas(double time, double id, double particleNum, double temperature): Object(time, id)
    {
        addTypes(Types::GAS);
    };

    Gas(double time, double particleNum, double temperature): Object(time)
    {
        addTypes(Types::GAS);
    };

    Gas(double time, double id, double pressure, double volume): Object(time, id)
    {
        addTypes(Types::GAS);
    };

    Gas(double time, double pressure, double volume): Object(time)
    {
        addTypes(Types::GAS);
    };

    void addTemperature(double amount)
    {
    };

protected:
    list<Particle*> _particles;
};

#endif
#ifndef POTENTIAL_ENERGY_FIELD_H
#define POTENTIAL_ENERGY_FIELD_H

#include "vec2.h"
#include <limits>
#include <list>
#include "Particle.h"

class PotentialEnergyField
{
public:
    virtual void apply(std::list<Particle*> particles)
    {
        for (auto p = particles.begin(); p != particles.end(); p++)
        {
            vec2 pos = (*p)->getPosition();
            gradient(pos.getX(), pos.getY());
        }
    };

protected:
};

class UniversalGravityField
{

};

class FuncObject
{
public:
    virtual double operator()(double x, double y) = 0;
};

class GravityFunc
{
public:
    GravityFunc(const vec2& center, double mass, double G)
        : _center(center), _mass(mass), _G(G) {};

    double operator()(double x, double y)
    {
        
        _center.getX() - x;
        _center.getY() - y;
    };

protected:
    double _G;
    double _mass;
    vec2 _center;
};

vec2 gradient(const FuncObject& f, double x, double y,
    double dx = std::numeric_limits<double>::epsilon(),
    double dy = std::numeric_limits<double>::epsilon())
{
    return vec2(
        (f(x + dx, y) - f(x, y)) / dx,
        (f(x, y + dy) - f(x, y)) / dy);
};

double directionalDerivative(const FuncObject& f, double x, double y, const vec2& direction)
{
    return gradient(f, x, y) * direction.normal();
};

#endif
#ifndef VEC3_H
#define VEC3_H

#include <cmath>
#include "Exceptions.h"
using namespace std;

class vec3
{
    friend bool operator==(const vec3& v, const vec3& u);
    friend bool operator!=(const vec3& v, const vec3& u);
    friend bool operator<(const vec3& v, const vec3& u);
    friend bool operator>(const vec3& v, const vec3& u);
    friend bool operator>=(const vec3& v, const vec3& u);
    friend bool operator<=(const vec3& v, const vec3& u);
    friend void operator*=(float scalar, vec3& v);
    friend void operator+=(vec3& v, vec3& u);
    friend void operator-=(vec3& v, vec3& u);
    friend float operator*(const vec3& v, const vec3& u);
    friend float operator*(const vec3& v, const vec3& u);
    friend vec3 crossProduct(const vec3& v, const vec3& u);
    friend float scalarProj(const vec3& v, const vec3& u);
    friend float angleBetween(const vec3& v, const vec3& u);
    friend bool isParallel(const vec3& v, const vec3& u);
    friend bool isPerpindicular(const vec3& v, const vec3& u);
    friend vec3 vecProj(const vec3& v, const vec3& u);
    friend vec3 operator*(float scalar, const vec3& v);
    friend vec3 operator+(const vec3& v, const vec3& u);
    friend vec3 operator-(const vec3& v, const vec3& u);

public:
    static const vec3 ones;
    static const vec3 zero;
    static const vec3 i;
    static const vec3 j;
    static const vec3 k;

    static vec3 makeFromCyclindricalCoordinates(float theta, float radius, float z)
    {
        return vec3(radius * cos(theta), radius * sin(theta), z);
    };

    static vec3 makeFromCartesianCoordinates(float x, float y, float z)
    {
        return vec3(x, y, z);
    };

    static vec3 makeFromSphericalCoordinates(float theta, float phi, float radius)
    {
        return vec3(radius * cos(theta) * cos(phi),
            radius * sin(theta) * sin(phi),
            radius * cos(theta));
    };

    vec3() {};

    vec3(float x, float y, float z)
    {
        v[0] = x;
        v[1] = y;
        v[2] = z;
    };

    vec3(const float v[3])
    {
        vec3(v[0], v[1], v[2]);
    };

    vec3 operator-() const
    {
        return -1.0f * (*this);
    };

    vec3 normal() const
    {
        return (1.0f / mag()) * (*this);
    };

    float getAngleFromXAxis() const
    {
        return acos(normal() * i);
    };

    float getAngleFromYAxis() const
    {
        return acos(normal() * j);
    };

    float getAngleFrom(const vec3& v) const
    {
        return angleBetween(v, *this);
    };

    vec3 orthogonal() const
    {
        if (v[0] != 0.0f)
            return vec3(-(v[1] + v[2]) / v[0], 1.0f, 1.0f).normal();
        else if (v[1] != 0.0f)
            return vec3(1, -(v[0] + v[2]) / v[1], 1.0f).normal();
        else if (v[2] != 0)
            return vec3(1.0f, 1.0f, -(v[0] + v[1]) / v[2]).normal();
        else
            return vec3(0.0f, 0.0f, 0.0f);
    };

    void normalize()
    {
        *this = normal();
    };

    float mag2() const
    {
        return v[0] * v[0] + v[1] * v[1] + v[2] * v[2];
    };

    float mag() const
    {
        return sqrt(mag2());
    };

    const float &operator[](unsigned int idx) const
    {
        if (idx > 2)
            throw IndexNotValid();

        return v[idx];          
    };

    float &operator[](unsigned int idx)
    {
        if (idx > 2)
            throw IndexNotValid();

        return v[idx];          
    };

    const float* toArray() const
    {
        return v;
    };

protected:
    float v[3];
};

const vec3 vec3::ones = vec3(1.0f, 1.0f, 1.0f);
const vec3 vec3::zero = vec3(0.0f, 0.0f, 0.0f);
const vec3 vec3::i = vec3(1.0f, 0.0f, 0.0f);
const vec3 vec3::j = vec3(0.0f, 1.0f, 0.0f);
const vec3 vec3::k = vec3(0.0f, 0.0f, 1.0f);

vec3 operator+(const vec3& v, const vec3& u)
{
    return vec3(v[0] + u[0], v[1] + u[1], v[2] + u[2]);
}

vec3 operator-(const vec3& v, const vec3& u)
{
    return v + (-1.0f * u);
}

float operator*(const vec3& v, const vec3& u)
{
    return v[0] * u[0] + v[1] * u[1] + v[2] * u[2];
}

float scalarProj(const vec3& v, const vec3& u)
{
    return v * u / u.mag();
}

vec3 vecProj(const vec3& v, const vec3& u)
{
    return (v * u / u.mag2()) * u;
}

vec3 operator*(float scalar, const vec3& v)
{
    return vec3(scalar * v[0], scalar * v[1], scalar * v[2]);
}

float angleBetween(const vec3& v, const vec3& u)
{
    return acos((v * u) / (v.mag() * u.mag()));
}

void operator*=(vec3& v, float scalar)
{
    v = scalar * v;
}

void operator+=(vec3& v, vec3& u)
{
    v = v + u;
}

void operator-=(vec3& v, vec3& u)
{
    v = v - u;
}

vec3 crossProduct(const vec3& v, const vec3& u)
{
    return vec3(v[1] * u[2] - v[2] * u[1],
        v[2] * u[0] - v[0] * u[3],
        v[0] * u[1] - v[1] * u[0]);
}

bool operator==(const vec3& v, const vec3& u)
{
    return v[0] == u[0] && v[1] == u[1] && v[2] == u[2];
}

bool operator!=(const vec3& v, const vec3& u)
{
    return !(v == u);
}

bool operator<(const vec3& v, const vec3& u)
{
    return v.mag2() < u.mag2();
}

bool operator>(const vec3& v, const vec3& u)
{
    return u < v;
}

bool operator>=(const vec3& v, const vec3& u)
{
    return v == u || v > u;
}

bool operator<=(const vec3& v, const vec3& u)
{
    return v < u || v == u;
}

bool isParallel(const vec3& v, const vec3& u)
{
    return crossProduct(u, v).mag() == 0.0f;
}

bool isPerpindicular(const vec3& v, const vec3& u)
{
    return u * v == 0.0f;
}

#endif
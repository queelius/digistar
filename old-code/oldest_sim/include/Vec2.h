#ifndef VEC2_H
#define VEC2_H

#include <cmath>
using namespace std;

// 2D vector
class vec2
{
    friend bool operator==(const vec2& v, const vec2& u);
    friend bool operator!=(const vec2& v, const vec2& u);
    friend bool operator<(const vec2& v, const vec2& u);
    friend bool operator>(const vec2& v, const vec2& u);
    friend bool operator>=(const vec2& v, const vec2& u);
    friend bool operator<=(const vec2& v, const vec2& u);
    friend void operator*=(double scalar, vec2& v);
    friend void operator+=(vec2& v, vec2& u);
    friend void operator-=(vec2& v, vec2& u);
    friend double operator*(const vec2& v, const vec2& u);
    friend double operator*(const vec2& v, const vec2& u);
    friend vec2 crossProduct(const vec2& v, const vec2& u);
    friend double scalarProj(const vec2& v, const vec2& u);
    friend double angleBetween(const vec2& v, const vec2& u);
    friend bool isParallel(const vec2& v, const vec2& u);
    friend bool isPerpindicular(const vec2& v, const vec2& u);
    friend vec2 vecProj(const vec2& v, const vec2& u);
    friend vec2 operator*(double scalar, const vec2& v);
    friend vec2 operator+(const vec2& v, const vec2& u);
    friend vec2 operator-(const vec2& v, const vec2& u);

public:
    static const vec2 zero;
    static const vec2 i;
    static const vec2 j;

    vec2(): _x(0), _y(0) {};
    vec2(double x, double y): _x(x), _y(y) {};
    vec2(const double v[2]): _x(v[0]), _y(v[1]) {};

    vec2 operator-() const
    {
        return -1 * *this;
    };

    vec2 normal() const
    {
        double len = sqrt(_x*_x + _y*_y);
        return vec2(_x / len, _y / len);
    };

    double getAngleFromXAxis() const
    {
        return acos(normal() * i);
    };

    double getAngleFromYAxis() const
    {
        return acos(normal() * j);
    };

    void setX(double x)
    {
        _x = x;
    };

    void setY(double y)
    {
        _y = y;
    };

    double getX() const
    {
        return _x;
    };

    double getY() const
    {
        return _y;
    };

    vec2 perp()
    {
        return vec2(-_y, _x);
    };

    void normalize()
    {
        double len = sqrt(_x*_x + _y*_y);
        _x /= len;
        _y /= len;
    };

    double mag2() const
    {
        return _x*_x + _y*_y;
    };

    double mag() const
    {
        return sqrt(_x*_x + _y*_y);
    };

protected:
    double _x;
    double _y;
};

const vec2 vec2::zero = vec2(0, 0);
const vec2 vec2::i = vec2(1, 0);
const vec2 vec2::j = vec2(0, 1);

// vector addition
vec2 operator+(const vec2& v, const vec2& u)
{
    return vec2(v._x + u._x, v._y + u._y);
}

// vector subtraction
vec2 operator-(const vec2& v, const vec2& u)
{
    return vec2(v._x - u._x, v._y - u._y);
}

// vector dot product
double operator*(const vec2& v, const vec2& u)
{
    return v._x * u._x + v._y * u._y;
}

// scalar projection of v1 onto v2
double scalarProj(const vec2& v, const vec2& u)
{
    return v * u / u.mag();
}

// vector projection of v1 onto v2
vec2 vecProj(const vec2& v, const vec2& u)
{
    return ((v * u) / u.mag2()) * u;
}

// scalar multiplication
vec2 operator*(double scalar, const vec2& v)
{
    return vec2(scalar*v._x, scalar*v._y);
}

double angleBetween(const vec2& v, const vec2& u)
{
    return acos((v * u) / (v.mag() * u.mag()));
}

void operator*=(double scalar, vec2& v)
{
    v._x *= scalar;
    v._y *= scalar;
}

void operator+=(vec2& v, vec2& u)
{
    v._x += u._x;
    v._y += u._y;
}

void operator-=(vec2& v, vec2& u)
{
    v._x -= u._x;
    v._y -= u._y;
}

vec2 crossProduct(const vec2& v, const vec2& u)
{
    return vec2(v._x * u._y, -u._x * v._y);
}

bool operator==(const vec2& v, const vec2& u)
{
    return v._x == u._x && v._y == u._y;
}

bool operator!=(const vec2& v, const vec2& u)
{
    return !(v == u);
}

bool operator<(const vec2& v, const vec2& u)
{
    return v.mag2() < u.mag2();
}

bool operator>(const vec2& v, const vec2& u)
{
    return v.mag2() > u.mag2();
}

bool operator>=(const vec2& v, const vec2& u)
{
    return v.mag2() > u.mag2() || v == u;
}

bool operator<=(const vec2& v, const vec2& u)
{
    return v.mag2() < u.mag2() || v == u;
}

bool isParallel(const vec2& v, const vec2& u)
{
    return crossProduct(u, v).mag() == 0;
}

bool isPerpindicular(const vec2& v, const vec2& u)
{
    return u * v == 0;
}

#endif
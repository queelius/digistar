#ifndef VECTOR_2D_H
#define VECTOR_2D_H

#include <cmath>
#include "Exception.h"

class Vector2D
{
public:
    static const unsigned int X_COMPONENT       = 0;
    static const unsigned int Y_COMPONENT       = 1;
    static const unsigned int DIMENSION_SIZE    = 2;

    /***************
     * constuctors *
     ***************/

                    Vector2D();
                    Vector2D(const double v[DIMENSION_SIZE]);
                    Vector2D(double x, double y);
                    Vector2D(const Vector2D &v);

    /***********
     * methods *
     ***********/

    unsigned int    getDimension()                      const;
    double          getAngleFromXAxis()                 const;
    double          getAngleFromYAxis()                 const;
    double          getLength()                         const;
    Vector2D        getNormalized()                     const;
    double          getMagnitude()                      const;
    const double    *toArray()                          const;
    double          getX()                              const;
    double          getY()                              const;
    double          getMagnitudeSquared()               const;
    Vector2D        getNormal()                         const;

    double          &getX();
    double          &getY();

    void            normalize();
    void            invert();

    double          &operator[](unsigned int index);
    double          operator[](unsigned int index)      const;
    const Vector2D  &operator=(const Vector2D &v);
    Vector2D        operator-()                         const;
    void            operator()(const double v[2]);
    void            operator()(double x, double y);
    const double    operator*()                         const;

    Vector2D        operator*=(double rhs);
    Vector2D        operator+=(const Vector2D &rhs);
    Vector2D        operator-=(const Vector2D &rhs);
    Vector2D        operator/=(double rhs);

protected:
    double          _components[DIMENSION_SIZE];

    void            copy(double x, double y);
    void            copy(const double v[]);
    void            copy(const Vector2D &v);
};

/***************************
 * operators and functions *
 ***************************/

bool        isParallel(const Vector2D &v1, const Vector2D &v2);
bool        isPerpindicular(const Vector2D &v1, const Vector2D &v2);
double      getProjection(const Vector2D &v1, const Vector2D &v2);
Vector2D    getFromPolarCoordinates(double magnitude, double angle);
double      getAngleDifference(const Vector2D &v1, const Vector2D &v2);
double      getCrossProduct(const Vector2D &v1, const Vector2D &v2);
double      getDotProduct(const Vector2D &v1, const Vector2D &v2);
double      getCosineOfAngleDifference(const Vector2D &v1, const Vector2D &v2);

Vector2D    operator-(const Vector2D &lhs, const Vector2D &rhs);
Vector2D    operator*(double lhs, const Vector2D &rhs);
Vector2D    operator/(double lhs, const Vector2D &rhs);
Vector2D    operator*(const Vector2D &lhs, double rhs);
Vector2D    operator/(const Vector2D &lhs, double rhs);
Vector2D    operator+(const Vector2D &lhs, const Vector2D &rhs);

double      operator*(const Vector2D &lhs, const Vector2D &rhs);
bool        operator==(const Vector2D &lhs, const Vector2D &rhs);
bool        operator!=(const Vector2D &lhs, const Vector2D &rhs);
bool        operator<=(const Vector2D &lhs, const Vector2D &rhs);
bool        operator>=(const Vector2D &lhs, const Vector2D &rhs);
bool        operator<(const Vector2D &lhs, const Vector2D &rhs);
bool        operator>(const Vector2D &lhs, const Vector2D &rhs);

#endif
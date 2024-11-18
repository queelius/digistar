#ifndef PRIMITIVES_H
#define PRIMITIVES_H

#include "glut.h"
#include <cmath>

enum PrimitiveType { PointType, RayType, LineType, VectorType };

class Primitive {
public:
    virtual PrimitiveType what() const =0;
};

class Vector: public Primitive {
    friend class Point;

    friend GLfloat getAngle(const Vector &v1, const Vector &v2);
    friend GLfloat dotProduct(const Vector &v1, const Vector &v2);

    friend bool operator==(const Vector &v1, const Vector &v2);
    friend bool operator!=(const Vector &v1, const Vector &v2);

    friend Vector operator-(const Point &minuend, const Point &subtrahend);
    friend Vector operator+(const Vector &v1, const Vector &v2);
    friend Vector operator-(const Vector &minuend, const Vector &subtrahend);
    friend Vector operator*(GLfloat scalar, const Vector &v);
    friend Vector operator*(const Vector &v, GLfloat scalar);
    friend Vector operator/(const Vector &v, GLfloat scalar);

    friend Point operator+(const Point &p, const Vector &v);
    friend Point operator+(const Vector &v, const Point &p);
    friend Point operator-(const Point &p, const Vector &v);

public:
    Vector(const Vector &v);
    Vector(GLfloat mag, GLfloat angle);
    Vector(const GLfloat pt[2]);
    
    PrimitiveType what() const;
    Vector operator-();
    const GLfloat *toArray() const;
    void normalize();

    GLfloat getMagnitude() const;
    GLfloat getMagnitude2() const;
    GLfloat getAngle() const;

protected:
    // implicitly defined as starting at {0,0}
    // so pt = {0,2} is a vector with a magnitude of 2
    // and a direction of 0 (with respect to x-axis)
    GLfloat pt[2];
};

class Point: public Primitive {
    friend class Vector;

    friend bool operator==(const Point &p1, const Point &p2);
    friend bool operator!=(const Point &p1, const Point &p2);

    friend Point operator+(const Point &p, const Vector &v);
    friend Point operator+(const Vector &v, const Point &p);
    friend Point operator-(const Point &p, const Vector &v);

    friend Vector operator-(const Point &minuend, const Point &subtrahend);
    friend Vector operator+(const Vector &v1, const Vector &v2);
    friend Vector operator-(const Vector &minuend, const Vector &subtrahend);

public:
    Point(GLfloat x, GLfloat y);
    Point(const GLfloat pt[2]);
    Point(const Point &p);

    GLfloat getX() const;
    GLfloat getY() const;

    PrimitiveType what() const;
    const GLfloat *toArray() const;

protected:
    GLfloat pt[2];
};

bool isParallel(const Vector &v1, const Vector &v2);
bool isPerpindicular(const Vector &v1, const Vector &v2);
GLfloat dotProduct(const Vector &v1, const Vector &v2);
GLfloat getAngle(const Vector &v1, const Vector &v2);

bool operator==(const Vector &v1, const Vector &v2);
bool operator!=(const Vector &v1, const Vector &v2);

Vector operator-(const Point &minuend, const Point &subtrahend);
Vector operator+(const Vector &v1, const Vector &v2);
Vector operator-(const Vector &minuend, const Vector &subtrahend);

Vector operator*(const Vector &v, GLfloat scalar);
Vector operator*(GLfloat scalar, const Vector &v);
Vector operator/(const Vector &v, GLfloat scalar);

bool operator==(const Point &p1, const Point &p2);
bool operator!=(const Point &p1, const Point &p2);

Point operator+(const Point &p, const Vector &v);
Point operator+(const Vector &v, const Point &p);
Point operator-(const Point &p, const Vector &v);

#endif
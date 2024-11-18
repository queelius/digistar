#include "primitives.h"

// vector implementation
Vector::Vector(const GLfloat pt[2]) {
    this->pt[0] = pt[0];
    this->pt[1] = pt[1];
}

Vector::Vector(GLfloat mag, GLfloat angle) {
    this->pt[0] = mag * cos(angle);
    this->pt[1] = mag * sin(angle);
}

Vector::Vector(const Vector &v) {
    this->pt[0] = v.pt[0];
    this->pt[1] = v.pt[1];
}

PrimitiveType Vector::what() const {
    return VectorType;
}

Vector Vector::operator-() {
    return (*this) * -1;
}

void Vector::normalize() {
    GLfloat mag = getMagnitude();
    if (mag != 0) {
        pt[0] /= mag;
        pt[1] /= mag;
    }
}

const GLfloat *Vector::toArray() const {
    return pt;
}

GLfloat Vector::getMagnitude2() const {
    return (pt[0] * pt[0] + pt[1] * pt[1]);
}

GLfloat Vector::getMagnitude() const {
    return sqrt(pt[0] * pt[0] + pt[1] * pt[1]);
}

GLfloat Vector::getAngle() const {
    return atan2(pt[1], pt[0]);
}

// point implementation
Point::Point(const GLfloat pt[2]) {
    this->pt[0] = pt[0];
    this->pt[1] = pt[1];
}

Point::Point(const Point &p) {
    this->pt[0] = p.pt[0];
    this->pt[1] = p.pt[1];
}

Point::Point(GLfloat x, GLfloat y) {
    this->pt[0] = x;
    this->pt[1] = y;
}

PrimitiveType Point::what() const {
    return PointType;
}

GLfloat Point::getX() const {
    return pt[0];
}

GLfloat Point::getY() const {
    return pt[1];
}

const GLfloat *Point::toArray() const {
    return pt;
}

// operator functions for vectors and points
bool operator==(const Vector &v1, const Vector &v2) {
    return (v1.pt[0] == v2.pt[0]);
}

bool operator!=(const Vector &v1, const Vector &v2) {
    return !(v1 == v2);
}

Vector operator-(const Point &minuend, const Point &subtrahend) {
    GLfloat pt[] = {minuend.pt[0] - subtrahend.pt[0], minuend.pt[1] - subtrahend.pt[1]};
    return Vector(pt);
}

Vector operator+(const Vector &v1, const Vector &v2) {
    GLfloat pt[] = {v1.pt[0] + v2.pt[0], v1.pt[1] + v2.pt[1]};
    return Vector(pt);
}

Vector operator-(const Vector &minuend, const Vector &subtrahend) {
    return minuend + (subtrahend * -1);
}

Vector operator*(const Vector &v, GLfloat scalar) {
    GLfloat pt[] = {v.pt[0] * scalar, v.pt[1] * scalar};
    return Vector(pt);
}

Vector operator/(const Vector &v, GLfloat scalar) {
    return v * (1 / scalar);
}

Vector operator*(GLfloat scalar, const Vector &v) {
    return v * scalar;
}

Point operator+(const Point &p, const Vector &v) {
    GLfloat pt[] = {p.pt[0] + v.pt[0], p.pt[1] + v.pt[1]};
    return Point(pt);
}

Point operator+(const Vector &v, const Point &p) {
    return p + v;
}

Point operator-(const Point &p, const Vector &v) {
    return p + v*-1;
}

bool operator==(const Point &p1, const Point &p2) {
    return p1.pt[0] == p2.pt[0] && p1.pt[1] == p1.pt[1];
}

bool operator!=(const Point &p1, const Point &p2) {
    return !(p1 == p2);
}

bool isPerpindicular(const Vector &v1, const Vector &v2) {
    return dotProduct(v1, v2) == 0;
}

bool isParallel(const Vector &v1, const Vector &v2) {
    return dotProduct(v1, v2) == v1.getMagnitude() * v2.getMagnitude();
}

GLfloat dotProduct(const Vector &v1, const Vector &v2) {
    return v1.pt[0] * v2.pt[1] + v1.pt[1] * v2.pt[0];    
}

GLfloat getAngle(const Vector &v1, const Vector &v2) {
    GLfloat cosine = dotProduct(v1, v2) / (v1.getMagnitude() * v2.getMagnitude());

    // correct for rounding errors
    if (cosine > 1)
        cosine = 1;   
    else if (cosine < -1)
        cosine = -1;
 
    if ((v1.pt[0] * v2.pt[1] - v1.pt[1] * v2.pt[0]) < 0)
        return -acos(cosine);
    else
        return acos(cosine);
}

Vector project(const Vector &v1, const Vector &v2) {
    return (dotProduct(v1, v2) / v1.getMagnitude2()) * v1;
}
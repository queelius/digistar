#include "Vector2D.h"

/************************
 * Vector2D constuctors *
 ************************/

Vector2D::Vector2D()
{
    /* do nothing */
}

Vector2D::Vector2D(const double v[DIMENSION_SIZE])
{
    copy(v);
}

Vector2D::Vector2D(double x, double y)
{
    copy(x, y);
}

Vector2D::Vector2D(const Vector2D &v)
{
    copy(v._components);
}

/********************
 * Vector2D methods *
 ********************/

void Vector2D::normalize()
{
    if (getMagnitudeSquared() == 0)
        throw Exception(DIVIDE_BY_ZERO_ERROR);

    getX() = getX() / getMagnitude();
    getY() = getY() / getMagnitude();
}

unsigned int Vector2D::getDimension() const
{
    return DIMENSION_SIZE;
}

double Vector2D::getAngleFromXAxis() const 
{
    return getAngleDifference(*this, Vector2D(1.0, 0));
}

void Vector2D::invert()
{
    getX() = -getX();
    getY() = -getY();
}

double Vector2D::getAngleFromYAxis() const
{
    return getAngleDifference(*this, Vector2D(0.0, 1.0));
}

double Vector2D::getLength() const
{
    return sqrt(getMagnitudeSquared());
}

Vector2D Vector2D::getNormalized() const
{
    if (getMagnitudeSquared() == 0)
        throw Exception(DIVIDE_BY_ZERO_ERROR);

    return *this / getLength();
}

double Vector2D::getMagnitude() const
{
    return getLength();
}

double &Vector2D::operator[](unsigned int index)
{
    if (index < DIMENSION_SIZE)
        return _components[index];
    else
        throw Exception(INDEX_OUT_OF_BOUNDS_ERROR);
}

double Vector2D::operator[](unsigned int index) const
{
    if (index < DIMENSION_SIZE)
        return _components[index];
    else
        throw Exception(INDEX_OUT_OF_BOUNDS_ERROR);
}

const Vector2D &Vector2D::operator=(const Vector2D &v)
{
    copy(v._components);
    return *this;
}

Vector2D Vector2D::operator-() const
{
    return Vector2D(-getX(), -getY());
}

double Vector2D::getX() const
{
    return _components[X_COMPONENT];
}

double Vector2D::getY() const
{
    return _components[Y_COMPONENT];
}

const double *Vector2D::toArray() const
{
    return _components;
}

void Vector2D::operator()(double x, double y)
{
    copy(x, y);
}

Vector2D Vector2D::getNormal() const
{
    return Vector2D(-getY(), getX()).getNormalized();
}

void Vector2D::operator()(const double v[DIMENSION_SIZE])
{
    copy(v);
}

void Vector2D::copy(const double v[DIMENSION_SIZE])
{
    _components[X_COMPONENT] = v[X_COMPONENT];
    _components[Y_COMPONENT] = v[Y_COMPONENT];
}

void Vector2D::copy(double x, double y)
{
    getX() = x;
    getY() = y;
}

void Vector2D::copy(const Vector2D &v)
{
    copy(v._components);
}



double &Vector2D::getX()
{
    return _components[X_COMPONENT];
}

double &Vector2D::getY()
{
    return _components[Y_COMPONENT];
}

double Vector2D::getMagnitudeSquared() const
{
    return getX() * getX() + getY() * getY();
}

Vector2D Vector2D::operator*=(double rhs)
{
    getX() *= rhs;
    getY() *= rhs;
    return *this;
}

Vector2D Vector2D::operator+=(const Vector2D &rhs)
{
    getX() += rhs.getX();
    getY() += rhs.getY();
    return *this;
}

Vector2D Vector2D::operator-=(const Vector2D &rhs)
{
    getX() -= rhs.getX();
    getY() -= rhs.getY();
    return *this;
}

Vector2D Vector2D::operator/=(double rhs)
{
    getX() /= rhs;
    getY() /= rhs;
    return *this;
}

/*****************************
 * Vector2D helper operators *
 *****************************/

bool operator==(const Vector2D &lhs, const Vector2D &rhs)
{
    return  lhs.getX() == rhs.getX() &&
            lhs.getY() == rhs.getY();
}

bool operator!=(const Vector2D &lhs, const Vector2D &rhs)
{
    return !(lhs == rhs);
}

bool operator<(const Vector2D &lhs, const Vector2D &rhs)
{
    return lhs.getMagnitudeSquared() < rhs.getMagnitudeSquared();
}

bool operator>(const Vector2D &lhs, const Vector2D &rhs)
{
    return lhs.getMagnitudeSquared() > rhs.getMagnitudeSquared();
}

bool operator<=(const Vector2D &lhs, const Vector2D &rhs)
{
    return lhs < rhs || lhs == rhs;
}

bool operator>=(const Vector2D &lhs, const Vector2D &rhs)
{
    return lhs > rhs || lhs == rhs;
}

Vector2D operator*(double lhs, const Vector2D &rhs)
{
    return Vector2D(lhs * rhs.getX(), lhs * rhs.getY());
}

Vector2D operator*(const Vector2D &lhs, double rhs)
{
    return rhs * lhs;
}

Vector2D operator/(const Vector2D &lhs, double rhs)
{
    if (rhs == 0)
        throw Exception(DIVIDE_BY_ZERO_ERROR);

    return (1 / rhs) * lhs;
}

Vector2D operator-(const Vector2D &lhs, const Vector2D &rhs) {
    return Vector2D(lhs.getX() - rhs.getX(), lhs.getY() - rhs.getY());
}

Vector2D operator+(const Vector2D &lhs, const Vector2D &rhs)
{
    return Vector2D(lhs.getX() + rhs.getX(), lhs.getY() + rhs.getY());
}

double operator*(const Vector2D &lhs, const Vector2D &rhs)
{
    return lhs.getX() * rhs.getX() + lhs.getY() * rhs.getY();
}

/*****************************
 * Vector2D helper functions *
 *****************************/

double getAngleDifference(const Vector2D &v1, const Vector2D &v2)
{
    if ((v1.getX() * v2.getY() - v1.getY() * v2.getX()) < 0)
        return -acos(getCosineOfAngleDifference(v1, v2));
    else
        return acos(getCosineOfAngleDifference(v1, v2));
}

bool isParallel(const Vector2D &v1, const Vector2D &v2)
{
    return v1 * v2 == v1.getMagnitude() * v2.getMagnitude();
}

bool isPerpindicular(const Vector2D &v1, const Vector2D &v2)
{
    return v1 * v2 == 0;
}

double getProjection(const Vector2D &v1, const Vector2D &v2)
{
    return v1 * v2 / v1.getMagnitude();
}

Vector2D getFromPolarCoordinates(double magnitude, double angle)
{
    Vector2D(magnitude * cos(angle), magnitude * sin(angle));
}

double getCrossProduct(const Vector2D &v1, const Vector2D &v2)
{
    return v1[0] * v2[1] - v1[1] * v2[0];
}

double getDotProduct(const Vector2D &v1, const Vector2D &v2)
{
    return v1 * v2;
}

double getCosineOfAngleDifference(const Vector2D &v1, const Vector2D &v2)
{
    double denom = v1.getLength() * v2.getLength();
    if (denom == 0)
        throw Exception(DIVIDE_BY_ZERO_ERROR);

    double cosine = v1 * v2 / denom;
    if (cosine > 1.0)
        cosine = 1.0;
    else if (cosine < -1.0)
        cosine = -1.0;

    return cosine;
}

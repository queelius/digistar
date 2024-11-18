#include "Vec2.h"

/************************
 * Vec2 constuctors *
 ************************/

Vec2::Vec2()
{
    /* do nothing */
}

Vec2::Vec2(const double v[DIMENSION_SIZE])
{
    copy(v);
}

Vec2::Vec2(double x, double y)
{
    copy(x, y);
}

Vec2::Vec2(const Vec2& v)
{
    copy(v._v);
}

/********************
 * Vec2 methods *
 ********************/

unsigned int Vec2::getDimension() const
{
    return DIMENSION_SIZE;
}

double Vec2::getAngleFromXAxis() const
{
	return getAngleDifference(*this, getXUnitVector());
};

double Vec2::getAngleFromYAxis() const
{
	return getAngleDifference(*this, getYUnitVector());
};

double Vec2::getLength() const
{
    return sqrt(getLength2());
}

Vec2 Vec2::getNormalized() const
{
    return *this / getLength();
}

double Vec2::getMagnitude() const
{
    return getLength();
}

const double& Vec2::operator[](int index) const
{
    return getElement(index);
}

double& Vec2::operator[](int index)
{
    return getElement(index);
}

Vec2& Vec2::operator=(const Vec2& v)
{
    copy(v._v);
    return *this;
}

Vec2 Vec2::operator-() const
{
    return Vec2(-_v[X_COMPONENT], -_v[Y_COMPONENT]);
}

void Vec2::operator()(const double v[DIMENSION_SIZE])
{
    copy(v);
}

const double* Vec2::toArray() const
{
    return _v;
}

void Vec2::operator()(double x, double y)
{
    _v[X_COMPONENT] = x;
    _v[Y_COMPONENT] = y;
}

void Vec2::copy(double x, double y)
{
    _v[0] = x;
    _v[1] = y;
}

////////////////TO DO//////////////
Vec2 Vec2::getNormal()
{
}

void Vec2::copy(const double v[DIMENSION_SIZE])
{
    _v[X_COMPONENT] = v[X_COMPONENT];
    _v[Y_COMPONENT] = v[Y_COMPONENT];
}

void Vec2::copy(const Vec2& v)
{
    copy(v._v);
}

double Vec2::getX() const
{
    return _v[X_COMPONENT];
}

double Vec2::getY() const
{
    return _v[Y_COMPONENT];
}

double Vec2::getElement(int index) const
{
    if (index < DIMENSION_SIZE)
        return _v[index];
    else
        throw std::exception("Invalid index");
}

double &Vec2::getElement(int index)
{
    if (index < DIMENSION_SIZE)
        return _v[index];
    else
        throw std::exception("Invalid index");

};

double Vec2::getLengthSquared() const
{
    return _v[X_COMPONENT] * _v[X_COMPONENT] +
		_v[Y_COMPONENT] * _v[Y_COMPONENT];
}

/*****************************
 * Vec2 helper operators *
 *****************************/

bool operator==(const Vec2& lhs, const Vec2& rhs)
{
    return  lhs.getX() == rhs.getX() &&
            lhs.getY() == rhs.getY();
}

bool operator!=(const Vec2& lhs, const Vec2& rhs)
{
    return !(lhs == rhs);
}

bool operator<(const Vec2& lhs, const Vec2& rhs)
{
    return lhs.getLength2() < rhs.getLength2();
}

bool operator>(const Vec2& lhs, const Vec2& rhs)
{
    return lhs.getLength2() > rhs.getLength2();
}

bool operator<=(const Vec2& lhs, const Vec2& rhs)
{
    return lhs < rhs || lhs == rhs;
}

bool operator>=(const Vec2& lhs, const Vec2& rhs)
{
    return lhs > rhs || lhs == rhs;
}

Vec2 operator*(double lhs, const Vec2& rhs)
{
    return Vec2(lhs * rhs.getX(), lhs * rhs.getY());
}

Vec2 operator*(const Vec2& lhs, double rhs)
{
    return rhs * lhs;
}

Vec2 operator/(const Vec2& lhs, double rhs)
{
    return (1 / rhs) * lhs;
}

Vec2 operator-(const Vec2& lhs, const Vec2& rhs) {
    return lhs + (-1 * rhs);
}

Vec2 operator+(const Vec2& lhs, const Vec2& rhs)
{
    return Vec2(lhs.getX() + rhs.getX(), lhs.getY() + rhs.getY());
}

double operator*(const Vec2& lhs, const Vec2& rhs)
{
    return lhs.getX() * rhs.getX() + lhs.getY() * rhs.getY();
}

/*****************************
 * Vec2 helper functions *
 *****************************/

double getAngleDifference(const Vec2&v1, const Vec2&v2)
{
    return acos(v1 * v2 / (v1.getLength() * v2.getLength()));
}

bool isParallel(const Vec2&v1, const Vec2&v2)
{
	return true;
}

bool isPerpindicular(const Vec2&v1, const Vec2&v2)
{
    return v1 * v2 == 0;
}

///////////////TO DO/////////////////
Vec2 getProjection(const Vec2&v1, const Vec2&v2)
{
}

///////////////TO DO/////////////////
Vec2 getFromPolarCoordinates(double magnitude, double angle)
{
}

///////////////TO DO/////////////////
Vec2 getXUnitVector()
{
}

///////////////TO DO/////////////////
Vec2 getYUnitVector()
{
}

///////////////TO DO/////////////////
Point2D operator+(const Point2D & lhs, const Vec2& rhs)
{
}

///////////////TO DO/////////////////
Vec2 operator-(const Point2D & lhs, const Point2D & rhs)
{
}

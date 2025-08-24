#pragma once

#include <cmath>
#include <ostream>

namespace digistar {

class Vec2 {
public:
    double x, y;
    
    // Constructors
    Vec2() : x(0), y(0) {}
    Vec2(double x_, double y_) : x(x_), y(y_) {}
    
    // Operators
    Vec2 operator+(const Vec2& v) const { return Vec2(x + v.x, y + v.y); }
    Vec2 operator-(const Vec2& v) const { return Vec2(x - v.x, y - v.y); }
    Vec2 operator*(double s) const { return Vec2(x * s, y * s); }
    Vec2 operator/(double s) const { return Vec2(x / s, y / s); }
    Vec2 operator-() const { return Vec2(-x, -y); }
    
    Vec2& operator+=(const Vec2& v) { x += v.x; y += v.y; return *this; }
    Vec2& operator-=(const Vec2& v) { x -= v.x; y -= v.y; return *this; }
    Vec2& operator*=(double s) { x *= s; y *= s; return *this; }
    Vec2& operator/=(double s) { x /= s; y /= s; return *this; }
    
    bool operator==(const Vec2& v) const { 
        return std::abs(x - v.x) < 1e-10 && std::abs(y - v.y) < 1e-10; 
    }
    bool operator!=(const Vec2& v) const { return !(*this == v); }
    
    // Methods
    double dot(const Vec2& v) const { return x * v.x + y * v.y; }
    double cross(const Vec2& v) const { return x * v.y - y * v.x; }  // 2D cross product (scalar)
    
    double length() const { return std::sqrt(x * x + y * y); }
    double lengthSquared() const { return x * x + y * y; }
    
    Vec2 normalized() const {
        double len = length();
        return (len > 0) ? (*this / len) : Vec2();
    }
    
    void normalize() {
        double len = length();
        if (len > 0) {
            *this /= len;
        }
    }
    
    // Perpendicular vector (rotated 90 degrees counter-clockwise)
    Vec2 perp() const { return Vec2(-y, x); }
    
    // Rotate by angle (radians)
    Vec2 rotated(double angle) const {
        double c = std::cos(angle);
        double s = std::sin(angle);
        return Vec2(x * c - y * s, x * s + y * c);
    }
};

// Non-member operators
inline Vec2 operator*(double s, const Vec2& v) { return v * s; }

// Stream output
inline std::ostream& operator<<(std::ostream& os, const Vec2& v) {
    os << "[" << v.x << ", " << v.y << "]";
    return os;
}

// Utility functions
inline double distance(const Vec2& a, const Vec2& b) {
    return (a - b).length();
}

inline double distanceSquared(const Vec2& a, const Vec2& b) {
    return (a - b).lengthSquared();
}

// Angle between two vectors (radians)
inline double angle(const Vec2& a, const Vec2& b) {
    return std::atan2(a.cross(b), a.dot(b));
}

} // namespace digistar
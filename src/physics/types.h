#pragma once

#include <cstdint>
#include <cmath>

namespace digistar {

// Basic 2D vector
struct Vector2 {
    float x, y;
    
    Vector2() : x(0), y(0) {}
    Vector2(float x_, float y_) : x(x_), y(y_) {}
    
    Vector2 operator+(const Vector2& v) const { return Vector2(x + v.x, y + v.y); }
    Vector2 operator-(const Vector2& v) const { return Vector2(x - v.x, y - v.y); }
    Vector2 operator*(float s) const { return Vector2(x * s, y * s); }
    Vector2 operator/(float s) const { return Vector2(x / s, y / s); }
    
    Vector2& operator+=(const Vector2& v) { x += v.x; y += v.y; return *this; }
    Vector2& operator-=(const Vector2& v) { x -= v.x; y -= v.y; return *this; }
    Vector2& operator*=(float s) { x *= s; y *= s; return *this; }
    Vector2& operator/=(float s) { x /= s; y /= s; return *this; }
    
    float length() const { return std::sqrt(x * x + y * y); }
    float length_squared() const { return x * x + y * y; }
    Vector2 normalized() const { float l = length(); return l > 0 ? (*this) / l : Vector2(0, 0); }
    
    float dot(const Vector2& v) const { return x * v.x + y * v.y; }
    float cross(const Vector2& v) const { return x * v.y - y * v.x; }  // 2D cross product (scalar)
};

// Material types
enum MaterialType : uint8_t {
    MATERIAL_GENERIC = 0,
    MATERIAL_METAL = 1,
    MATERIAL_ROCK = 2,
    MATERIAL_ICE = 3,
    MATERIAL_GAS = 4,
    MATERIAL_PLASMA = 5,
    MATERIAL_ORGANIC = 6,
    MATERIAL_COMPOSITE = 7
};

// Integration methods
enum IntegratorType : uint8_t {
    INTEGRATOR_VELOCITY_VERLET = 0,
    INTEGRATOR_SEMI_IMPLICIT = 1,
    INTEGRATOR_LEAPFROG = 2,
    INTEGRATOR_FORWARD_EULER = 3,  // For comparison only - not recommended
    INTEGRATOR_RK4 = 4              // Future
};

// Particle state flags (can be combined)
enum ParticleFlags : uint8_t {
    PARTICLE_ACTIVE = 1 << 0,
    PARTICLE_FIXED = 1 << 1,      // Position is fixed (boundary condition)
    PARTICLE_NO_GRAVITY = 1 << 2,
    PARTICLE_NO_COLLISION = 1 << 3,
    PARTICLE_HIGH_PRIORITY = 1 << 4,  // Update more frequently
    PARTICLE_MARKED_FOR_FUSION = 1 << 5,
    PARTICLE_MARKED_FOR_FISSION = 1 << 6
};

// Constants
namespace Constants {
    constexpr float G = 6.67430e-11f;  // Gravitational constant
    constexpr float c = 299792458.0f;   // Speed of light
    constexpr float k_B = 1.380649e-23f; // Boltzmann constant
    constexpr float sigma = 5.670374419e-8f; // Stefan-Boltzmann constant
    
    // Simulation defaults
    constexpr float DEFAULT_TIMESTEP = 0.01f;
    constexpr float MAX_VELOCITY = 1000.0f;
    constexpr float MIN_TEMPERATURE = 0.0f;
    constexpr float MAX_TEMPERATURE = 10000.0f;
    
    // Spring defaults
    constexpr float DEFAULT_SPRING_STIFFNESS = 100.0f;
    constexpr float DEFAULT_SPRING_DAMPING = 1.0f;
    constexpr float DEFAULT_BREAK_STRAIN = 0.5f;
    
    // Contact defaults
    constexpr float DEFAULT_CONTACT_STIFFNESS = 1000.0f;
    constexpr float DEFAULT_CONTACT_DAMPING = 10.0f;
}

// Inline utility functions
inline Vector2 min_distance(const Vector2& p1, const Vector2& p2, float world_size) {
    Vector2 diff = p2 - p1;
    
    // Handle toroidal wrapping
    if (diff.x > world_size * 0.5f) diff.x -= world_size;
    if (diff.x < -world_size * 0.5f) diff.x += world_size;
    if (diff.y > world_size * 0.5f) diff.y -= world_size;
    if (diff.y < -world_size * 0.5f) diff.y += world_size;
    
    return diff;
}

inline void wrap_position(Vector2& pos, float world_size) {
    while (pos.x < 0) pos.x += world_size;
    while (pos.x >= world_size) pos.x -= world_size;
    while (pos.y < 0) pos.y += world_size;
    while (pos.y >= world_size) pos.y -= world_size;
}

} // namespace digistar
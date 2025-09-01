#pragma once

// Simplified standalone backend for examples
// This provides a basic particle simulation interface without requiring the full backend infrastructure

#include <vector>
#include <cmath>
#include <memory>
#include <algorithm>

// Basic 2D math types
struct float2 {
    float x, y;
    
    float2() : x(0), y(0) {}
    float2(float x_, float y_) : x(x_), y(y_) {}
    
    float2 operator+(const float2& other) const { return float2(x + other.x, y + other.y); }
    float2 operator-(const float2& other) const { return float2(x - other.x, y - other.y); }
    float2 operator*(float scalar) const { return float2(x * scalar, y * scalar); }
    float2 operator/(float scalar) const { return float2(x / scalar, y / scalar); }
};

// Particle structure
struct Particle {
    float2 pos;
    float2 vel;
    float mass;
    float radius;
};

// Simulation parameters
struct SimulationParams {
    float box_size = 1000.0f;
    float gravity_constant = 1.0f;
    float softening = 0.1f;
    float dt = 0.016f;
    int grid_size = 512;
};

// Force calculation algorithms
enum class ForceAlgorithm {
    BRUTE_FORCE,
    PARTICLE_MESH
};

// Simple backend for examples
class SimpleBackend {
private:
    std::vector<Particle> particles;
    std::vector<float2> forces;
    SimulationParams params;
    ForceAlgorithm algorithm = ForceAlgorithm::BRUTE_FORCE;
    size_t max_particles;
    
    // FFT workspace for Particle Mesh (if needed)
    void* pm_workspace = nullptr;
    
public:
    SimpleBackend() : max_particles(1000000) {}
    ~SimpleBackend() { cleanup(); }
    
    void initialize(size_t num_particles, const SimulationParams& p) {
        params = p;
        particles.resize(num_particles);
        forces.resize(num_particles);
        max_particles = num_particles;
        
        if (algorithm == ForceAlgorithm::PARTICLE_MESH) {
            initializePM();
        }
    }
    
    void setAlgorithm(ForceAlgorithm algo) {
        algorithm = algo;
        if (algorithm == ForceAlgorithm::PARTICLE_MESH && particles.size() > 0) {
            initializePM();
        }
    }
    
    void setParticles(const std::vector<Particle>& p) {
        particles = p;
        forces.resize(particles.size());
    }
    
    void getParticles(std::vector<Particle>& p) {
        p = particles;
    }
    
    void step(float dt) {
        computeForces();
        integrate(dt);
    }
    
    void computeForces() {
        // Clear forces
        for (auto& f : forces) {
            f.x = f.y = 0;
        }
        
        if (algorithm == ForceAlgorithm::PARTICLE_MESH) {
            computeForcePM();
        } else {
            computeForceBrute();
        }
    }
    
    void integrate(float dt) {
        size_t n = particles.size();
        
        for (size_t i = 0; i < n; i++) {
            // Update velocity (forces array contains accelerations)
            particles[i].vel.x += forces[i].x * dt;
            particles[i].vel.y += forces[i].y * dt;
            
            // Update position
            particles[i].pos.x += particles[i].vel.x * dt;
            particles[i].pos.y += particles[i].vel.y * dt;
            
            // Periodic boundary conditions
            if (particles[i].pos.x < 0) particles[i].pos.x += params.box_size;
            if (particles[i].pos.x >= params.box_size) particles[i].pos.x -= params.box_size;
            if (particles[i].pos.y < 0) particles[i].pos.y += params.box_size;
            if (particles[i].pos.y >= params.box_size) particles[i].pos.y -= params.box_size;
        }
    }
    
    void cleanup() {
        particles.clear();
        forces.clear();
        if (pm_workspace) {
            // Cleanup would go here
            pm_workspace = nullptr;
        }
    }
    
private:
    void computeForceBrute() {
        size_t n = particles.size();
        const float G = params.gravity_constant;
        const float soft2 = params.softening * params.softening;
        
        for (size_t i = 0; i < n; i++) {
            float fx = 0, fy = 0;
            
            for (size_t j = 0; j < n; j++) {
                if (i == j) continue;
                
                float dx = particles[j].pos.x - particles[i].pos.x;
                float dy = particles[j].pos.y - particles[i].pos.y;
                
                // Handle periodic boundaries
                if (dx > params.box_size * 0.5f) dx -= params.box_size;
                if (dx < -params.box_size * 0.5f) dx += params.box_size;
                if (dy > params.box_size * 0.5f) dy -= params.box_size;
                if (dy < -params.box_size * 0.5f) dy += params.box_size;
                
                float r2 = dx*dx + dy*dy + soft2;
                float r = sqrt(r2);
                float a = G * particles[j].mass / (r2 * r);
                
                fx += a * dx;
                fy += a * dy;
            }
            
            forces[i].x = fx;
            forces[i].y = fy;
        }
    }
    
    void computeForcePM() {
        // Simplified particle mesh - for now just use brute force
        // A real implementation would use FFT
        computeForceBrute();
    }
    
    void initializePM() {
        // Would initialize FFT workspace here
        // For now, this is a placeholder
    }
};
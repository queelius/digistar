#pragma once

#include <unordered_map>
#include <vector>
#include <functional>
#include <mutex>
#include "vec2.h"

namespace digistar {

// Simple particle structure for DSL integration
struct Particle {
    int id;
    double mass;
    Vec2 position;
    Vec2 velocity;
    double temperature;
    bool active = true;
};

// Simple spring structure
struct Spring {
    int id;
    int particle1_id;
    int particle2_id;
    double stiffness;
    double damping;
    double equilibrium_distance;
    bool active = true;
};

// Simplified simulation class for DSL testing
class Simulation {
private:
    std::unordered_map<int, Particle> particles;
    std::unordered_map<int, Spring> springs;
    
    // ID management
    int next_particle_id = 1;
    int next_spring_id = 1;
    
    // Provisional ID mapping
    std::unordered_map<int, int> provisional_to_actual;
    mutable std::mutex id_map_mutex;
    
    // Simulation parameters
    double dt = 0.01;
    double gravity_strength = 9.81;
    Vec2 gravity_direction{0, -1};
    
    // Statistics
    size_t total_particles_created = 0;
    size_t total_springs_created = 0;
    
public:
    Simulation() = default;
    
    // Particle operations
    int createParticle(double mass, Vec2 pos, Vec2 vel, double temp = 300);
    void destroyParticle(int id);
    Particle* getParticle(int id);
    const Particle* getParticle(int id) const;
    
    // Spring operations
    int createSpring(int p1_id, int p2_id, double stiffness, 
                    double damping, double equilibrium = -1);
    void breakSpring(int id);
    Spring* getSpring(int id);
    const Spring* getSpring(int id) const;
    
    // ID mapping for provisional IDs
    void mapProvisionalId(int provisional, int actual);
    int resolveId(int id) const;
    
    // Iteration helpers
    void forEachParticle(std::function<void(int, const Particle&)> func) const;
    void forEachSpring(std::function<void(int, const Spring&)> func) const;
    
    // Find particles in region
    std::vector<int> findParticlesInRadius(Vec2 center, double radius) const;
    
    // Simulation parameters
    double getDt() const { return dt; }
    void setDt(double new_dt) { dt = new_dt; }
    
    double getGravity() const { return gravity_strength; }
    void setGravity(double g) { gravity_strength = g; }
    
    // Statistics
    size_t getParticleCount() const;
    size_t getSpringCount() const;
    size_t getTotalParticlesCreated() const { return total_particles_created; }
    size_t getTotalSpringsCreated() const { return total_springs_created; }
    
    // Physics update (simple integration for testing)
    void update(double dt);
    
    // Clear all particles and springs
    void clear();
};

} // namespace digistar
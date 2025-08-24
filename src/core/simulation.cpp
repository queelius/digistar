#include "simulation.h"
#include <cmath>

namespace digistar {

int Simulation::createParticle(double mass, Vec2 pos, Vec2 vel, double temp) {
    int id = next_particle_id++;
    particles[id] = Particle{id, mass, pos, vel, temp, true};
    total_particles_created++;
    return id;
}

void Simulation::destroyParticle(int id) {
    auto it = particles.find(id);
    if (it != particles.end()) {
        it->second.active = false;
        particles.erase(it);
        
        // Also break any springs connected to this particle
        std::vector<int> springs_to_break;
        for (auto& [sid, spring] : springs) {
            if (spring.particle1_id == id || spring.particle2_id == id) {
                springs_to_break.push_back(sid);
            }
        }
        for (int sid : springs_to_break) {
            breakSpring(sid);
        }
    }
}

Particle* Simulation::getParticle(int id) {
    auto it = particles.find(id);
    return (it != particles.end()) ? &it->second : nullptr;
}

const Particle* Simulation::getParticle(int id) const {
    auto it = particles.find(id);
    return (it != particles.end()) ? &it->second : nullptr;
}

int Simulation::createSpring(int p1_id, int p2_id, double stiffness, 
                            double damping, double equilibrium) {
    // Resolve provisional IDs
    p1_id = resolveId(p1_id);
    p2_id = resolveId(p2_id);
    
    auto* p1 = getParticle(p1_id);
    auto* p2 = getParticle(p2_id);
    
    if (!p1 || !p2) {
        return -1;  // Invalid particles
    }
    
    // Calculate equilibrium distance if not specified
    if (equilibrium < 0) {
        equilibrium = (p1->position - p2->position).length();
    }
    
    int id = next_spring_id++;
    springs[id] = Spring{id, p1_id, p2_id, stiffness, damping, equilibrium, true};
    total_springs_created++;
    return id;
}

void Simulation::breakSpring(int id) {
    auto it = springs.find(id);
    if (it != springs.end()) {
        it->second.active = false;
        springs.erase(it);
    }
}

Spring* Simulation::getSpring(int id) {
    auto it = springs.find(id);
    return (it != springs.end()) ? &it->second : nullptr;
}

const Spring* Simulation::getSpring(int id) const {
    auto it = springs.find(id);
    return (it != springs.end()) ? &it->second : nullptr;
}

void Simulation::mapProvisionalId(int provisional, int actual) {
    std::lock_guard<std::mutex> lock(id_map_mutex);
    provisional_to_actual[provisional] = actual;
}

int Simulation::resolveId(int id) const {
    std::lock_guard<std::mutex> lock(id_map_mutex);
    auto it = provisional_to_actual.find(id);
    if (it != provisional_to_actual.end()) {
        return it->second;
    }
    return id;  // Assume it's already an actual ID
}

void Simulation::forEachParticle(std::function<void(int, const Particle&)> func) const {
    for (const auto& [id, particle] : particles) {
        if (particle.active) {
            func(id, particle);
        }
    }
}

void Simulation::forEachSpring(std::function<void(int, const Spring&)> func) const {
    for (const auto& [id, spring] : springs) {
        if (spring.active) {
            func(id, spring);
        }
    }
}

std::vector<int> Simulation::findParticlesInRadius(Vec2 center, double radius) const {
    std::vector<int> result;
    double radius_sq = radius * radius;
    
    for (const auto& [id, particle] : particles) {
        if (particle.active) {
            double dist_sq = (particle.position - center).lengthSquared();
            if (dist_sq <= radius_sq) {
                result.push_back(id);
            }
        }
    }
    
    return result;
}

size_t Simulation::getParticleCount() const {
    size_t count = 0;
    for (const auto& [id, p] : particles) {
        if (p.active) count++;
    }
    return count;
}

size_t Simulation::getSpringCount() const {
    size_t count = 0;
    for (const auto& [id, s] : springs) {
        if (s.active) count++;
    }
    return count;
}

void Simulation::update(double dt) {
    // Simple Euler integration for testing
    
    // Apply gravity
    Vec2 gravity_force = gravity_direction * gravity_strength;
    
    // Update velocities and positions
    for (auto& [id, particle] : particles) {
        if (!particle.active) continue;
        
        // Apply gravity
        particle.velocity += gravity_force * dt;
        
        // Update position
        particle.position += particle.velocity * dt;
    }
    
    // Apply spring forces
    for (const auto& [id, spring] : springs) {
        if (!spring.active) continue;
        
        auto* p1 = getParticle(spring.particle1_id);
        auto* p2 = getParticle(spring.particle2_id);
        
        if (!p1 || !p2) continue;
        
        Vec2 delta = p2->position - p1->position;
        double distance = delta.length();
        
        if (distance > 0) {
            Vec2 direction = delta / distance;
            
            // Spring force: F = -k * (distance - equilibrium)
            double spring_force = spring.stiffness * (distance - spring.equilibrium_distance);
            
            // Damping force: F = -d * relative_velocity
            Vec2 relative_vel = p2->velocity - p1->velocity;
            double damping_force = spring.damping * relative_vel.dot(direction);
            
            double total_force = spring_force + damping_force;
            Vec2 force = direction * total_force;
            
            // Apply equal and opposite forces
            p1->velocity += force * (dt / p1->mass);
            p2->velocity -= force * (dt / p2->mass);
        }
    }
}

void Simulation::clear() {
    particles.clear();
    springs.clear();
    provisional_to_actual.clear();
    next_particle_id = 1;
    next_spring_id = 1;
}

} // namespace digistar
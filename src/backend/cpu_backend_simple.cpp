#include "cpu_backend_simple.h"
#include <cmath>
#include <algorithm>

namespace digistar {

void CpuBackendSimple::initialize(const SimulationConfig& cfg) {
    config = cfg;
    stats = SimulationStats();
}

void CpuBackendSimple::shutdown() {
    // Nothing to clean up for simple backend
}

void CpuBackendSimple::step(SimulationState& state, const PhysicsConfig& physics_config, float dt) {
    last_step_start = std::chrono::high_resolution_clock::now();
    
    // Clear forces
    state.particles.clear_forces();
    
    // Compute gravity if enabled
    if (physics_config.enabled_systems & PhysicsConfig::GRAVITY) {
        auto t1 = std::chrono::high_resolution_clock::now();
        computeGravityDirect(state.particles, physics_config.gravity_strength);
        auto t2 = std::chrono::high_resolution_clock::now();
        stats.gravity_time_ms = std::chrono::duration<float, std::milli>(t2 - t1).count();
    }
    
    // Compute contacts if enabled
    if (physics_config.enabled_systems & PhysicsConfig::CONTACTS) {
        auto t1 = std::chrono::high_resolution_clock::now();
        computeContacts(state.particles, state.contacts, 
                       physics_config.contact_stiffness, 
                       physics_config.contact_damping);
        auto t2 = std::chrono::high_resolution_clock::now();
        stats.contact_time_ms = std::chrono::duration<float, std::milli>(t2 - t1).count();
    }
    
    // Integrate positions and velocities
    auto t1 = std::chrono::high_resolution_clock::now();
    integrateSemiImplicit(state.particles, dt);
    auto t2 = std::chrono::high_resolution_clock::now();
    stats.integration_time_ms = std::chrono::duration<float, std::milli>(t2 - t1).count();
    
    // Update statistics
    auto now = std::chrono::high_resolution_clock::now();
    stats.update_time_ms = std::chrono::duration<float, std::milli>(now - last_step_start).count();
    stats.active_particles = state.particles.count;
    stats.active_contacts = state.contacts.count;
}

void CpuBackendSimple::computeGravityDirect(ParticlePool& particles, float gravity_constant) {
    size_t n = particles.count;
    if (n == 0) return;
    
    const float softening2 = 1.0f; // Small softening to prevent singularities
    
    // O(n^2) direct gravity calculation
    for (size_t i = 0; i < n; i++) {
        uint32_t idx_i = particles.active_indices[i];
        float fx = 0, fy = 0;
        
        for (size_t j = 0; j < n; j++) {
            if (i == j) continue;
            
            uint32_t idx_j = particles.active_indices[j];
            
            float dx = particles.pos_x[idx_j] - particles.pos_x[idx_i];
            float dy = particles.pos_y[idx_j] - particles.pos_y[idx_i];
            
            // Handle toroidal boundary if enabled
            if (config.use_toroidal) {
                if (dx > config.world_size * 0.5f) dx -= config.world_size;
                if (dx < -config.world_size * 0.5f) dx += config.world_size;
                if (dy > config.world_size * 0.5f) dy -= config.world_size;
                if (dy < -config.world_size * 0.5f) dy += config.world_size;
            }
            
            float r2 = dx * dx + dy * dy + softening2;
            float r = std::sqrt(r2);
            float f = gravity_constant * particles.mass[idx_j] / (r2 * r);
            
            fx += f * dx;
            fy += f * dy;
        }
        
        particles.force_x[idx_i] += fx * particles.mass[idx_i];
        particles.force_y[idx_i] += fy * particles.mass[idx_i];
    }
}

void CpuBackendSimple::computeContacts(ParticlePool& particles, ContactPool& contacts, 
                                       float stiffness, float damping) {
    // Clear existing contacts
    contacts.clear();
    
    size_t n = particles.count;
    if (n == 0) return;
    
    // Simple O(n^2) contact detection and force calculation
    for (size_t i = 0; i < n; i++) {
        uint32_t idx_i = particles.active_indices[i];
        
        for (size_t j = i + 1; j < n; j++) {
            uint32_t idx_j = particles.active_indices[j];
            
            float dx = particles.pos_x[idx_j] - particles.pos_x[idx_i];
            float dy = particles.pos_y[idx_j] - particles.pos_y[idx_i];
            
            // Handle toroidal boundary if enabled
            if (config.use_toroidal) {
                if (dx > config.world_size * 0.5f) dx -= config.world_size;
                if (dx < -config.world_size * 0.5f) dx += config.world_size;
                if (dy > config.world_size * 0.5f) dy -= config.world_size;
                if (dy < -config.world_size * 0.5f) dy += config.world_size;
            }
            
            float dist2 = dx * dx + dy * dy;
            float min_dist = particles.radius[idx_i] + particles.radius[idx_j];
            float min_dist2 = min_dist * min_dist;
            
            if (dist2 < min_dist2 && dist2 > 0) {
                float dist = std::sqrt(dist2);
                float overlap = min_dist - dist;
                
                // Normal vector
                float nx = dx / dist;
                float ny = dy / dist;
                
                // Relative velocity
                float dvx = particles.vel_x[idx_j] - particles.vel_x[idx_i];
                float dvy = particles.vel_y[idx_j] - particles.vel_y[idx_i];
                float dv_normal = dvx * nx + dvy * ny;
                
                // Contact force (spring-damper model)
                float force = stiffness * overlap - damping * dv_normal;
                if (force < 0) force = 0; // No attractive contact forces
                
                // Apply forces
                float fx = force * nx;
                float fy = force * ny;
                
                particles.force_x[idx_i] -= fx;
                particles.force_y[idx_i] -= fy;
                particles.force_x[idx_j] += fx;
                particles.force_y[idx_j] += fy;
                
                // Record contact if there's room
                if (contacts.count < config.max_contacts) {
                    uint32_t contact_idx = contacts.count++;
                    contacts.particle1[contact_idx] = idx_i;
                    contacts.particle2[contact_idx] = idx_j;
                    contacts.normal_x[contact_idx] = nx;
                    contacts.normal_y[contact_idx] = ny;
                    contacts.overlap[contact_idx] = overlap;
                }
            }
        }
    }
}

void CpuBackendSimple::integrateSemiImplicit(ParticlePool& particles, float dt) {
    size_t n = particles.count;
    if (n == 0) return;
    
    for (size_t i = 0; i < n; i++) {
        uint32_t idx = particles.active_indices[i];
        
        // Skip fixed particles
        if (!particles.alive[idx]) {
            continue;
        }
        
        // Update velocity (F = ma, so a = F/m)
        particles.vel_x[idx] += (particles.force_x[idx] / particles.mass[idx]) * dt;
        particles.vel_y[idx] += (particles.force_y[idx] / particles.mass[idx]) * dt;
        
        // Clamp velocity to prevent instabilities
        const float max_vel = 1000.0f;
        float vx = particles.vel_x[idx];
        float vy = particles.vel_y[idx];
        float v2 = vx * vx + vy * vy;
        if (v2 > max_vel * max_vel) {
            float v = std::sqrt(v2);
            particles.vel_x[idx] = (vx / v) * max_vel;
            particles.vel_y[idx] = (vy / v) * max_vel;
        }
        
        // Update position
        particles.pos_x[idx] += particles.vel_x[idx] * dt;
        particles.pos_y[idx] += particles.vel_y[idx] * dt;
        
        // Handle toroidal boundary conditions
        if (config.use_toroidal) {
            while (particles.pos_x[idx] < 0) 
                particles.pos_x[idx] += config.world_size;
            while (particles.pos_x[idx] >= config.world_size) 
                particles.pos_x[idx] -= config.world_size;
            while (particles.pos_y[idx] < 0) 
                particles.pos_y[idx] += config.world_size;
            while (particles.pos_y[idx] >= config.world_size) 
                particles.pos_y[idx] -= config.world_size;
        }
    }
}

} // namespace digistar
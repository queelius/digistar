#pragma once

#include <vector>
#include <random>
#include <algorithm>
#include "../backend/ISimulationBackend.h"

// Extended particle with temperature for merging
struct ParticleExt : public Particle {
    float temp_internal = 300.0f;  // Internal temperature (K)
    float temp_kinetic = 0.0f;     // Kinetic temperature from motion
    bool is_merged = false;        // Mark for removal after merging
    int merge_target = -1;          // Index of particle to merge with
};

class ParticleMerger {
protected:
    // Merge parameters
    float overlap_threshold = 0.5f;     // Fraction of radius sum for overlap
    float velocity_threshold = 10.0f;   // Max relative velocity for merging (m/s)
    float merge_probability = 0.1f;     // Base probability when conditions met
    float specific_heat = 1000.0f;      // J/(kg·K) - specific heat capacity
    
private:
    
    std::mt19937 rng;
    std::uniform_real_distribution<float> dist{0.0f, 1.0f};
    
public:
    ParticleMerger() : rng(std::random_device{}()) {}
    
    // Configuration
    void setOverlapThreshold(float threshold) { overlap_threshold = threshold; }
    void setVelocityThreshold(float threshold) { velocity_threshold = threshold; }
    void setMergeProbability(float prob) { merge_probability = prob; }
    
    // Check if two particles should merge
    bool shouldMerge(const ParticleExt& p1, const ParticleExt& p2, float distance) {
        // Already marked for merging
        if (p1.is_merged || p2.is_merged) return false;
        
        // Check overlap
        float radius_sum = p1.radius + p2.radius;
        if (distance > overlap_threshold * radius_sum) return false;
        
        // Check relative velocity
        float2 v_rel = p1.vel - p2.vel;
        float v_rel_mag = sqrt(v_rel.x * v_rel.x + v_rel.y * v_rel.y);
        if (v_rel_mag > velocity_threshold) return false;
        
        // Temperature check - hot particles less likely to merge
        float temp_factor = 1.0f / (1.0f + (p1.temp_internal + p2.temp_internal) / 10000.0f);
        
        // Stochastic merge with probability
        float prob = merge_probability * temp_factor;
        return dist(rng) < prob;
    }
    
    // Merge particle p2 into p1, conserving physical quantities
    ParticleExt mergeParticles(const ParticleExt& p1, const ParticleExt& p2) {
        ParticleExt result;
        
        // Conservation of mass
        result.mass = p1.mass + p2.mass;
        
        // Conservation of momentum -> weighted average position
        result.pos.x = (p1.mass * p1.pos.x + p2.mass * p2.pos.x) / result.mass;
        result.pos.y = (p1.mass * p1.pos.y + p2.mass * p2.pos.y) / result.mass;
        
        // Conservation of momentum -> velocity
        float2 momentum;
        momentum.x = p1.mass * p1.vel.x + p2.mass * p2.vel.x;
        momentum.y = p1.mass * p1.vel.y + p2.mass * p2.vel.y;
        result.vel = momentum / result.mass;
        
        // Volume conservation -> new radius
        float volume = pow(p1.radius, 3) + pow(p2.radius, 3);
        result.radius = cbrt(volume);
        
        // Energy handling: kinetic energy lost becomes internal heat
        float2 v_rel = p1.vel - p2.vel;
        float v_rel_sq = v_rel.x * v_rel.x + v_rel.y * v_rel.y;
        float ke_lost = 0.5f * (p1.mass * p2.mass) / result.mass * v_rel_sq;
        
        // Temperature: weighted average plus heating from collision
        float temp_avg = (p1.mass * p1.temp_internal + p2.mass * p2.temp_internal) / result.mass;
        float temp_increase = ke_lost / (specific_heat * result.mass);
        result.temp_internal = temp_avg + temp_increase;
        
        // Kinetic temperature will be recalculated based on local mean
        result.temp_kinetic = 0.0f;
        
        return result;
    }
    
    // Process all particles and identify merge pairs
    std::vector<std::pair<int, int>> findMergePairs(std::vector<ParticleExt>& particles) {
        std::vector<std::pair<int, int>> merge_pairs;
        
        // Simple O(n²) for now - should use spatial hashing for efficiency
        for (size_t i = 0; i < particles.size(); i++) {
            if (particles[i].is_merged) continue;
            
            for (size_t j = i + 1; j < particles.size(); j++) {
                if (particles[j].is_merged) continue;
                
                // Calculate distance
                float2 diff = particles[i].pos - particles[j].pos;
                float dist = sqrt(diff.x * diff.x + diff.y * diff.y);
                
                if (shouldMerge(particles[i], particles[j], dist)) {
                    merge_pairs.push_back({i, j});
                    particles[i].merge_target = j;
                    particles[j].is_merged = true;  // Mark j for removal
                    break;  // Each particle can only merge once per step
                }
            }
        }
        
        return merge_pairs;
    }
    
    // Execute merges and return new particle list
    std::vector<ParticleExt> executeMerges(const std::vector<ParticleExt>& particles,
                                           const std::vector<std::pair<int, int>>& merge_pairs) {
        std::vector<ParticleExt> new_particles;
        new_particles.reserve(particles.size());
        
        // Create merged particles
        for (const auto& [i, j] : merge_pairs) {
            ParticleExt merged = mergeParticles(particles[i], particles[j]);
            new_particles.push_back(merged);
        }
        
        // Add unmerged particles
        for (size_t i = 0; i < particles.size(); i++) {
            if (!particles[i].is_merged && particles[i].merge_target == -1) {
                new_particles.push_back(particles[i]);
            }
        }
        
        return new_particles;
    }
    
    // Main merge step
    std::vector<ParticleExt> mergingStep(std::vector<ParticleExt>& particles) {
        // Find pairs to merge
        auto merge_pairs = findMergePairs(particles);
        
        if (merge_pairs.empty()) {
            return particles;  // No merges needed
        }
        
        // Execute merges
        return executeMerges(particles, merge_pairs);
    }
    
    // Calculate local kinetic temperature
    void updateKineticTemperatures(std::vector<ParticleExt>& particles, float cell_size = 50.0f) {
        // For each particle, calculate temperature from velocity relative to local mean
        // This is a simplified version - proper implementation would use spatial hashing
        
        for (auto& p : particles) {
            float2 local_mean_vel = {0, 0};
            float total_mass = 0;
            int neighbor_count = 0;
            
            // Find local neighbors
            for (const auto& other : particles) {
                float2 diff = p.pos - other.pos;
                float dist_sq = diff.x * diff.x + diff.y * diff.y;
                
                if (dist_sq < cell_size * cell_size) {
                    local_mean_vel.x += other.vel.x * other.mass;
                    local_mean_vel.y += other.vel.y * other.mass;
                    total_mass += other.mass;
                    neighbor_count++;
                }
            }
            
            if (neighbor_count > 0 && total_mass > 0) {
                local_mean_vel = local_mean_vel / total_mass;
                float2 v_rel = p.vel - local_mean_vel;
                float v_rel_sq = v_rel.x * v_rel.x + v_rel.y * v_rel.y;
                
                // Kinetic temperature proportional to velocity squared
                // Using rough conversion: 1/2 * m * v² = 3/2 * k * T
                // T = m * v² / (3 * k), where k is Boltzmann constant
                const float k_boltzmann = 1.38e-23f;
                p.temp_kinetic = p.mass * v_rel_sq / (3.0f * k_boltzmann);
            }
        }
    }
};
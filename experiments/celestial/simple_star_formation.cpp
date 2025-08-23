#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>
#include "../../src/dynamics/ParticleMerger.h"

// Simple star formation test with extreme parameters to ensure star formation

int main() {
    std::cout << "=== Simplified Star Formation Test ===\n\n";
    std::cout << "Creating a dense cluster that will collapse and heat up to form a star.\n\n";
    
    // Create very dense initial cluster
    std::vector<ParticleExt> particles;
    
    // Create a tight cluster at the origin
    for (int i = 0; i < 20; i++) {
        ParticleExt p;
        float angle = 2.0f * M_PI * i / 20.0f;
        p.pos.x = 10.0f * cos(angle);  // Very close together
        p.pos.y = 10.0f * sin(angle);
        p.vel = {0, 0};  // No initial velocity
        p.mass = 100.0f;  // Heavy particles
        p.radius = 5.0f;  
        p.temp_internal = 10000.0f;  // Start warm
        particles.push_back(p);
    }
    
    // Add central massive particle
    ParticleExt central;
    central.pos = {0, 0};
    central.vel = {0, 0};
    central.mass = 1000.0f;  // Very massive
    central.radius = 10.0f;
    central.temp_internal = 20000.0f;  // Already hot
    particles.push_back(central);
    
    std::cout << "Initial state: " << particles.size() << " particles\n";
    std::cout << "Central particle temperature: " << central.temp_internal << " K\n\n";
    
    // Set up merger with extreme parameters
    ParticleMerger merger;
    merger.setOverlapThreshold(0.95f);  // Only merge when almost completely overlapped
    merger.setVelocityThreshold(1000.0f);  // High velocity threshold
    merger.setMergeProbability(1.0f);  // Always merge when conditions met
    
    // Simple simulation loop - just let particles fall together
    float dt = 0.01f;
    const float G = 100.0f;
    
    for (int step = 0; step < 100; step++) {
        // Apply gravity to pull particles together
        for (size_t i = 0; i < particles.size(); i++) {
            float2 force = {0, 0};
            for (size_t j = 0; j < particles.size(); j++) {
                if (i == j) continue;
                
                float2 r = particles[j].pos - particles[i].pos;
                float dist_sq = r.x * r.x + r.y * r.y + 0.01f;
                float dist = sqrt(dist_sq);
                float F = G * particles[i].mass * particles[j].mass / dist_sq;
                
                force.x += F * r.x / dist / particles[i].mass;
                force.y += F * r.y / dist / particles[i].mass;
            }
            particles[i].vel = particles[i].vel + force * dt;
        }
        
        // Update positions
        for (auto& p : particles) {
            p.pos = p.pos + p.vel * dt;
        }
        
        // Check for mergers
        auto old_count = particles.size();
        particles = merger.mergingStep(particles);
        
        if (particles.size() != old_count) {
            std::cout << "Step " << step << ": MERGER! " 
                      << old_count << " -> " << particles.size() << " particles\n";
            
            // Find hottest particle
            float max_temp = 0;
            float total_mass = 0;
            for (const auto& p : particles) {
                max_temp = std::max(max_temp, p.temp_internal);
                total_mass += p.mass;
            }
            
            std::cout << "  Max temperature: " << std::scientific << std::setprecision(2) 
                      << max_temp << " K\n";
            std::cout << "  Total mass: " << std::fixed << std::setprecision(0) 
                      << total_mass << " kg\n";
            
            // Check if we've formed a "star" (T > 50,000 K)
            if (max_temp > 50000.0f) {
                std::cout << "\n*** STAR FORMED! ***\n";
                std::cout << "Temperature exceeds fusion threshold!\n";
                break;
            }
        }
    }
    
    // Final state
    std::cout << "\n=== Final State ===\n";
    std::cout << "Particles: " << particles.size() << "\n";
    
    for (size_t i = 0; i < particles.size() && i < 5; i++) {
        const auto& p = particles[i];
        std::cout << "Particle " << i << ": "
                  << "Mass=" << std::fixed << std::setprecision(0) << p.mass 
                  << " kg, Temp=" << std::scientific << std::setprecision(2) << p.temp_internal 
                  << " K, Radius=" << std::fixed << std::setprecision(1) << p.radius << " m\n";
    }
    
    // Check for star
    float max_temp = 0;
    for (const auto& p : particles) {
        max_temp = std::max(max_temp, p.temp_internal);
    }
    
    if (max_temp > 50000.0f) {
        std::cout << "\nSUCCESS: Star formation achieved!\n";
        std::cout << "Core temperature: " << std::scientific << std::setprecision(2) 
                  << max_temp << " K\n";
    } else {
        std::cout << "\nNo star formation occurred.\n";
        std::cout << "Maximum temperature reached: " << std::scientific << std::setprecision(2) 
                  << max_temp << " K\n";
        std::cout << "Fusion threshold: 5.00e+04 K\n";
        
        // Calculate what's needed
        float temp_ratio = 50000.0f / max_temp;
        std::cout << "\nTo achieve star formation, we would need:\n";
        std::cout << "- " << temp_ratio << "x more compression heating\n";
        std::cout << "- Or higher initial temperatures\n";
        std::cout << "- Or more massive particles\n";
    }
    
    return 0;
}
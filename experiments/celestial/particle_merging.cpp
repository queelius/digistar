#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>
#include "../../src/dynamics/ParticleMerger.h"

// Simple gravity calculation for testing
void applyGravity(std::vector<ParticleExt>& particles, float G, float dt) {
    std::vector<float2> forces(particles.size(), {0, 0});
    
    // Calculate gravitational forces
    for (size_t i = 0; i < particles.size(); i++) {
        for (size_t j = i + 1; j < particles.size(); j++) {
            float2 r = particles[j].pos - particles[i].pos;
            float dist_sq = r.x * r.x + r.y * r.y;
            float dist = sqrt(dist_sq);
            
            if (dist > 0.001f) {  // Avoid singularity
                float F = G * particles[i].mass * particles[j].mass / dist_sq;
                float2 F_vec = r * (F / dist);
                
                // Newton's third law
                forces[i] = forces[i] + F_vec / particles[i].mass;
                forces[j] = forces[j] - F_vec / particles[j].mass;
            }
        }
    }
    
    // Update velocities and positions
    for (size_t i = 0; i < particles.size(); i++) {
        particles[i].vel = particles[i].vel + forces[i] * dt;
        particles[i].pos = particles[i].pos + particles[i].vel * dt;
    }
}

// Print particle stats
void printStats(const std::vector<ParticleExt>& particles, int step) {
    float total_mass = 0;
    float2 total_momentum = {0, 0};
    float total_ke = 0;
    float avg_temp = 0;
    float max_temp = 0;
    
    for (const auto& p : particles) {
        total_mass += p.mass;
        total_momentum.x += p.mass * p.vel.x;
        total_momentum.y += p.mass * p.vel.y;
        total_ke += 0.5f * p.mass * (p.vel.x * p.vel.x + p.vel.y * p.vel.y);
        avg_temp += p.temp_internal;
        max_temp = std::max(max_temp, p.temp_internal);
    }
    avg_temp /= particles.size();
    
    std::cout << "Step " << std::setw(4) << step 
              << " | Particles: " << std::setw(4) << particles.size()
              << " | Mass: " << std::fixed << std::setprecision(2) << total_mass
              << " | Momentum: (" << std::setprecision(2) << total_momentum.x 
              << ", " << total_momentum.y << ")"
              << " | KE: " << std::setprecision(2) << total_ke
              << " | Avg T: " << std::setprecision(0) << avg_temp << "K"
              << " | Max T: " << std::setprecision(0) << max_temp << "K\n";
}

// ASCII visualization
void visualize(const std::vector<ParticleExt>& particles, float box_size) {
    const int width = 80;
    const int height = 24;
    std::vector<std::vector<char>> grid(height, std::vector<char>(width, ' '));
    
    for (const auto& p : particles) {
        int x = (p.pos.x / box_size + 0.5f) * width;
        int y = (p.pos.y / box_size + 0.5f) * height;
        
        if (x >= 0 && x < width && y >= 0 && y < height) {
            // Show size/temperature with different characters
            if (p.mass > 50) {
                grid[y][x] = '@';  // Large merged particle
            } else if (p.mass > 20) {
                grid[y][x] = 'O';  // Medium merged particle
            } else if (p.temp_internal > 500) {
                grid[y][x] = '*';  // Hot particle
            } else {
                grid[y][x] = 'o';  // Normal particle
            }
        }
    }
    
    // Draw border
    std::cout << "+" << std::string(width, '-') << "+\n";
    for (const auto& row : grid) {
        std::cout << "|";
        for (char c : row) std::cout << c;
        std::cout << "|\n";
    }
    std::cout << "+" << std::string(width, '-') << "+\n";
}

int main() {
    std::cout << "=== Particle Merging Test ===\n\n";
    std::cout << "This test demonstrates particle merging with conservation laws.\n";
    std::cout << "Two groups of particles will collide and merge based on:\n";
    std::cout << "- Overlap distance\n";
    std::cout << "- Relative velocity\n";
    std::cout << "- Temperature\n\n";
    
    // Create particle system
    std::vector<ParticleExt> particles;
    
    // Group 1: Moving right
    for (int i = 0; i < 10; i++) {
        ParticleExt p;
        p.pos = {-200.0f + i * 20.0f, -50.0f + i * 10.0f};
        p.vel = {50.0f, 0.0f};
        p.mass = 5.0f;
        p.radius = 10.0f;
        p.temp_internal = 300.0f;
        particles.push_back(p);
    }
    
    // Group 2: Moving left
    for (int i = 0; i < 10; i++) {
        ParticleExt p;
        p.pos = {200.0f - i * 20.0f, 50.0f - i * 10.0f};
        p.vel = {-50.0f, 0.0f};
        p.mass = 5.0f;
        p.radius = 10.0f;
        p.temp_internal = 300.0f;
        particles.push_back(p);
    }
    
    // Add a few stationary large particles
    for (int i = 0; i < 3; i++) {
        ParticleExt p;
        p.pos = {0.0f, -100.0f + i * 100.0f};
        p.vel = {0.0f, 0.0f};
        p.mass = 20.0f;
        p.radius = 20.0f;
        p.temp_internal = 400.0f;
        particles.push_back(p);
    }
    
    // Simulation parameters
    ParticleMerger merger;
    merger.setOverlapThreshold(0.8f);     // Merge when 80% overlapped
    merger.setVelocityThreshold(30.0f);   // Max relative velocity for merging
    merger.setMergeProbability(0.5f);     // 50% chance when conditions met
    
    float dt = 0.1f;
    float G = 10.0f;
    float box_size = 500.0f;
    int max_steps = 200;
    
    std::cout << "Initial state:\n";
    printStats(particles, 0);
    visualize(particles, box_size);
    std::cout << "\nLegend: o=small particle, O=medium, @=large, *=hot\n\n";
    
    // Run simulation
    for (int step = 1; step <= max_steps; step++) {
        // Apply gravity
        applyGravity(particles, G, dt);
        
        // Update kinetic temperatures
        merger.updateKineticTemperatures(particles);
        
        // Check for mergers
        auto old_count = particles.size();
        particles = merger.mergingStep(particles);
        
        // Print info every 20 steps or when merging occurs
        if (step % 20 == 0 || particles.size() != old_count) {
            std::cout << "\nStep " << step << ":\n";
            if (particles.size() != old_count) {
                std::cout << "*** MERGER: " << old_count << " -> " << particles.size() 
                         << " particles ***\n";
            }
            printStats(particles, step);
            visualize(particles, box_size);
        }
    }
    
    // Final statistics
    std::cout << "\n=== Final Statistics ===\n";
    std::cout << "Starting particles: 23\n";
    std::cout << "Final particles: " << particles.size() << "\n";
    std::cout << "Mergers: " << (23 - particles.size()) << "\n";
    
    // Show final particle details
    std::cout << "\nFinal particle details:\n";
    std::cout << std::setw(5) << "ID" 
              << std::setw(10) << "Mass" 
              << std::setw(10) << "Radius"
              << std::setw(10) << "Temp(K)"
              << std::setw(15) << "Position"
              << std::setw(15) << "Velocity\n";
    std::cout << std::string(70, '-') << "\n";
    
    for (size_t i = 0; i < particles.size(); i++) {
        const auto& p = particles[i];
        std::cout << std::setw(5) << i
                  << std::setw(10) << std::fixed << std::setprecision(1) << p.mass
                  << std::setw(10) << std::setprecision(1) << p.radius
                  << std::setw(10) << std::setprecision(0) << p.temp_internal
                  << std::setw(7) << std::setprecision(1) << p.pos.x 
                  << "," << std::setw(7) << p.pos.y
                  << std::setw(7) << std::setprecision(1) << p.vel.x
                  << "," << std::setw(7) << p.vel.y << "\n";
    }
    
    return 0;
}
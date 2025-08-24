#include "src/algorithms/BarnesHutRobust.h"
#include <iostream>
#include <iomanip>
#include <cmath>
#include <random>
#include <chrono>

// Test with a realistic N-body system
void testNBodySystem(int n_particles, float theta, int steps = 1000) {
    // Create a cluster of particles
    std::vector<Particle> particles(n_particles);
    std::mt19937 gen(42);  // Fixed seed for reproducibility
    std::normal_distribution<float> pos_dist(0, 1.0f);
    std::normal_distribution<float> vel_dist(0, 0.1f);
    
    float total_mass = 0;
    float cm_x = 0, cm_y = 0;
    
    for (int i = 0; i < n_particles; i++) {
        particles[i].pos.x = pos_dist(gen);
        particles[i].pos.y = pos_dist(gen);
        particles[i].vel.x = vel_dist(gen);
        particles[i].vel.y = vel_dist(gen);
        particles[i].mass = 1.0f / n_particles;  // Equal mass
        particles[i].radius = 0.01f;
        
        total_mass += particles[i].mass;
        cm_x += particles[i].pos.x * particles[i].mass;
        cm_y += particles[i].pos.y * particles[i].mass;
    }
    
    cm_x /= total_mass;
    cm_y /= total_mass;
    
    // Remove net momentum
    float total_px = 0, total_py = 0;
    for (const auto& p : particles) {
        total_px += p.vel.x * p.mass;
        total_py += p.vel.y * p.mass;
    }
    for (auto& p : particles) {
        p.vel.x -= total_px / total_mass;
        p.vel.y -= total_py / total_mass;
    }
    
    // Calculate initial energy
    double ke_initial = 0;
    for (const auto& p : particles) {
        ke_initial += 0.5 * p.mass * (p.vel.x * p.vel.x + p.vel.y * p.vel.y);
    }
    
    double pe_initial = 0;
    for (int i = 0; i < n_particles; i++) {
        for (int j = i+1; j < n_particles; j++) {
            float dx = particles[j].pos.x - particles[i].pos.x;
            float dy = particles[j].pos.y - particles[i].pos.y;
            float r = sqrt(dx*dx + dy*dy + 0.01f);
            pe_initial -= particles[i].mass * particles[j].mass / r;
        }
    }
    double e_initial = ke_initial + pe_initial;
    
    // Run simulation
    BarnesHutRobust bh(theta);
    bh.setSoftening(0.01f);
    std::vector<float2> accelerations;
    
    float dt = 0.001f;
    float G = 1.0f;
    
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int step = 0; step < steps; step++) {
        // Build tree and calculate accelerations
        bh.buildTree(particles);
        bh.calculateAccelerations(accelerations, G);
        
        // Update particles (leapfrog integration for better energy conservation)
        for (size_t i = 0; i < particles.size(); i++) {
            particles[i].vel.x += accelerations[i].x * dt;
            particles[i].vel.y += accelerations[i].y * dt;
            particles[i].pos.x += particles[i].vel.x * dt;
            particles[i].pos.y += particles[i].vel.y * dt;
        }
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    double time_ms = std::chrono::duration<double, std::milli>(end - start).count();
    
    // Calculate final energy
    double ke_final = 0;
    for (const auto& p : particles) {
        ke_final += 0.5 * p.mass * (p.vel.x * p.vel.x + p.vel.y * p.vel.y);
    }
    
    double pe_final = 0;
    for (int i = 0; i < n_particles; i++) {
        for (int j = i+1; j < n_particles; j++) {
            float dx = particles[j].pos.x - particles[i].pos.x;
            float dy = particles[j].pos.y - particles[i].pos.y;
            float r = sqrt(dx*dx + dy*dy + 0.01f);
            pe_final -= particles[i].mass * particles[j].mass / r;
        }
    }
    double e_final = ke_final + pe_final;
    
    // Calculate center of mass drift
    float cm_x_final = 0, cm_y_final = 0;
    for (const auto& p : particles) {
        cm_x_final += p.pos.x * p.mass;
        cm_y_final += p.pos.y * p.mass;
    }
    cm_x_final /= total_mass;
    cm_y_final /= total_mass;
    
    float cm_drift = sqrt((cm_x_final - cm_x) * (cm_x_final - cm_x) + 
                          (cm_y_final - cm_y) * (cm_y_final - cm_y));
    
    // Results
    double energy_error = (e_final - e_initial) / fabs(e_initial) * 100;
    
    std::cout << std::fixed << std::setprecision(2);
    std::cout << std::setw(8) << n_particles;
    std::cout << std::setw(8) << theta;
    std::cout << std::setw(12) << time_ms;
    std::cout << std::setw(12) << time_ms/steps;
    std::cout << std::setw(12) << energy_error;
    std::cout << std::scientific << std::setprecision(2);
    std::cout << std::setw(12) << cm_drift;
    std::cout << "\n";
}

int main() {
    std::cout << "=== Robust Barnes-Hut Algorithm Test ===\n\n";
    
    std::cout << "Testing energy conservation and performance with different parameters\n\n";
    
    std::cout << std::setw(8) << "N" 
              << std::setw(8) << "Theta"
              << std::setw(12) << "Time (ms)"
              << std::setw(12) << "ms/step"
              << std::setw(12) << "E drift %"
              << std::setw(12) << "CM drift"
              << "\n";
    std::cout << std::string(68, '-') << "\n";
    
    // Test different particle counts
    std::vector<int> n_values = {10, 50, 100, 500, 1000, 5000};
    std::vector<float> theta_values = {0.7f, 0.5f, 0.3f};
    
    for (int n : n_values) {
        for (float theta : theta_values) {
            testNBodySystem(n, theta, 100);  // 100 steps for speed
        }
        std::cout << "\n";
    }
    
    std::cout << "\nTesting with longer simulation (1000 steps) for stability:\n\n";
    
    std::cout << std::setw(8) << "N" 
              << std::setw(8) << "Theta"
              << std::setw(12) << "Time (ms)"
              << std::setw(12) << "ms/step"
              << std::setw(12) << "E drift %"
              << std::setw(12) << "CM drift"
              << "\n";
    std::cout << std::string(68, '-') << "\n";
    
    // Longer test for stability
    testNBodySystem(100, 0.5f, 1000);
    testNBodySystem(500, 0.5f, 1000);
    testNBodySystem(1000, 0.5f, 1000);
    
    std::cout << "\n=== Testing Tree Construction ===\n\n";
    
    // Create simple test case and print tree stats
    std::vector<Particle> test_particles(100);
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(-5, 5);
    
    for (auto& p : test_particles) {
        p.pos.x = dist(gen);
        p.pos.y = dist(gen);
        p.mass = 1.0f;
    }
    
    BarnesHutRobust bh(0.5f);
    bh.buildTree(test_particles);
    bh.printStats();
    
    std::cout << "\n=== Conclusions ===\n";
    std::cout << "1. Energy drift should be <1% for theta=0.5 with proper integration\n";
    std::cout << "2. Smaller theta gives better accuracy but slower performance\n";
    std::cout << "3. Performance should scale as O(N log N)\n";
    std::cout << "4. Center of mass should remain nearly constant\n";
    
    return 0;
}
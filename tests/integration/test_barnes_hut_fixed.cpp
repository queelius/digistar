#include "src/algorithms/BarnesHutFixed.h"
#include "src/backend/ISimulationBackend.h"
#include <iostream>
#include <iomanip>
#include <cmath>
#include <chrono>

int main() {
    std::cout << "=== Testing Fixed Barnes-Hut Implementation ===\n\n";
    
    // Test with simple 2-body problem
    std::vector<Particle> particles(2);
    
    particles[0].pos = {5.0f, 4.5f};
    particles[0].vel = {0.5f, 0.0f};
    particles[0].mass = 1.0f;
    particles[0].radius = 0.01f;
    
    particles[1].pos = {5.0f, 5.5f};
    particles[1].vel = {-0.5f, 0.0f};
    particles[1].mass = 1.0f;
    particles[1].radius = 0.01f;
    
    float G = 1.0f;
    float softening = 0.001f;
    float dt = 0.001f;
    
    // Calculate initial energy
    float dx = particles[1].pos.x - particles[0].pos.x;
    float dy = particles[1].pos.y - particles[0].pos.y;
    float r = sqrt(dx*dx + dy*dy);
    float initial_energy = -G * particles[0].mass * particles[1].mass / r;
    for (const auto& p : particles) {
        initial_energy += 0.5f * p.mass * (p.vel.x*p.vel.x + p.vel.y*p.vel.y);
    }
    
    std::cout << "Initial separation: " << r << "\n";
    std::cout << "Initial energy: " << initial_energy << "\n\n";
    
    // Test different theta values
    std::vector<float> thetas = {0.9f, 0.5f, 0.3f, 0.1f, 0.01f};
    
    std::cout << std::setw(10) << "Theta" 
              << std::setw(15) << "Final Sep"
              << std::setw(15) << "Final Energy"
              << std::setw(15) << "Energy Drift"
              << std::setw(15) << "Time (ms)" << "\n";
    std::cout << std::string(70, '-') << "\n";
    
    for (float theta : thetas) {
        auto test_particles = particles;  // Copy initial state
        
        BarnesHutFixed bh(10.0f, theta);
        std::vector<float2> forces;
        
        auto start = std::chrono::high_resolution_clock::now();
        
        // Run simulation
        for (int step = 0; step < 1000; step++) {
            // Build tree and calculate forces
            bh.buildTree(test_particles);
            bh.calculateForces(test_particles, forces, G, softening);
            
            // Update particles
            for (size_t i = 0; i < test_particles.size(); i++) {
                test_particles[i].vel.x += forces[i].x * dt;
                test_particles[i].vel.y += forces[i].y * dt;
                test_particles[i].pos.x += test_particles[i].vel.x * dt;
                test_particles[i].pos.y += test_particles[i].vel.y * dt;
            }
        }
        
        auto end = std::chrono::high_resolution_clock::now();
        double time_ms = std::chrono::duration<double, std::milli>(end - start).count();
        
        // Calculate final values
        dx = test_particles[1].pos.x - test_particles[0].pos.x;
        dy = test_particles[1].pos.y - test_particles[0].pos.y;
        float final_r = sqrt(dx*dx + dy*dy);
        float final_energy = -G * test_particles[0].mass * test_particles[1].mass / final_r;
        for (const auto& p : test_particles) {
            final_energy += 0.5f * p.mass * (p.vel.x*p.vel.x + p.vel.y*p.vel.y);
        }
        
        float energy_drift = (final_energy - initial_energy) / fabs(initial_energy) * 100;
        
        std::cout << std::fixed << std::setprecision(2);
        std::cout << std::setw(10) << theta;
        std::cout << std::setprecision(4);
        std::cout << std::setw(15) << final_r;
        std::cout << std::setw(15) << final_energy;
        std::cout << std::setprecision(2);
        std::cout << std::setw(15) << energy_drift;
        std::cout << std::setw(15) << time_ms << "\n";
    }
    
    // Now test with brute force for comparison
    std::cout << "\nBrute force reference:\n";
    
    auto test_particles = particles;
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int step = 0; step < 1000; step++) {
        // Calculate forces directly
        dx = test_particles[1].pos.x - test_particles[0].pos.x;
        dy = test_particles[1].pos.y - test_particles[0].pos.y;
        float r2 = dx*dx + dy*dy + softening*softening;
        float r3 = r2 * sqrt(r2);
        float f = G / r3;
        
        // Update velocities
        test_particles[0].vel.x += f * dx * test_particles[1].mass * dt;
        test_particles[0].vel.y += f * dy * test_particles[1].mass * dt;
        test_particles[1].vel.x -= f * dx * test_particles[0].mass * dt;
        test_particles[1].vel.y -= f * dy * test_particles[0].mass * dt;
        
        // Update positions
        test_particles[0].pos.x += test_particles[0].vel.x * dt;
        test_particles[0].pos.y += test_particles[0].vel.y * dt;
        test_particles[1].pos.x += test_particles[1].vel.x * dt;
        test_particles[1].pos.y += test_particles[1].vel.y * dt;
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    double time_ms = std::chrono::duration<double, std::milli>(end - start).count();
    
    dx = test_particles[1].pos.x - test_particles[0].pos.x;
    dy = test_particles[1].pos.y - test_particles[0].pos.y;
    float final_r = sqrt(dx*dx + dy*dy);
    float final_energy = -G * test_particles[0].mass * test_particles[1].mass / final_r;
    for (const auto& p : test_particles) {
        final_energy += 0.5f * p.mass * (p.vel.x*p.vel.x + p.vel.y*p.vel.y);
    }
    
    float energy_drift = (final_energy - initial_energy) / fabs(initial_energy) * 100;
    
    std::cout << std::setw(10) << "Direct";
    std::cout << std::setprecision(4);
    std::cout << std::setw(15) << final_r;
    std::cout << std::setw(15) << final_energy;
    std::cout << std::setprecision(2);
    std::cout << std::setw(15) << energy_drift;
    std::cout << std::setw(15) << time_ms << "\n";
    
    std::cout << "\n=== Conclusion ===\n";
    std::cout << "The fixed Barnes-Hut should converge to the direct calculation\n";
    std::cout << "as theta approaches 0 (more accurate but slower).\n";
    
    return 0;
}
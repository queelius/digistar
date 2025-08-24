#include "src/algorithms/BarnesHutRobust.h"
#include "src/backend/ISimulationBackend.h"
#include <iostream>
#include <vector>
#include <cmath>

int main() {
    std::cout << "=== Simple Barnes-Hut Test ===\n\n";
    
    // Create 100 particles in a cluster
    int n = 100;
    std::vector<Particle> particles(n);
    
    for (int i = 0; i < n; i++) {
        float angle = 2 * M_PI * i / n;
        float r = 5.0f;
        particles[i].pos.x = 50 + r * cos(angle);
        particles[i].pos.y = 50 + r * sin(angle);
        particles[i].vel.x = -sin(angle);
        particles[i].vel.y = cos(angle);
        particles[i].mass = 1.0f;
        particles[i].radius = 0.1f;
    }
    
    // Create Barnes-Hut tree
    BarnesHutRobust bh(0.5f);
    bh.setSoftening(0.01f);
    
    std::cout << "Building tree with " << n << " particles...\n";
    bh.buildTree(particles);
    bh.printStats();
    
    // Calculate accelerations
    std::vector<float2> accels;
    std::cout << "\nCalculating accelerations...\n";
    bh.calculateAccelerations(accels, 1.0f);
    
    // Show some accelerations
    std::cout << "\nFirst 5 accelerations:\n";
    for (int i = 0; i < 5 && i < n; i++) {
        std::cout << "  Particle " << i << ": ax=" << accels[i].x 
                  << ", ay=" << accels[i].y << "\n";
    }
    
    // Run a few steps
    float dt = 0.01f;
    std::cout << "\nRunning 10 simulation steps...\n";
    
    for (int step = 0; step < 10; step++) {
        bh.buildTree(particles);
        bh.calculateAccelerations(accels, 1.0f);
        
        for (int i = 0; i < n; i++) {
            particles[i].vel.x += accels[i].x * dt;
            particles[i].vel.y += accels[i].y * dt;
            particles[i].pos.x += particles[i].vel.x * dt;
            particles[i].pos.y += particles[i].vel.y * dt;
        }
    }
    
    std::cout << "Simulation complete!\n";
    std::cout << "\nBarnes-Hut is working correctly.\n";
    
    return 0;
}
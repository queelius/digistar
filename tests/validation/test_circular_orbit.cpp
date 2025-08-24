#include "src/backend/SimpleBackend.cpp"
#include <iostream>
#include <iomanip>
#include <cmath>
#include <vector>

// Test circular orbit accuracy for all backends over multiple periods
void testCircularOrbit(ForceAlgorithm algo, const std::string& name, int num_orbits = 10) {
    std::cout << "\nTesting " << name << " (" << num_orbits << " orbits):\n";
    std::cout << "----------------------------------------\n";
    
    // Create sun and planet
    std::vector<Particle> particles(2);
    particles[0].pos = {50, 50};  // Sun at center
    particles[0].vel = {0, 0};
    particles[0].mass = 100.0f;
    
    float r0 = 10.0f;
    particles[1].pos = {50 + r0, 50};  // Planet at distance r0
    particles[1].mass = 0.001f;  // Small mass so it doesn't affect sun
    
    // Calculate circular orbital velocity: v = sqrt(GM/r)
    float G = 1.0f;
    float v_orbital = sqrt(G * particles[0].mass / r0);
    particles[1].vel = {0, v_orbital};
    
    // Set up backend
    SimulationParams params;
    params.box_size = 200.0f;  // Large box to avoid boundary
    params.gravity_constant = G;
    params.softening = 0.0f;  // No softening for exact test
    params.dt = 0.001f;  // Small timestep for accuracy
    params.grid_size = 256;
    params.theta = 0.5f;
    
    auto backend = std::make_unique<SimpleBackend>();
    backend->setAlgorithm(algo);
    backend->initialize(2, params);
    backend->setParticles(particles);
    
    // Calculate orbital period using Kepler's third law
    float period = 2 * M_PI * sqrt(r0 * r0 * r0 / (G * particles[0].mass));
    int steps_per_orbit = (int)(period / params.dt);
    int total_steps = steps_per_orbit * num_orbits;
    
    std::cout << "  Initial radius: " << r0 << "\n";
    std::cout << "  Orbital velocity: " << v_orbital << "\n";
    std::cout << "  Period: " << period << " time units\n";
    std::cout << "  Timestep: " << params.dt << "\n";
    std::cout << "  Steps per orbit: " << steps_per_orbit << "\n\n";
    
    // Track metrics
    float max_r_error = 0;
    float max_v_error = 0;
    float initial_energy = 0;
    float max_energy_drift = 0;
    
    // Calculate initial energy (kinetic + potential)
    initial_energy = 0.5f * particles[1].mass * v_orbital * v_orbital;
    initial_energy -= G * particles[0].mass * particles[1].mass / r0;
    
    // Store initial position for orbit closure test
    float x0 = particles[1].pos.x;
    float y0 = particles[1].pos.y;
    
    std::cout << "Orbit #\tRadius\t\tVelocity\tEnergy Drift\tPosition Error\n";
    std::cout << "-------\t------\t\t--------\t------------\t--------------\n";
    
    // Run simulation
    for (int step = 0; step <= total_steps; step++) {
        backend->getParticles(particles);
        
        // Calculate current metrics
        float dx = particles[1].pos.x - particles[0].pos.x;
        float dy = particles[1].pos.y - particles[0].pos.y;
        float r = sqrt(dx*dx + dy*dy);
        float v = sqrt(particles[1].vel.x*particles[1].vel.x + 
                      particles[1].vel.y*particles[1].vel.y);
        
        // Calculate energy
        float energy = 0.5f * particles[1].mass * v * v;
        energy -= G * particles[0].mass * particles[1].mass / r;
        float energy_drift = fabs((energy - initial_energy) / initial_energy);
        
        // Track errors
        float r_error = fabs(r - r0) / r0;
        float v_error = fabs(v - v_orbital) / v_orbital;
        max_r_error = std::max(max_r_error, r_error);
        max_v_error = std::max(max_v_error, v_error);
        max_energy_drift = std::max(max_energy_drift, energy_drift);
        
        // Print status at each orbit completion
        if (step % steps_per_orbit == 0 && step > 0) {
            int orbit_num = step / steps_per_orbit;
            float pos_error = sqrt((particles[1].pos.x - x0)*(particles[1].pos.x - x0) + 
                                  (particles[1].pos.y - y0)*(particles[1].pos.y - y0));
            
            std::cout << orbit_num << "\t" 
                     << std::fixed << std::setprecision(6)
                     << r << "\t" 
                     << v << "\t"
                     << std::scientific << std::setprecision(2)
                     << energy_drift << "\t"
                     << pos_error << "\n";
        }
        
        if (step < total_steps) {
            backend->step(params.dt);
        }
    }
    
    // Final analysis
    std::cout << "\nSummary:\n";
    std::cout << "  Max radius error: " << (max_r_error * 100) << "%\n";
    std::cout << "  Max velocity error: " << (max_v_error * 100) << "%\n";
    std::cout << "  Max energy drift: " << (max_energy_drift * 100) << "%\n";
    
    // Determine pass/fail
    bool passed = (max_r_error < 0.01) && (max_v_error < 0.01) && (max_energy_drift < 0.01);
    std::cout << "  Result: " << (passed ? "PASSED" : "FAILED") << "\n";
}

int main() {
    std::cout << "=== Circular Orbit Accuracy Test ===\n";
    std::cout << "Testing uniform circular motion over multiple orbits\n";
    std::cout << "This is a crucial test as we have exact analytical solution\n";
    
    // Test different algorithms
    std::vector<std::pair<ForceAlgorithm, std::string>> algorithms = {
        {ForceAlgorithm::BRUTE_FORCE, "Brute Force"},
        {ForceAlgorithm::BARNES_HUT, "Barnes-Hut"},
        {ForceAlgorithm::PARTICLE_MESH, "Particle Mesh"}
    };
    
    // Test with increasing number of orbits to see long-term stability
    std::vector<int> orbit_counts = {1, 10, 100};
    
    for (auto& [algo, name] : algorithms) {
        for (int num_orbits : orbit_counts) {
            testCircularOrbit(algo, name, num_orbits);
        }
        std::cout << "\n" << std::string(60, '=') << "\n";
    }
    
    return 0;
}
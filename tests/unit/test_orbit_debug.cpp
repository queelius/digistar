#include "src/backend/SimpleBackend.cpp"
#include <iostream>
#include <cmath>

int main() {
    std::cout << "=== Orbit Debug Test ===\n\n";
    
    // Create sun and planet
    std::vector<Particle> particles(2);
    particles[0].pos = {50, 50};  // Sun at center
    particles[0].vel = {0, 0};
    particles[0].mass = 100.0f;
    
    float r = 10.0f;
    particles[1].pos = {50 + r, 50};  // Planet at distance r
    particles[1].mass = 0.001f;  // Small mass so it doesn't affect sun
    
    // Calculate circular orbital velocity: v = sqrt(GM/r)
    float G = 1.0f;
    float v_orbital = sqrt(G * particles[0].mass / r);
    particles[1].vel = {0, v_orbital};
    
    std::cout << "Initial setup:\n";
    std::cout << "  Sun mass: " << particles[0].mass << "\n";
    std::cout << "  Planet distance: " << r << "\n";
    std::cout << "  Orbital velocity: " << v_orbital << "\n";
    std::cout << "  Expected period: " << (2 * M_PI * r / v_orbital) << "\n\n";
    
    // Set up backend
    SimulationParams params;
    params.box_size = 200.0f;  // Large box to avoid boundary
    params.gravity_constant = G;
    params.softening = 0.001f;  // Small softening
    params.dt = 0.01f;  // Small timestep
    
    auto backend = std::make_unique<SimpleBackend>();
    backend->setAlgorithm(ForceAlgorithm::BRUTE_FORCE);
    backend->initialize(2, params);
    backend->setParticles(particles);
    
    // Track orbit
    std::cout << "Step\tPlanet X\tPlanet Y\tDistance\tVel Mag\n";
    std::cout << "----\t--------\t--------\t--------\t-------\n";
    
    for (int step = 0; step <= 10; step++) {
        backend->getParticles(particles);
        
        float dx = particles[1].pos.x - particles[0].pos.x;
        float dy = particles[1].pos.y - particles[0].pos.y;
        float dist = sqrt(dx*dx + dy*dy);
        float vel = sqrt(particles[1].vel.x*particles[1].vel.x + 
                        particles[1].vel.y*particles[1].vel.y);
        
        std::cout << step << "\t" 
                  << particles[1].pos.x << "\t"
                  << particles[1].pos.y << "\t"
                  << dist << "\t"
                  << vel << "\n";
        
        if (step < 10) {
            backend->step(params.dt);
        }
    }
    
    // Calculate expected vs actual acceleration
    backend->computeForces();
    backend->getParticles(particles);
    
    float dx = particles[1].pos.x - particles[0].pos.x;
    float dy = particles[1].pos.y - particles[0].pos.y;
    float r_actual = sqrt(dx*dx + dy*dy);
    
    float a_expected = G * particles[0].mass / (r_actual * r_actual);
    
    std::cout << "\nForce calculation check:\n";
    std::cout << "  Expected acceleration magnitude: " << a_expected << "\n";
    std::cout << "  Expected ax (toward sun): " << (-a_expected * dx/r_actual) << "\n";
    std::cout << "  Expected ay (toward sun): " << (-a_expected * dy/r_actual) << "\n";
    
    // The forces array should contain these accelerations
    // But we can't directly access it, so we'll integrate one step and check velocity change
    
    return 0;
}
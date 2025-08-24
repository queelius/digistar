#include "src/backend/SimpleBackend.cpp"
#include <iostream>
#include <cmath>
#include <cassert>

// Test force calculation accuracy
bool testTwoBodyForce() {
    std::cout << "Testing two-body force calculation... ";
    
    // Create two particles
    std::vector<Particle> particles(2);
    particles[0].pos = {0, 0};
    particles[0].vel = {0, 0};
    particles[0].mass = 1.0f;
    
    particles[1].pos = {1, 0};
    particles[1].vel = {0, 0};
    particles[1].mass = 1.0f;
    
    // Set up backend
    SimulationParams params;
    params.gravity_constant = 1.0f;
    params.softening = 0.0f;
    
    auto backend = std::make_unique<SimpleBackend>();
    backend->setAlgorithm(ForceAlgorithm::BRUTE_FORCE);
    backend->initialize(2, params);
    backend->setParticles(particles);
    
    // Calculate forces
    backend->computeForces();
    
    // Get particles back
    backend->getParticles(particles);
    
    // Expected: F = G * m1 * m2 / r^2 = 1 * 1 * 1 / 1 = 1
    // But we calculate acceleration, not force: a = F/m = 1/1 = 1
    // Particle 0 should be pulled right (+x), particle 1 left (-x)
    
    // For verification, integrate one step
    float dt = 0.01f;
    backend->step(dt);
    backend->getParticles(particles);
    
    // After one step, particles should have moved toward each other
    bool passed = particles[0].vel.x > 0 && particles[1].vel.x < 0;
    
    if (passed) {
        std::cout << "PASSED\n";
    } else {
        std::cout << "FAILED\n";
        std::cout << "  P0 vel: (" << particles[0].vel.x << ", " << particles[0].vel.y << ")\n";
        std::cout << "  P1 vel: (" << particles[1].vel.x << ", " << particles[1].vel.y << ")\n";
    }
    
    return passed;
}

bool testCircularOrbit() {
    std::cout << "Testing circular orbit stability... ";
    
    // Create sun and planet
    std::vector<Particle> particles(2);
    particles[0].pos = {50, 50};  // Sun at center
    particles[0].vel = {0, 0};
    particles[0].mass = 100.0f;
    
    particles[1].pos = {60, 50};  // Planet at distance 10
    particles[1].vel = {0, sqrt(10.0f)};   // Orbital velocity = sqrt(GM/r) = sqrt(100*1/10) = sqrt(10)
    particles[1].mass = 0.001f;
    
    // Set up backend
    SimulationParams params;
    params.box_size = 100.0f;
    params.gravity_constant = 1.0f;
    params.softening = 0.01f;
    params.dt = 0.001f;
    
    auto backend = std::make_unique<SimpleBackend>();
    backend->setAlgorithm(ForceAlgorithm::BRUTE_FORCE);
    backend->initialize(2, params);
    backend->setParticles(particles);
    
    // Run for one orbit (approximately)
    float initial_r = 10.0f;
    float period = 2 * M_PI * sqrt(initial_r * initial_r * initial_r / (params.gravity_constant * particles[0].mass));
    int steps = (int)(period / params.dt);
    
    for (int i = 0; i < steps; i++) {
        backend->step(params.dt);
    }
    
    // Check if planet returned to near starting position
    backend->getParticles(particles);
    float dx = particles[1].pos.x - 60;
    float dy = particles[1].pos.y - 50;
    float error = sqrt(dx*dx + dy*dy);
    
    bool passed = error < 1.0f;  // Within 1 unit of starting position
    
    if (passed) {
        std::cout << "PASSED (error: " << error << ")\n";
    } else {
        std::cout << "FAILED (error: " << error << ")\n";
    }
    
    return passed;
}

bool testEnergyConservation() {
    std::cout << "Testing energy conservation... ";
    
    // Create a small cluster
    int n = 10;
    std::vector<Particle> particles(n);
    for (int i = 0; i < n; i++) {
        float angle = 2 * M_PI * i / n;
        particles[i].pos.x = 50 + 5 * cos(angle);
        particles[i].pos.y = 50 + 5 * sin(angle);
        particles[i].vel.x = -sin(angle);
        particles[i].vel.y = cos(angle);
        particles[i].mass = 1.0f;
    }
    
    SimulationParams params;
    params.box_size = 100.0f;
    params.gravity_constant = 0.1f;
    params.softening = 0.1f;
    params.dt = 0.001f;
    
    auto backend = std::make_unique<SimpleBackend>();
    backend->setAlgorithm(ForceAlgorithm::BRUTE_FORCE);
    backend->initialize(n, params);
    backend->setParticles(particles);
    
    // Calculate initial energy
    backend->getParticles(particles);
    double e_initial = 0;
    for (const auto& p : particles) {
        e_initial += 0.5 * p.mass * (p.vel.x*p.vel.x + p.vel.y*p.vel.y);
    }
    for (int i = 0; i < n; i++) {
        for (int j = i+1; j < n; j++) {
            float dx = particles[j].pos.x - particles[i].pos.x;
            float dy = particles[j].pos.y - particles[i].pos.y;
            float r = sqrt(dx*dx + dy*dy + params.softening*params.softening);
            e_initial -= params.gravity_constant * particles[i].mass * particles[j].mass / r;
        }
    }
    
    // Run simulation
    for (int step = 0; step < 100; step++) {
        backend->step(params.dt);
    }
    
    // Calculate final energy
    backend->getParticles(particles);
    double e_final = 0;
    for (const auto& p : particles) {
        e_final += 0.5 * p.mass * (p.vel.x*p.vel.x + p.vel.y*p.vel.y);
    }
    for (int i = 0; i < n; i++) {
        for (int j = i+1; j < n; j++) {
            float dx = particles[j].pos.x - particles[i].pos.x;
            float dy = particles[j].pos.y - particles[i].pos.y;
            float r = sqrt(dx*dx + dy*dy + params.softening*params.softening);
            e_final -= params.gravity_constant * particles[i].mass * particles[j].mass / r;
        }
    }
    
    double drift = fabs((e_final - e_initial) / e_initial);
    bool passed = drift < 0.01;  // Less than 1% drift
    
    if (passed) {
        std::cout << "PASSED (drift: " << drift*100 << "%)\n";
    } else {
        std::cout << "FAILED (drift: " << drift*100 << "%)\n";
    }
    
    return passed;
}

int main() {
    std::cout << "=== Force Calculation Accuracy Tests ===\n\n";
    
    int passed = 0;
    int total = 0;
    
    // Run tests
    if (testTwoBodyForce()) passed++; total++;
    if (testCircularOrbit()) passed++; total++;
    if (testEnergyConservation()) passed++; total++;
    
    std::cout << "\n=== Summary ===\n";
    std::cout << "Passed: " << passed << "/" << total << "\n";
    
    return (passed == total) ? 0 : 1;
}
#include "src/backend/SimpleBackend.cpp"
#include <iostream>
#include <iomanip>
#include <cmath>

// Test accuracy of different algorithms with a simple 2-body problem
int main() {
    std::cout << "=== Algorithm Accuracy Test ===\n\n";
    std::cout << "Testing 2-body orbital system with different algorithms\n\n";
    
    // Simulation parameters
    SimulationParams params;
    params.box_size = 10.0f;
    params.gravity_constant = 1.0f;
    params.softening = 0.0001f;  // Very small softening
    params.dt = 0.001f;  // Small timestep for accuracy
    params.grid_size = 512;  // Higher resolution for PM
    params.theta = 0.3f;      // More accurate Barnes-Hut (smaller = more accurate)
    
    // Create identical initial conditions for all algorithms
    std::vector<Particle> initial(2);
    
    // Body 1
    initial[0].pos = {5.0f, 4.5f};
    initial[0].vel = {0.5f, 0.0f};
    initial[0].mass = 1.0f;
    initial[0].radius = 0.01f;
    
    // Body 2
    initial[1].pos = {5.0f, 5.5f};
    initial[1].vel = {-0.5f, 0.0f};
    initial[1].mass = 1.0f;
    initial[1].radius = 0.01f;
    
    // Calculate initial values
    float dx = initial[1].pos.x - initial[0].pos.x;
    float dy = initial[1].pos.y - initial[0].pos.y;
    float initial_separation = sqrt(dx*dx + dy*dy);
    float initial_energy = -params.gravity_constant * initial[0].mass * initial[1].mass / initial_separation;
    for (const auto& p : initial) {
        initial_energy += 0.5f * p.mass * (p.vel.x * p.vel.x + p.vel.y * p.vel.y);
    }
    
    std::cout << "Initial conditions:\n";
    std::cout << "  Separation: " << initial_separation << "\n";
    std::cout << "  Total energy: " << initial_energy << "\n\n";
    
    // Test each algorithm
    std::vector<ForceAlgorithm> algorithms = {
        ForceAlgorithm::BRUTE_FORCE,
        ForceAlgorithm::BARNES_HUT,
        ForceAlgorithm::PARTICLE_MESH
    };
    
    std::vector<std::string> names = {
        "Brute Force (Reference)",
        "Barnes-Hut",
        "Particle Mesh"
    };
    
    std::vector<std::vector<Particle>> results(3);
    
    for (size_t i = 0; i < algorithms.size(); i++) {
        std::cout << "Testing " << names[i] << "...\n";
        
        // Create backend
        auto backend = std::make_unique<SimpleBackend>();
        backend->setAlgorithm(algorithms[i]);
        backend->initialize(initial.size(), params);
        
        // Set initial particles
        auto particles = initial;
        backend->setParticles(particles);
        
        // Run for 1000 steps
        int steps = 1000;
        for (int step = 0; step < steps; step++) {
            backend->step(params.dt);
        }
        
        // Get final state
        backend->getParticles(particles);
        results[i] = particles;
        
        // Calculate final values
        dx = particles[1].pos.x - particles[0].pos.x;
        dy = particles[1].pos.y - particles[0].pos.y;
        float final_separation = sqrt(dx*dx + dy*dy);
        float final_energy = -params.gravity_constant * particles[0].mass * particles[1].mass / final_separation;
        for (const auto& p : particles) {
            final_energy += 0.5f * p.mass * (p.vel.x * p.vel.x + p.vel.y * p.vel.y);
        }
        
        float energy_error = (final_energy - initial_energy) / initial_energy * 100;
        
        std::cout << "  Final separation: " << final_separation << "\n";
        std::cout << "  Final energy: " << final_energy << "\n";
        std::cout << "  Energy drift: " << energy_error << "%\n\n";
    }
    
    // Compare results
    std::cout << "=== Position Comparison vs Brute Force ===\n\n";
    
    for (size_t i = 1; i < algorithms.size(); i++) {
        std::cout << names[i] << ":\n";
        
        double max_error = 0;
        for (size_t j = 0; j < 2; j++) {
            float dx = results[i][j].pos.x - results[0][j].pos.x;
            float dy = results[i][j].pos.y - results[0][j].pos.y;
            float error = sqrt(dx*dx + dy*dy);
            max_error = std::max(max_error, (double)error);
            
            std::cout << "  Body " << j << " position error: " << std::scientific << error << "\n";
        }
        std::cout << "  Maximum error: " << max_error << "\n\n";
    }
    
    std::cout << "=== Conclusion ===\n";
    std::cout << "Brute Force provides the reference solution.\n";
    std::cout << "Barnes-Hut should show very small errors (<1e-6).\n";
    std::cout << "Particle Mesh may show larger errors due to grid discretization.\n";
    
    return 0;
}
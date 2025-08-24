#include "src/backend/SimpleBackend.cpp"
#include <iostream>
#include <iomanip>
#include <cmath>
#include <random>
#include <chrono>

// Test with a more realistic scenario - multiple bodies
int main() {
    std::cout << "=== Multi-Body Algorithm Accuracy Test ===\n\n";
    std::cout << "Testing 10-body system with different algorithms\n\n";
    
    // Simulation parameters
    SimulationParams params;
    params.box_size = 20.0f;
    params.gravity_constant = 0.1f;
    params.softening = 0.01f;  // Reasonable softening
    params.dt = 0.01f;
    params.grid_size = 256;
    params.theta = 0.5f;  // Standard Barnes-Hut accuracy
    
    // Create initial cluster of 10 bodies
    std::vector<Particle> initial(10);
    std::random_device rd;
    std::mt19937 gen(42);  // Fixed seed for reproducibility
    std::uniform_real_distribution<float> pos_dist(-2.0f, 2.0f);
    std::uniform_real_distribution<float> vel_dist(-0.1f, 0.1f);
    
    float center = params.box_size / 2;
    for (size_t i = 0; i < initial.size(); i++) {
        initial[i].pos.x = center + pos_dist(gen);
        initial[i].pos.y = center + pos_dist(gen);
        initial[i].vel.x = vel_dist(gen);
        initial[i].vel.y = vel_dist(gen);
        initial[i].mass = 1.0f;
        initial[i].radius = 0.01f;
    }
    
    // Calculate initial energy
    double initial_ke = 0, initial_pe = 0;
    for (const auto& p : initial) {
        initial_ke += 0.5 * p.mass * (p.vel.x * p.vel.x + p.vel.y * p.vel.y);
    }
    for (size_t i = 0; i < initial.size(); i++) {
        for (size_t j = i+1; j < initial.size(); j++) {
            float dx = initial[j].pos.x - initial[i].pos.x;
            float dy = initial[j].pos.y - initial[i].pos.y;
            float r = sqrt(dx*dx + dy*dy + params.softening*params.softening);
            initial_pe -= params.gravity_constant * initial[i].mass * initial[j].mass / r;
        }
    }
    double initial_energy = initial_ke + initial_pe;
    
    std::cout << "Initial total energy: " << initial_energy << "\n\n";
    
    // Test each algorithm
    std::vector<ForceAlgorithm> algorithms = {
        ForceAlgorithm::BRUTE_FORCE,
        ForceAlgorithm::BARNES_HUT,
        ForceAlgorithm::PARTICLE_MESH
    };
    
    std::vector<std::string> names = {
        "Brute Force (O(n²))",
        "Barnes-Hut (O(n log n))",
        "Particle Mesh (O(n))"
    };
    
    std::vector<std::vector<Particle>> results(3);
    std::vector<double> energies(3);
    std::vector<double> times(3);
    
    int steps = 100;
    
    for (size_t i = 0; i < algorithms.size(); i++) {
        std::cout << "Testing " << names[i] << "...\n";
        
        // Create backend
        auto backend = std::make_unique<SimpleBackend>();
        backend->setAlgorithm(algorithms[i]);
        backend->initialize(initial.size(), params);
        
        // Set initial particles
        auto particles = initial;
        backend->setParticles(particles);
        
        // Time the simulation
        auto start = std::chrono::high_resolution_clock::now();
        
        for (int step = 0; step < steps; step++) {
            backend->step(params.dt);
        }
        
        auto end = std::chrono::high_resolution_clock::now();
        times[i] = std::chrono::duration<double, std::milli>(end - start).count();
        
        // Get final state
        backend->getParticles(particles);
        results[i] = particles;
        
        // Calculate final energy
        double ke = 0, pe = 0;
        for (const auto& p : particles) {
            ke += 0.5 * p.mass * (p.vel.x * p.vel.x + p.vel.y * p.vel.y);
        }
        for (size_t j = 0; j < particles.size(); j++) {
            for (size_t k = j+1; k < particles.size(); k++) {
                float dx = particles[k].pos.x - particles[j].pos.x;
                float dy = particles[k].pos.y - particles[j].pos.y;
                float r = sqrt(dx*dx + dy*dy + params.softening*params.softening);
                pe -= params.gravity_constant * particles[j].mass * particles[k].mass / r;
            }
        }
        energies[i] = ke + pe;
        
        float energy_error = (energies[i] - initial_energy) / fabs(initial_energy) * 100;
        
        std::cout << "  Time: " << times[i] << " ms\n";
        std::cout << "  Final energy: " << energies[i] << "\n";
        std::cout << "  Energy drift: " << energy_error << "%\n\n";
    }
    
    // Compare positions
    std::cout << "=== Position Comparison vs Brute Force ===\n\n";
    
    for (size_t i = 1; i < algorithms.size(); i++) {
        std::cout << names[i] << ":\n";
        
        double total_error = 0;
        double max_error = 0;
        
        for (size_t j = 0; j < initial.size(); j++) {
            float dx = results[i][j].pos.x - results[0][j].pos.x;
            float dy = results[i][j].pos.y - results[0][j].pos.y;
            float error = sqrt(dx*dx + dy*dy);
            total_error += error;
            max_error = std::max(max_error, (double)error);
        }
        
        std::cout << "  Average position error: " << std::scientific << total_error/initial.size() << "\n";
        std::cout << "  Maximum position error: " << max_error << "\n";
        std::cout << "  Speedup: " << std::fixed << std::setprecision(2) << times[0]/times[i] << "x\n\n";
    }
    
    std::cout << "=== Performance Summary ===\n\n";
    for (size_t i = 0; i < algorithms.size(); i++) {
        std::cout << std::left << std::setw(25) << names[i] 
                  << " Time: " << std::right << std::setw(8) << std::fixed << std::setprecision(2) 
                  << times[i] << " ms"
                  << "  Energy drift: " << std::setw(7) << std::setprecision(3) 
                  << (energies[i] - initial_energy) / fabs(initial_energy) * 100 << "%\n";
    }
    
    std::cout << "\n=== Conclusion ===\n";
    std::cout << "With " << initial.size() << " bodies and " << steps << " timesteps:\n";
    std::cout << "- Brute Force provides the reference solution\n";
    std::cout << "- Barnes-Hut should be faster with similar accuracy\n";
    std::cout << "- Particle Mesh is fastest but less accurate for small N\n";
    
    return 0;
}
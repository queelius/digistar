#include "src/backend/SimpleBackend.cpp"
#include <iostream>
#include <iomanip>
#include <chrono>
#include <random>
#include <cmath>

// Faster version of backend test with smaller particle counts
int main() {
    std::cout << "=== Testing Backend (Fast Version) ===\n\n";
    
    // Test parameters
    SimulationParams params;
    params.box_size = 100.0f;
    params.gravity_constant = 1.0f;
    params.softening = 0.01f;
    params.dt = 0.001f;
    params.grid_size = 128;  // Smaller grid
    params.theta = 0.5f;
    
    // Test with smaller particle counts
    std::vector<int> particle_counts = {10, 100, 1000};
    
    std::cout << "Performance and Accuracy Test:\n";
    std::cout << std::setw(8) << "N"
              << std::setw(15) << "Algorithm"
              << std::setw(12) << "ms/step"
              << std::setw(12) << "E drift %"
              << "\n";
    std::cout << std::string(47, '-') << "\n";
    
    for (int n : particle_counts) {
        // Create test particles (cluster)
        std::vector<Particle> initial(n);
        std::mt19937 gen(42);
        std::normal_distribution<float> pos_dist(params.box_size/2, 5.0f);
        std::normal_distribution<float> vel_dist(0, 0.5f);
        
        for (auto& p : initial) {
            p.pos.x = pos_dist(gen);
            p.pos.y = pos_dist(gen);
            p.vel.x = vel_dist(gen);
            p.vel.y = vel_dist(gen);
            p.mass = 1.0f / n;
            p.radius = 0.01f;
        }
        
        // Calculate initial energy
        double e_initial = 0;
        for (const auto& p : initial) {
            e_initial += 0.5 * p.mass * (p.vel.x*p.vel.x + p.vel.y*p.vel.y);
        }
        for (int i = 0; i < n; i++) {
            for (int j = i+1; j < n; j++) {
                float dx = initial[j].pos.x - initial[i].pos.x;
                float dy = initial[j].pos.y - initial[i].pos.y;
                float r = sqrt(dx*dx + dy*dy + params.softening*params.softening);
                e_initial -= params.gravity_constant * initial[i].mass * initial[j].mass / r;
            }
        }
        
        // Test each algorithm
        std::vector<ForceAlgorithm> algorithms;
        std::vector<std::string> names;
        
        if (n <= 100) {
            algorithms.push_back(ForceAlgorithm::BRUTE_FORCE);
            names.push_back("Brute Force");
        }
        
        algorithms.push_back(ForceAlgorithm::BARNES_HUT);
        names.push_back("Barnes-Hut");
        
        if (n >= 100) {
            algorithms.push_back(ForceAlgorithm::PARTICLE_MESH);
            names.push_back("Particle Mesh");
        }
        
        for (size_t a = 0; a < algorithms.size(); a++) {
            auto backend = std::make_unique<SimpleBackend>();
            backend->setAlgorithm(algorithms[a]);
            backend->initialize(n, params);
            
            auto particles = initial;
            backend->setParticles(particles);
            
            // Run for 10 steps (not 100) and measure time
            auto start = std::chrono::high_resolution_clock::now();
            
            for (int step = 0; step < 10; step++) {
                backend->step(params.dt);
            }
            
            auto end = std::chrono::high_resolution_clock::now();
            double time_ms = std::chrono::duration<double, std::milli>(end - start).count() / 10;
            
            // Get final particles and calculate energy
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
            
            double e_drift = (e_final - e_initial) / fabs(e_initial) * 100;
            
            std::cout << std::setw(8) << n;
            std::cout << std::setw(15) << names[a];
            std::cout << std::fixed << std::setprecision(2);
            std::cout << std::setw(12) << time_ms;
            std::cout << std::setw(12) << e_drift;
            std::cout << "\n";
        }
        std::cout << "\n";
    }
    
    std::cout << "Test complete!\n";
    return 0;
}
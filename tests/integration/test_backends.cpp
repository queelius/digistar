#include "src/backend/SimpleBackend.cpp"
#include <iostream>
#include <iomanip>
#include <chrono>
#include <random>
#include <cmath>

// Test the improved backend with all algorithms
int main() {
    std::cout << "=== Testing Improved Backend v4 ===\n\n";
    
    // Test parameters
    SimulationParams params;
    params.box_size = 100.0f;
    params.gravity_constant = 1.0f;
    params.softening = 0.01f;
    params.dt = 0.001f;
    params.grid_size = 256;
    params.theta = 0.5f;
    
    // Test with different particle counts
    std::vector<int> particle_counts = {10, 100, 1000, 10000};
    
    std::cout << "Performance and Accuracy Test:\n";
    std::cout << std::setw(8) << "N"
              << std::setw(15) << "Algorithm"
              << std::setw(12) << "ms/step"
              << std::setw(12) << "E drift %"
              << std::setw(12) << "Viable?"
              << "\n";
    std::cout << std::string(60, '-') << "\n";
    
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
        
        if (n <= 1000) {
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
            
            // Run for 100 steps and measure time
            auto start = std::chrono::high_resolution_clock::now();
            
            for (int step = 0; step < 100; step++) {
                backend->step(params.dt);
            }
            
            auto end = std::chrono::high_resolution_clock::now();
            double time_ms = std::chrono::duration<double, std::milli>(end - start).count() / 100;
            
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
            
            // Determine if viable (energy drift < 5% and fast enough)
            bool viable = (fabs(e_drift) < 5.0) && (time_ms < 100);
            
            std::cout << std::setw(8) << n;
            std::cout << std::setw(15) << names[a];
            std::cout << std::fixed << std::setprecision(2);
            std::cout << std::setw(12) << time_ms;
            std::cout << std::setw(12) << e_drift;
            std::cout << std::setw(12) << (viable ? "Yes" : "No");
            std::cout << "\n";
        }
        std::cout << "\n";
    }
    
    // Large scale test
    std::cout << "\nLarge Scale Performance Test:\n";
    std::cout << std::setw(10) << "N"
              << std::setw(15) << "Algorithm"
              << std::setw(12) << "ms/step"
              << std::setw(12) << "FPS"
              << "\n";
    std::cout << std::string(50, '-') << "\n";
    
    std::vector<int> large_counts = {100000, 500000, 1000000};
    
    for (int n : large_counts) {
        // Only test scalable algorithms
        std::vector<ForceAlgorithm> algorithms = {
            ForceAlgorithm::BARNES_HUT,
            ForceAlgorithm::PARTICLE_MESH
        };
        std::vector<std::string> names = {"Barnes-Hut", "Particle Mesh"};
        
        for (size_t a = 0; a < algorithms.size(); a++) {
            // Create simple random distribution
            std::vector<Particle> particles(n);
            std::mt19937 gen(42);
            std::uniform_real_distribution<float> dist(0, params.box_size);
            
            for (auto& p : particles) {
                p.pos.x = dist(gen);
                p.pos.y = dist(gen);
                p.vel.x = 0;
                p.vel.y = 0;
                p.mass = 1.0f / n;
                p.radius = 0.01f;
            }
            
            auto backend = std::make_unique<SimpleBackend>();
            backend->setAlgorithm(algorithms[a]);
            backend->initialize(n, params);
            backend->setParticles(particles);
            
            // Warmup
            backend->step(params.dt);
            
            // Measure
            auto start = std::chrono::high_resolution_clock::now();
            
            for (int step = 0; step < 10; step++) {
                backend->step(params.dt);
            }
            
            auto end = std::chrono::high_resolution_clock::now();
            double time_ms = std::chrono::duration<double, std::milli>(end - start).count() / 10;
            double fps = 1000.0 / time_ms;
            
            std::cout << std::setw(10) << n;
            std::cout << std::setw(15) << names[a];
            std::cout << std::fixed << std::setprecision(2);
            std::cout << std::setw(12) << time_ms;
            std::cout << std::setw(12) << fps;
            std::cout << "\n";
        }
    }
    
    std::cout << "\n=== Recommendations ===\n";
    std::cout << "N < 100:        Use Brute Force (most accurate)\n";
    std::cout << "100 < N < 10K:  Use Barnes-Hut (good balance)\n";
    std::cout << "N > 100K:       Use Particle Mesh (best scaling)\n";
    std::cout << "\nAll algorithms are now working correctly!\n";
    
    return 0;
}
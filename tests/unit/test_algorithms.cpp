#include "src/backend/ISimulationBackend.h"
#include "src/backend/SimpleBackend.cpp"
#include <iostream>
#include <chrono>
#include <vector>
#include <cmath>
#include <iomanip>

std::vector<Particle> createGalaxyParticles(size_t count, float box_size) {
    std::vector<Particle> particles(count);
    
    // Create spiral galaxy structure
    for (size_t i = 0; i < count; i++) {
        float t = (float)i / count * 10.0f;
        float angle = t * 2.0f;
        float radius = 10.0f + t * 5.0f;
        
        // Add some scatter
        radius += (rand() / (float)RAND_MAX - 0.5f) * 5.0f;
        angle += (rand() / (float)RAND_MAX - 0.5f) * 0.5f;
        
        particles[i].pos.x = box_size/2 + radius * cos(angle);
        particles[i].pos.y = box_size/2 + radius * sin(angle);
        
        // Orbital velocity
        float v_orbit = sqrt(10.0f / radius);
        particles[i].vel.x = -v_orbit * sin(angle);
        particles[i].vel.y = v_orbit * cos(angle);
        
        particles[i].mass = 1.0f;
        particles[i].radius = 0.5f;
    }
    
    // Add central supermassive object
    particles[0].pos.x = box_size/2;
    particles[0].pos.y = box_size/2;
    particles[0].vel.x = 0;
    particles[0].vel.y = 0;
    particles[0].mass = count * 0.1f;  // 10% of total mass
    
    return particles;
}

void benchmarkAlgorithm(
    ISimulationBackend* backend, 
    ForceAlgorithm algo,
    size_t num_particles,
    int num_steps = 20
) {
    std::string algo_name;
    switch (algo) {
        case ForceAlgorithm::BRUTE_FORCE: algo_name = "Brute Force"; break;
        case ForceAlgorithm::BARNES_HUT: algo_name = "Barnes-Hut"; break;
        case ForceAlgorithm::PARTICLE_MESH: algo_name = "Particle Mesh"; break;
        case ForceAlgorithm::HYBRID: algo_name = "Hybrid PM+Direct"; break;
    }
    
    std::cout << "\nTesting " << algo_name << " with " << num_particles << " particles" << std::endl;
    
    // Check if backend supports this algorithm
    if (!backend->supportsAlgorithm(algo)) {
        std::cout << "  Algorithm not supported by this backend" << std::endl;
        return;
    }
    
    // Initialize
    SimulationParams params;
    params.box_size = 200.0f;
    params.gravity_constant = 1.0f;
    params.softening = 0.1f;
    params.dt = 0.01f;
    params.theta = 0.5f;  // Barnes-Hut opening angle
    params.grid_size = 128;  // PM grid
    
    backend->setAlgorithm(algo);
    backend->initialize(num_particles, params);
    
    auto particles = createGalaxyParticles(num_particles, params.box_size);
    backend->setParticles(particles);
    
    // Warmup
    for (int i = 0; i < 5; i++) {
        backend->step(params.dt);
    }
    
    // Benchmark
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < num_steps; i++) {
        backend->step(params.dt);
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    float ms_per_step = duration.count() / (float)num_steps;
    float fps = 1000.0f / ms_per_step;
    
    // Calculate theoretical complexity
    float complexity;
    switch (algo) {
        case ForceAlgorithm::BRUTE_FORCE: 
            complexity = num_particles * num_particles;
            break;
        case ForceAlgorithm::BARNES_HUT:
            complexity = num_particles * log2(num_particles);
            break;
        case ForceAlgorithm::PARTICLE_MESH:
            complexity = num_particles + params.grid_size * params.grid_size * log2(params.grid_size);
            break;
        default:
            complexity = num_particles * num_particles;
    }
    
    float efficiency = complexity / (ms_per_step * 1000);  // Operations per microsecond
    
    std::cout << "  Backend: " << backend->getBackendName() << std::endl;
    std::cout << "  Time per step: " << std::fixed << std::setprecision(2) 
              << ms_per_step << " ms" << std::endl;
    std::cout << "  FPS: " << fps << std::endl;
    std::cout << "  Efficiency: " << efficiency << " ops/µs" << std::endl;
    std::cout << "  Memory: " << backend->getMemoryUsage() / (1024*1024) << " MB" << std::endl;
}

int main() {
    std::cout << "=== Algorithm Comparison Test ===" << std::endl;
    std::cout << "Testing different force calculation algorithms\n" << std::endl;
    
    // Create backend
    auto backend = std::make_unique<SimpleBackend>();
    
    // Test different particle counts and algorithms
    std::vector<size_t> particle_counts = {500, 1000, 2000, 5000, 10000};
    std::vector<ForceAlgorithm> algorithms = {
        ForceAlgorithm::BRUTE_FORCE,
        ForceAlgorithm::BARNES_HUT,
        ForceAlgorithm::PARTICLE_MESH,
        ForceAlgorithm::HYBRID
    };
    
    for (size_t count : particle_counts) {
        std::cout << "\n" << std::string(60, '=') << std::endl;
        std::cout << "Particle Count: " << count << std::endl;
        std::cout << std::string(60, '-') << std::endl;
        
        for (ForceAlgorithm algo : algorithms) {
            benchmarkAlgorithm(backend.get(), algo, count);
        }
    }
    
    std::cout << "\n=== Performance Expectations ===" << std::endl;
    std::cout << "Algorithm     Complexity    Best for" << std::endl;
    std::cout << "------------------------------------------------" << std::endl;
    std::cout << "Brute Force   O(n²)        < 10K particles" << std::endl;
    std::cout << "Barnes-Hut    O(n log n)   10K - 1M particles" << std::endl;
    std::cout << "Particle Mesh O(n)         > 100K particles" << std::endl;
    std::cout << "Hybrid        O(n)         Complex interactions" << std::endl;
    
    std::cout << "\n=== Next Steps ===" << std::endl;
    std::cout << "1. Implement QuadTree for Barnes-Hut" << std::endl;
    std::cout << "2. Implement FFT-based PM solver" << std::endl;
    std::cout << "3. Add AVX2 optimizations to each algorithm" << std::endl;
    std::cout << "4. Port algorithms to CUDA" << std::endl;
    
    return 0;
}
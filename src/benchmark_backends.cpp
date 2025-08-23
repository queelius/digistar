#include "backend/ISimulationBackend.h"
#include <iostream>
#include <chrono>
#include <vector>
#include <random>
#include <iomanip>

// Initialize particles with random positions and velocities
std::vector<Particle> createParticles(size_t count, float box_size) {
    std::vector<Particle> particles(count);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> pos_dist(0, box_size);
    std::uniform_real_distribution<float> vel_dist(-10, 10);
    std::uniform_real_distribution<float> mass_dist(0.5, 2.0);
    
    for (auto& p : particles) {
        p.pos.x = pos_dist(gen);
        p.pos.y = pos_dist(gen);
        p.vel.x = vel_dist(gen);
        p.vel.y = vel_dist(gen);
        p.mass = mass_dist(gen);
        p.radius = 0.5f;
    }
    
    return particles;
}

// Benchmark a specific backend
void benchmarkBackend(ISimulationBackend* backend, size_t num_particles, int num_steps = 100) {
    std::cout << "\n=== Benchmarking " << backend->getBackendName() << " ===" << std::endl;
    std::cout << "Particles: " << num_particles << std::endl;
    std::cout << "Max particles: " << backend->getMaxParticles() << std::endl;
    
    // Initialize
    SimulationParams params;
    params.box_size = 1000.0f;
    params.gravity_constant = 1.0f;
    params.softening = 0.1f;
    params.dt = 0.016f;
    params.grid_size = 256;  // Smaller grid for CPU backends
    
    backend->initialize(num_particles, params);
    
    // Create and upload particles
    auto particles = createParticles(num_particles, params.box_size);
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
    
    // Report results
    float ms_per_step = duration.count() / (float)num_steps;
    float fps = 1000.0f / ms_per_step;
    float particles_per_sec = (num_particles * fps) / 1e6;
    
    std::cout << "Time per step: " << std::fixed << std::setprecision(2) 
              << ms_per_step << " ms" << std::endl;
    std::cout << "FPS: " << fps << std::endl;
    std::cout << "Million particles/sec: " << particles_per_sec << std::endl;
    std::cout << "Memory usage: " << (backend->getMemoryUsage() / (1024*1024)) 
              << " MB" << std::endl;
    
    // Verify particles are moving
    backend->getParticles(particles);
    float avg_speed = 0;
    for (const auto& p : particles) {
        avg_speed += sqrt(p.vel.x * p.vel.x + p.vel.y * p.vel.y);
    }
    avg_speed /= particles.size();
    std::cout << "Average particle speed: " << avg_speed << std::endl;
}

int main(int argc, char* argv[]) {
    std::cout << "=== Multi-Backend Particle Simulation Benchmark ===" << std::endl;
    
    // Parse command line
    BackendFactory::BackendType type = BackendFactory::AUTO;
    if (argc > 1) {
        std::string arg = argv[1];
        if (arg == "cuda") type = BackendFactory::CUDA;
        else if (arg == "avx2") type = BackendFactory::AVX2;
        else if (arg == "sse2") type = BackendFactory::SCALAR;
        else if (arg == "auto") type = BackendFactory::AUTO;
        else {
            std::cout << "Usage: " << argv[0] << " [cuda|avx2|sse2|auto]" << std::endl;
            return 1;
        }
    }
    
    // System info
    std::cout << "\nSystem capabilities:" << std::endl;
    std::cout << "  CUDA available: " << (BackendFactory::hasCUDA() ? "Yes" : "No") << std::endl;
    std::cout << "  AVX2 support: " << (BackendFactory::hasAVX2() ? "Yes" : "No") << std::endl;
    std::cout << "  CPU threads: " << BackendFactory::getNumCPUCores() << std::endl;
    
    // Test different particle counts
    std::vector<size_t> test_counts;
    
    if (type == BackendFactory::CUDA) {
        // GPU can handle more
        test_counts = {10000, 100000, 1000000, 5000000, 10000000};
    } else {
        // CPU backends
        test_counts = {1000, 10000, 50000, 100000, 500000};
    }
    
    // Create backend
    auto backend = BackendFactory::create(type, test_counts.back());
    
    if (!backend) {
        std::cerr << "Failed to create backend!" << std::endl;
        return 1;
    }
    
    // Run benchmarks
    for (size_t count : test_counts) {
        try {
            benchmarkBackend(backend.get(), count);
        } catch (const std::exception& e) {
            std::cerr << "Benchmark failed at " << count 
                      << " particles: " << e.what() << std::endl;
            break;
        }
    }
    
    // Compare backends if AUTO mode
    if (type == BackendFactory::AUTO) {
        std::cout << "\n=== Backend Comparison ===" << std::endl;
        
        size_t test_count = 10000;  // Use same count for fair comparison
        
        // Test each available backend
        if (BackendFactory::hasCUDA()) {
            auto cuda_backend = BackendFactory::create(BackendFactory::CUDA, test_count);
            benchmarkBackend(cuda_backend.get(), test_count, 50);
        }
        
        if (BackendFactory::hasAVX2()) {
            auto avx2_backend = BackendFactory::create(BackendFactory::AVX2, test_count);
            benchmarkBackend(avx2_backend.get(), test_count, 50);
        }
        
        auto sse2_backend = BackendFactory::create(BackendFactory::SCALAR, test_count);
        benchmarkBackend(sse2_backend.get(), test_count, 50);
    }
    
    return 0;
}
#include <iostream>
#include <chrono>
#include <vector>
#include <memory>
#include <iomanip>
#include <random>
#include <cmath>
#include <fstream>

#include "src/backend/ISimulationBackend.h"
#include "src/backend/BackendFactory.cpp"

// Test scenarios
std::vector<Particle> createGalaxy(size_t n, float box_size) {
    std::vector<Particle> particles(n);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> scatter(0.0f, 2.0f);
    
    // Central black hole
    particles[0].pos = {box_size/2, box_size/2};
    particles[0].vel = {0, 0};
    particles[0].mass = n * 0.05f;
    particles[0].radius = 2.0f;
    
    // Spiral arms
    for (size_t i = 1; i < n; i++) {
        float t = (float)i / n;
        int arm = i % 3;
        float angle = t * 4 * M_PI + (arm * 2 * M_PI / 3);
        float radius = 10.0f * exp(0.15f * angle);
        radius = fmin(radius, box_size * 0.4f);
        
        radius += scatter(gen);
        angle += scatter(gen) * 0.1f;
        
        particles[i].pos.x = box_size/2 + radius * cos(angle);
        particles[i].pos.y = box_size/2 + radius * sin(angle);
        
        float v_orbit = sqrt(50.0f / radius);
        particles[i].vel.x = -v_orbit * sin(angle);
        particles[i].vel.y = v_orbit * cos(angle);
        
        particles[i].mass = 0.1f + (rand() / (float)RAND_MAX) * 0.9f;
        particles[i].radius = 0.5f;
    }
    
    return particles;
}

std::vector<Particle> createUniform(size_t n, float box_size) {
    std::vector<Particle> particles(n);
    
    for (size_t i = 0; i < n; i++) {
        particles[i].pos.x = (rand() / (float)RAND_MAX) * box_size;
        particles[i].pos.y = (rand() / (float)RAND_MAX) * box_size;
        particles[i].vel.x = (rand() / (float)RAND_MAX - 0.5f) * 10;
        particles[i].vel.y = (rand() / (float)RAND_MAX - 0.5f) * 10;
        particles[i].mass = 0.5f + (rand() / (float)RAND_MAX) * 1.5f;
        particles[i].radius = 0.5f;
    }
    
    return particles;
}

int main() {
    std::cout << "=== Digital Star Backend Benchmark ===\n\n";
    
    // Check available backends
    std::cout << "System Info:\n";
    std::cout << "  CPU Cores: " << BackendFactory::getNumCPUCores() << "\n";
    std::cout << "  AVX2: " << (BackendFactory::hasAVX2() ? "Yes" : "No") << "\n";
    std::cout << "  CUDA: " << (BackendFactory::hasCUDA() ? "Yes" : "No") << "\n\n";
    
    // Test parameters
    SimulationParams params;
    params.box_size = 1000.0f;
    params.gravity_constant = 1.0f;
    params.softening = 0.5f;
    params.dt = 0.01f;
    params.theta = 0.5f;
    
    // Test configurations
    std::vector<size_t> particle_counts = {1000, 5000, 10000};
    if (BackendFactory::hasAVX2()) {
        particle_counts.push_back(20000);
    }
    
    // Test each configuration
    for (size_t n : particle_counts) {
        std::cout << "\nTesting with " << n << " particles:\n";
        std::cout << std::string(50, '-') << "\n";
        
        auto particles = createGalaxy(n, params.box_size);
        
        // Test CPU + Brute Force
        if (n <= 5000) {
            auto backend = BackendFactory::create(BackendType::CPU, ForceAlgorithm::BRUTE_FORCE);
            backend->initialize(n, params);
            backend->setParticles(particles);
            
            auto start = std::chrono::high_resolution_clock::now();
            for (int i = 0; i < 10; i++) backend->step(params.dt);
            auto end = std::chrono::high_resolution_clock::now();
            
            double ms = std::chrono::duration<double, std::milli>(end - start).count() / 10;
            std::cout << "  CPU + Brute Force: " << ms << " ms/step\n";
        }
        
        // Test CPU + Barnes-Hut
        {
            auto backend = BackendFactory::create(BackendType::CPU, ForceAlgorithm::BARNES_HUT);
            backend->initialize(n, params);
            backend->setParticles(particles);
            
            auto start = std::chrono::high_resolution_clock::now();
            for (int i = 0; i < 10; i++) backend->step(params.dt);
            auto end = std::chrono::high_resolution_clock::now();
            
            double ms = std::chrono::duration<double, std::milli>(end - start).count() / 10;
            std::cout << "  CPU + Barnes-Hut: " << ms << " ms/step\n";
        }
        
        // Test AVX2 if available
        if (BackendFactory::hasAVX2()) {
            if (n <= 5000) {
                auto backend = BackendFactory::create(BackendType::AVX2, ForceAlgorithm::BRUTE_FORCE);
                backend->initialize(n, params);
                backend->setParticles(particles);
                
                auto start = std::chrono::high_resolution_clock::now();
                for (int i = 0; i < 10; i++) backend->step(params.dt);
                auto end = std::chrono::high_resolution_clock::now();
                
                double ms = std::chrono::duration<double, std::milli>(end - start).count() / 10;
                std::cout << "  AVX2 + Brute Force: " << ms << " ms/step\n";
            }
            
            {
                auto backend = BackendFactory::create(BackendType::AVX2, ForceAlgorithm::BARNES_HUT);
                backend->initialize(n, params);
                backend->setParticles(particles);
                
                auto start = std::chrono::high_resolution_clock::now();
                for (int i = 0; i < 10; i++) backend->step(params.dt);
                auto end = std::chrono::high_resolution_clock::now();
                
                double ms = std::chrono::duration<double, std::milli>(end - start).count() / 10;
                std::cout << "  AVX2 + Barnes-Hut: " << ms << " ms/step\n";
            }
        }
    }
    
    return 0;
}

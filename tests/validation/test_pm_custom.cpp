#include "src/backend/SimpleBackend.cpp"
#include <iostream>
#include <chrono>

int main() {
    std::cout << "Testing custom FFT-based PM\n";
    
    // Simple test with small particle count
    SimulationParams params;
    params.box_size = 1000.0f;
    params.gravity_constant = 1.0f;
    params.softening = 0.5f;
    params.dt = 0.01f;
    params.grid_size = 64;  // Power of 2
    
    size_t n = 1000;
    std::vector<Particle> particles(n);
    
    // Create simple particle distribution
    for (size_t i = 0; i < n; i++) {
        particles[i].pos.x = (rand() / (float)RAND_MAX) * params.box_size;
        particles[i].pos.y = (rand() / (float)RAND_MAX) * params.box_size;
        particles[i].vel.x = 0;
        particles[i].vel.y = 0;
        particles[i].mass = 1.0f;
        particles[i].radius = 0.5f;
    }
    
    try {
        std::cout << "Creating backend...\n";
        auto backend = std::make_unique<SimpleBackend>();
        
        std::cout << "Setting PM algorithm...\n";
        backend->setAlgorithm(ForceAlgorithm::PARTICLE_MESH);
        
        std::cout << "Initializing with " << n << " particles...\n";
        backend->initialize(n, params);
        
        std::cout << "Setting particles...\n";
        backend->setParticles(particles);
        
        std::cout << "Running step...\n";
        auto start = std::chrono::high_resolution_clock::now();
        backend->step(params.dt);
        auto end = std::chrono::high_resolution_clock::now();
        
        double ms = std::chrono::duration<double, std::milli>(end - start).count();
        std::cout << "Step completed in " << ms << " ms\n";
        
        // Try more particles
        std::vector<size_t> test_counts = {5000, 10000, 50000, 100000};
        
        for (size_t test_n : test_counts) {
            std::cout << "\nTesting with " << test_n << " particles:\n";
            
            particles.resize(test_n);
            for (size_t i = n; i < test_n; i++) {
                particles[i].pos.x = (rand() / (float)RAND_MAX) * params.box_size;
                particles[i].pos.y = (rand() / (float)RAND_MAX) * params.box_size;
                particles[i].vel.x = 0;
                particles[i].vel.y = 0;
                particles[i].mass = 1.0f;
                particles[i].radius = 0.5f;
            }
            
            backend->initialize(test_n, params);
            backend->setParticles(particles);
            
            start = std::chrono::high_resolution_clock::now();
            for (int i = 0; i < 5; i++) {
                backend->step(params.dt);
            }
            end = std::chrono::high_resolution_clock::now();
            
            ms = std::chrono::duration<double, std::milli>(end - start).count() / 5;
            std::cout << "  Average step time: " << ms << " ms\n";
            std::cout << "  FPS: " << 1000.0/ms << "\n";
        }
        
    } catch (const std::exception& e) {
        std::cout << "Error: " << e.what() << "\n";
        return 1;
    }
    
    std::cout << "\n✓ Custom FFT-based PM working!\n";
    return 0;
}
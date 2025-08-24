#include "src/backend/SimpleBackend.cpp"
#include <iostream>
#include <iomanip>
#include <cmath>
#include <vector>
#include <chrono>

// Test convergence by varying accuracy parameters
int main() {
    std::cout << "=== Algorithm Convergence Test ===\n\n";
    std::cout << "Testing if algorithms converge to same solution with increasing accuracy\n\n";
    
    // Create a simple 3-body problem for testing
    std::vector<Particle> initial(3);
    
    // Central massive body
    initial[0].pos = {10.0f, 10.0f};
    initial[0].vel = {0.0f, 0.0f};
    initial[0].mass = 100.0f;
    initial[0].radius = 0.1f;
    
    // Orbiting body 1
    initial[1].pos = {11.0f, 10.0f};
    initial[1].vel = {0.0f, 10.0f};
    initial[1].mass = 1.0f;
    initial[1].radius = 0.01f;
    
    // Orbiting body 2
    initial[2].pos = {10.0f, 12.0f};
    initial[2].vel = {-7.0f, 0.0f};
    initial[2].mass = 1.0f;
    initial[2].radius = 0.01f;
    
    // Test parameters to vary
    std::vector<float> timesteps = {0.01f, 0.005f, 0.001f, 0.0005f, 0.0001f};
    std::vector<float> softenings = {0.1f, 0.01f, 0.001f, 0.0001f};
    std::vector<int> grid_sizes = {64, 128, 256, 512, 1024};
    std::vector<float> thetas = {0.9f, 0.7f, 0.5f, 0.3f, 0.1f};  // Barnes-Hut accuracy
    
    std::cout << "=== Testing Timestep Convergence ===\n\n";
    std::cout << std::setw(10) << "dt" 
              << std::setw(20) << "Brute Force" 
              << std::setw(20) << "Barnes-Hut"
              << std::setw(20) << "Particle Mesh"
              << std::setw(20) << "BH vs BF Error"
              << std::setw(20) << "PM vs BF Error" << "\n";
    std::cout << std::string(110, '-') << "\n";
    
    for (float dt : timesteps) {
        SimulationParams params;
        params.box_size = 20.0f;
        params.gravity_constant = 1.0f;
        params.softening = 0.001f;
        params.dt = dt;
        params.grid_size = 256;
        params.theta = 0.5f;
        
        std::vector<std::vector<Particle>> results(3);
        std::vector<float2> final_positions[3];
        
        // Test each algorithm
        std::vector<ForceAlgorithm> algorithms = {
            ForceAlgorithm::BRUTE_FORCE,
            ForceAlgorithm::BARNES_HUT,
            ForceAlgorithm::PARTICLE_MESH
        };
        
        for (size_t i = 0; i < algorithms.size(); i++) {
            auto backend = std::make_unique<SimpleBackend>();
            backend->setAlgorithm(algorithms[i]);
            backend->initialize(initial.size(), params);
            
            auto particles = initial;
            backend->setParticles(particles);
            
            // Run for fixed simulation time
            float total_time = 1.0f;
            int steps = (int)(total_time / dt);
            
            for (int step = 0; step < steps; step++) {
                backend->step(params.dt);
            }
            
            backend->getParticles(particles);
            results[i] = particles;
            
            // Store center of mass
            float cx = 0, cy = 0, total_mass = 0;
            for (const auto& p : particles) {
                cx += p.pos.x * p.mass;
                cy += p.pos.y * p.mass;
                total_mass += p.mass;
            }
            final_positions[i].push_back({cx/total_mass, cy/total_mass});
        }
        
        // Calculate errors
        float bh_error = 0, pm_error = 0;
        for (size_t j = 0; j < initial.size(); j++) {
            float dx_bh = results[1][j].pos.x - results[0][j].pos.x;
            float dy_bh = results[1][j].pos.y - results[0][j].pos.y;
            bh_error += sqrt(dx_bh*dx_bh + dy_bh*dy_bh);
            
            float dx_pm = results[2][j].pos.x - results[0][j].pos.x;
            float dy_pm = results[2][j].pos.y - results[0][j].pos.y;
            pm_error += sqrt(dx_pm*dx_pm + dy_pm*dy_pm);
        }
        bh_error /= initial.size();
        pm_error /= initial.size();
        
        std::cout << std::scientific << std::setprecision(2);
        std::cout << std::setw(10) << dt;
        std::cout << std::setw(20) << final_positions[0][0].x;
        std::cout << std::setw(20) << final_positions[1][0].x;
        std::cout << std::setw(20) << final_positions[2][0].x;
        std::cout << std::setw(20) << bh_error;
        std::cout << std::setw(20) << pm_error << "\n";
    }
    
    std::cout << "\n=== Testing Barnes-Hut Theta Convergence ===\n\n";
    std::cout << std::setw(10) << "theta" 
              << std::setw(20) << "CoM Position"
              << std::setw(20) << "vs BF Error"
              << std::setw(20) << "Time (ms)" << "\n";
    std::cout << std::string(70, '-') << "\n";
    
    SimulationParams base_params;
    base_params.box_size = 20.0f;
    base_params.gravity_constant = 1.0f;
    base_params.softening = 0.001f;
    base_params.dt = 0.001f;
    base_params.grid_size = 256;
    
    // Get reference solution from Brute Force
    auto backend_ref = std::make_unique<SimpleBackend>();
    backend_ref->setAlgorithm(ForceAlgorithm::BRUTE_FORCE);
    backend_ref->initialize(initial.size(), base_params);
    auto particles_ref = initial;
    backend_ref->setParticles(particles_ref);
    
    for (int step = 0; step < 1000; step++) {
        backend_ref->step(base_params.dt);
    }
    backend_ref->getParticles(particles_ref);
    
    for (float theta : thetas) {
        base_params.theta = theta;
        
        auto backend = std::make_unique<SimpleBackend>();
        backend->setAlgorithm(ForceAlgorithm::BARNES_HUT);
        backend->initialize(initial.size(), base_params);
        
        auto particles = initial;
        backend->setParticles(particles);
        
        auto start = std::chrono::high_resolution_clock::now();
        for (int step = 0; step < 1000; step++) {
            backend->step(base_params.dt);
        }
        auto end = std::chrono::high_resolution_clock::now();
        double time_ms = std::chrono::duration<double, std::milli>(end - start).count();
        
        backend->getParticles(particles);
        
        // Calculate error vs reference
        float error = 0;
        for (size_t j = 0; j < initial.size(); j++) {
            float dx = particles[j].pos.x - particles_ref[j].pos.x;
            float dy = particles[j].pos.y - particles_ref[j].pos.y;
            error += sqrt(dx*dx + dy*dy);
        }
        error /= initial.size();
        
        // Center of mass
        float cx = 0, cy = 0, total_mass = 0;
        for (const auto& p : particles) {
            cx += p.pos.x * p.mass;
            cy += p.pos.y * p.mass;
            total_mass += p.mass;
        }
        
        std::cout << std::fixed << std::setprecision(2);
        std::cout << std::setw(10) << theta;
        std::cout << std::scientific << std::setprecision(3);
        std::cout << std::setw(20) << cx/total_mass;
        std::cout << std::setw(20) << error;
        std::cout << std::fixed << std::setprecision(2);
        std::cout << std::setw(20) << time_ms << "\n";
    }
    
    std::cout << "\n=== Testing Particle Mesh Grid Convergence ===\n\n";
    std::cout << std::setw(10) << "grid" 
              << std::setw(20) << "CoM Position"
              << std::setw(20) << "vs BF Error"
              << std::setw(20) << "Time (ms)"
              << std::setw(20) << "Memory (MB)" << "\n";
    std::cout << std::string(90, '-') << "\n";
    
    for (int grid_size : grid_sizes) {
        base_params.grid_size = grid_size;
        base_params.theta = 0.5f;
        
        auto backend = std::make_unique<SimpleBackend>();
        backend->setAlgorithm(ForceAlgorithm::PARTICLE_MESH);
        backend->initialize(initial.size(), base_params);
        
        auto particles = initial;
        backend->setParticles(particles);
        
        auto start = std::chrono::high_resolution_clock::now();
        for (int step = 0; step < 1000; step++) {
            backend->step(base_params.dt);
        }
        auto end = std::chrono::high_resolution_clock::now();
        double time_ms = std::chrono::duration<double, std::milli>(end - start).count();
        
        backend->getParticles(particles);
        
        // Calculate error vs reference
        float error = 0;
        for (size_t j = 0; j < initial.size(); j++) {
            float dx = particles[j].pos.x - particles_ref[j].pos.x;
            float dy = particles[j].pos.y - particles_ref[j].pos.y;
            error += sqrt(dx*dx + dy*dy);
        }
        error /= initial.size();
        
        // Center of mass
        float cx = 0, cy = 0, total_mass = 0;
        for (const auto& p : particles) {
            cx += p.pos.x * p.mass;
            cy += p.pos.y * p.mass;
            total_mass += p.mass;
        }
        
        // Estimate memory usage
        size_t memory_mb = (grid_size * grid_size * sizeof(float) * 8) / (1024 * 1024);
        
        std::cout << std::setw(10) << grid_size;
        std::cout << std::scientific << std::setprecision(3);
        std::cout << std::setw(20) << cx/total_mass;
        std::cout << std::setw(20) << error;
        std::cout << std::fixed << std::setprecision(2);
        std::cout << std::setw(20) << time_ms;
        std::cout << std::setw(20) << memory_mb << "\n";
    }
    
    std::cout << "\n=== Testing Softening Convergence ===\n\n";
    std::cout << std::setw(12) << "softening" 
              << std::setw(20) << "BF Energy"
              << std::setw(20) << "BH Energy"
              << std::setw(20) << "PM Energy"
              << std::setw(20) << "BH vs BF %" << "\n";
    std::cout << std::string(92, '-') << "\n";
    
    for (float softening : softenings) {
        base_params.softening = softening;
        base_params.dt = 0.001f;
        base_params.grid_size = 256;
        base_params.theta = 0.5f;
        
        std::vector<double> energies(3);
        
        for (size_t i = 0; i < 3; i++) {
            auto backend = std::make_unique<SimpleBackend>();
            backend->setAlgorithm(i == 0 ? ForceAlgorithm::BRUTE_FORCE : 
                                 i == 1 ? ForceAlgorithm::BARNES_HUT : 
                                          ForceAlgorithm::PARTICLE_MESH);
            backend->initialize(initial.size(), base_params);
            
            auto particles = initial;
            backend->setParticles(particles);
            
            for (int step = 0; step < 1000; step++) {
                backend->step(base_params.dt);
            }
            
            backend->getParticles(particles);
            
            // Calculate total energy
            double ke = 0, pe = 0;
            for (const auto& p : particles) {
                ke += 0.5 * p.mass * (p.vel.x * p.vel.x + p.vel.y * p.vel.y);
            }
            for (size_t j = 0; j < particles.size(); j++) {
                for (size_t k = j+1; k < particles.size(); k++) {
                    float dx = particles[k].pos.x - particles[j].pos.x;
                    float dy = particles[k].pos.y - particles[j].pos.y;
                    float r = sqrt(dx*dx + dy*dy + softening*softening);
                    pe -= base_params.gravity_constant * particles[j].mass * particles[k].mass / r;
                }
            }
            energies[i] = ke + pe;
        }
        
        std::cout << std::scientific << std::setprecision(3);
        std::cout << std::setw(12) << softening;
        std::cout << std::setw(20) << energies[0];
        std::cout << std::setw(20) << energies[1];
        std::cout << std::setw(20) << energies[2];
        std::cout << std::fixed << std::setprecision(2);
        std::cout << std::setw(20) << (energies[1] - energies[0])/fabs(energies[0]) * 100 << "\n";
    }
    
    std::cout << "\n=== Conclusions ===\n";
    std::cout << "1. Smaller timesteps should make all algorithms converge\n";
    std::cout << "2. Smaller theta (Barnes-Hut) should approach Brute Force\n";
    std::cout << "3. Larger grids (PM) should improve accuracy\n";
    std::cout << "4. Softening affects close encounters differently\n";
    
    return 0;
}
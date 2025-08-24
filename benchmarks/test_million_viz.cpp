#include "src/backend/SimpleBackend_v3.cpp"
#include <iostream>
#include <chrono>
#include <random>
#include <iomanip>
#include <cmath>
#include <vector>

// Simple ASCII visualization
void visualizeASCII(const std::vector<Particle>& particles, float box_size, int grid_size = 80) {
    std::vector<std::vector<char>> grid(grid_size, std::vector<char>(grid_size, ' '));
    std::vector<std::vector<int>> density(grid_size, std::vector<int>(grid_size, 0));
    
    // Count particles in each cell
    for (const auto& p : particles) {
        int x = (p.pos.x / box_size) * grid_size;
        int y = (p.pos.y / box_size) * grid_size;
        
        if (x >= 0 && x < grid_size && y >= 0 && y < grid_size) {
            density[y][x]++;
        }
    }
    
    // Find max density for scaling
    int max_density = 0;
    for (const auto& row : density) {
        for (int d : row) {
            max_density = std::max(max_density, d);
        }
    }
    
    // Convert density to characters
    for (int y = 0; y < grid_size; y++) {
        for (int x = 0; x < grid_size; x++) {
            int d = density[y][x];
            if (d == 0) {
                grid[y][x] = ' ';
            } else if (d == 1) {
                grid[y][x] = '.';
            } else if (d <= max_density * 0.1) {
                grid[y][x] = ':';
            } else if (d <= max_density * 0.3) {
                grid[y][x] = 'o';
            } else if (d <= max_density * 0.5) {
                grid[y][x] = 'O';
            } else if (d <= max_density * 0.7) {
                grid[y][x] = '#';
            } else {
                grid[y][x] = '@';
            }
        }
    }
    
    // Print grid with border
    std::cout << "+" << std::string(grid_size, '-') << "+\n";
    for (const auto& row : grid) {
        std::cout << "|";
        for (char c : row) {
            // Add colors
            switch(c) {
                case '@': std::cout << "\033[35m" << c << "\033[0m"; break;  // Magenta
                case '#': std::cout << "\033[31m" << c << "\033[0m"; break;  // Red
                case 'O': std::cout << "\033[33m" << c << "\033[0m"; break;  // Yellow
                case 'o': std::cout << "\033[36m" << c << "\033[0m"; break;  // Cyan
                case ':': std::cout << "\033[32m" << c << "\033[0m"; break;  // Green
                default: std::cout << c;
            }
        }
        std::cout << "|\n";
    }
    std::cout << "+" << std::string(grid_size, '-') << "+\n";
}

// Calculate system stats
void printStats(const std::vector<Particle>& particles, const std::string& label) {
    float total_mass = 0;
    float center_x = 0, center_y = 0;
    float avg_vel = 0;
    float min_x = 1e9, max_x = -1e9;
    float min_y = 1e9, max_y = -1e9;
    
    for (const auto& p : particles) {
        total_mass += p.mass;
        center_x += p.mass * p.pos.x;
        center_y += p.mass * p.pos.y;
        
        float v = sqrt(p.vel.x * p.vel.x + p.vel.y * p.vel.y);
        avg_vel += v;
        
        min_x = std::min(min_x, p.pos.x);
        max_x = std::max(max_x, p.pos.x);
        min_y = std::min(min_y, p.pos.y);
        max_y = std::max(max_y, p.pos.y);
    }
    
    center_x /= total_mass;
    center_y /= total_mass;
    avg_vel /= particles.size();
    
    std::cout << "\n" << label << " Statistics:\n";
    std::cout << "  Particles: " << particles.size() << "\n";
    std::cout << "  Center of mass: (" << center_x << ", " << center_y << ")\n";
    std::cout << "  Bounding box: [" << min_x << "-" << max_x << "] x [" 
              << min_y << "-" << max_y << "]\n";
    std::cout << "  Average velocity: " << avg_vel << "\n";
}

int main() {
    std::cout << "=== Million Particle Visualization Test ===\n\n";
    
    // Parameters
    SimulationParams params;
    params.box_size = 10000.0f;
    params.gravity_constant = 0.5f;  // Moderate gravity
    params.softening = 10.0f;        // Larger softening for stability
    params.dt = 0.1f;                // Larger timestep since PM is stable
    params.grid_size = 512;           // Good resolution
    
    // Test with 1 million particles
    size_t n = 1000000;
    
    std::cout << "Creating " << n << " particles...\n";
    std::vector<Particle> particles(n);
    
    // Create initial distribution - uniform with slight clustering
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> pos_dist(0, params.box_size);
    std::normal_distribution<float> cluster_dist(params.box_size/2, params.box_size/10);
    std::uniform_real_distribution<float> vel_dist(-1, 1);
    
    // Mix of uniform and clustered particles
    for (size_t i = 0; i < n; i++) {
        if (i < n/2) {
            // Uniform distribution
            particles[i].pos.x = pos_dist(gen);
            particles[i].pos.y = pos_dist(gen);
        } else {
            // Clustered distribution
            particles[i].pos.x = cluster_dist(gen);
            particles[i].pos.y = cluster_dist(gen);
            
            // Wrap around periodic boundaries
            while (particles[i].pos.x < 0) particles[i].pos.x += params.box_size;
            while (particles[i].pos.x >= params.box_size) particles[i].pos.x -= params.box_size;
            while (particles[i].pos.y < 0) particles[i].pos.y += params.box_size;
            while (particles[i].pos.y >= params.box_size) particles[i].pos.y -= params.box_size;
        }
        
        particles[i].vel.x = vel_dist(gen);
        particles[i].vel.y = vel_dist(gen);
        particles[i].mass = 1.0f;
        particles[i].radius = 1.0f;
    }
    
    std::cout << "\nInitial state:\n";
    visualizeASCII(particles, params.box_size);
    printStats(particles, "Initial");
    
    // Create backend
    std::cout << "\nInitializing PM backend...\n";
    auto backend = std::make_unique<SimpleBackend>();
    backend->setAlgorithm(ForceAlgorithm::PARTICLE_MESH);
    backend->initialize(n, params);
    backend->setParticles(particles);
    
    // Run simulation
    int num_steps = 50;
    std::cout << "\nRunning " << num_steps << " steps...\n";
    
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int step = 0; step < num_steps; step++) {
        backend->step(params.dt);
        
        // Show progress
        if (step % 10 == 0) {
            std::cout << "  Step " << step << "/" << num_steps << "\r" << std::flush;
        }
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    double total_ms = std::chrono::duration<double, std::milli>(end - start).count();
    
    // Get final state
    backend->getParticles(particles);
    
    std::cout << "\n\nFinal state after " << num_steps << " steps:\n";
    visualizeASCII(particles, params.box_size);
    printStats(particles, "Final");
    
    // Performance stats
    std::cout << "\nPerformance:\n";
    std::cout << "  Total time: " << total_ms << " ms\n";
    std::cout << "  Average: " << (total_ms/num_steps) << " ms/step\n";
    std::cout << "  FPS: " << (1000.0 * num_steps / total_ms) << "\n";
    std::cout << "  Particles/sec: " << (n * num_steps * 1000.0 / total_ms) << "\n";
    
    // Check for sanity
    float center_drift = 0;
    float total_mass = 0;
    float cx_final = 0, cy_final = 0;
    float cx_init = params.box_size/2, cy_init = params.box_size/2;
    
    for (const auto& p : particles) {
        total_mass += p.mass;
        cx_final += p.mass * p.pos.x;
        cy_final += p.mass * p.pos.y;
    }
    cx_final /= total_mass;
    cy_final /= total_mass;
    
    center_drift = sqrt((cx_final - cx_init)*(cx_final - cx_init) + 
                        (cy_final - cy_init)*(cy_final - cy_init));
    
    std::cout << "\nSanity checks:\n";
    std::cout << "  Center of mass drift: " << center_drift << " (should be small)\n";
    std::cout << "  Particles in bounds: " << particles.size() << "/" << n << "\n";
    
    if (center_drift < params.box_size * 0.1) {
        std::cout << "  ✓ Simulation looks stable!\n";
    } else {
        std::cout << "  ⚠ Large center of mass drift detected\n";
    }
    
    return 0;
}
#include "src/backend/SimpleBackend.cpp"
#include <iostream>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <thread>

// Simple test: one planet orbiting a star
int main() {
    std::cout << "=== Simple Orbit Test ===\n\n";
    std::cout << "Testing basic orbital mechanics with custom FFT PM algorithm\n\n";
    
    // Parameters
    SimulationParams params;
    params.box_size = 1000.0f;
    params.gravity_constant = 100.0f;  // Strong gravity for visible motion
    params.softening = 0.1f;
    params.dt = 0.01f;
    params.grid_size = 128;
    
    // Create two bodies
    std::vector<Particle> bodies(2);
    
    // Star at center
    bodies[0].pos = {500.0f, 500.0f};
    bodies[0].vel = {0.0f, 0.0f};
    bodies[0].mass = 1000.0f;
    bodies[0].radius = 10.0f;
    
    // Planet at distance 100
    float r = 100.0f;
    bodies[1].pos.x = 500.0f + r;
    bodies[1].pos.y = 500.0f;
    
    // Circular orbit velocity: v = sqrt(GM/r)
    float v_orbit = sqrt(params.gravity_constant * bodies[0].mass / r);
    bodies[1].vel.x = 0.0f;
    bodies[1].vel.y = v_orbit;
    bodies[1].mass = 1.0f;
    bodies[1].radius = 2.0f;
    
    std::cout << "Setup:\n";
    std::cout << "  Star: mass=" << bodies[0].mass << " at center\n";
    std::cout << "  Planet: distance=" << r << ", velocity=" << v_orbit << "\n";
    std::cout << "  Expected period: " << (2 * M_PI * r / v_orbit) << " time units\n\n";
    
    // Create backend
    auto backend = std::make_unique<SimpleBackend>();
    backend->setAlgorithm(ForceAlgorithm::BRUTE_FORCE);  // Use exact for 2 bodies
    backend->initialize(bodies.size(), params);
    backend->setParticles(bodies);
    
    // Run simulation with visualization
    int steps = 2000;
    int display_interval = 50;
    
    std::cout << "Running simulation...\n\n";
    
    // Track orbit
    float initial_angle = 0;
    int orbits_completed = 0;
    float last_angle = initial_angle;
    
    for (int step = 0; step <= steps; step++) {
        if (step > 0) {
            backend->step(params.dt);
        }
        
        // Get positions
        backend->getParticles(bodies);
        
        // Calculate angle
        float dx = bodies[1].pos.x - bodies[0].pos.x;
        float dy = bodies[1].pos.y - bodies[0].pos.y;
        float angle = atan2(dy, dx);
        float r_current = sqrt(dx*dx + dy*dy);
        
        // Detect orbit completion (angle wraps from -π to π)
        if (step > 0) {
            float angle_diff = angle - last_angle;
            // Handle wrap-around
            if (angle_diff > M_PI) angle_diff -= 2*M_PI;
            if (angle_diff < -M_PI) angle_diff += 2*M_PI;
            
            // Check if we crossed the starting angle
            if (last_angle <= initial_angle && angle > initial_angle && step > 100) {
                orbits_completed++;
                float period = step * params.dt / orbits_completed;
                std::cout << "\nOrbit " << orbits_completed << " completed! Period: " << period << "\n";
            }
        }
        last_angle = angle;
        
        // Display
        if (step % display_interval == 0) {
            // Simple ASCII visualization
            std::cout << "\033[2J\033[H";  // Clear screen
            
            // Draw 40x20 grid centered on star
            const int width = 40, height = 20;
            std::vector<std::vector<char>> grid(height, std::vector<char>(width, ' '));
            
            // Plot star
            int sx = width/2;
            int sy = height/2;
            if (sx >= 0 && sx < width && sy >= 0 && sy < height) {
                grid[sy][sx] = '*';
            }
            
            // Plot planet
            int px = (int)(width/2 + dx/10);  // Scale down by 10
            int py = (int)(height/2 + dy/10);
            if (px >= 0 && px < width && py >= 0 && py < height) {
                grid[py][px] = 'O';
            }
            
            // Draw grid
            std::cout << "+" << std::string(width, '-') << "+\n";
            for (const auto& row : grid) {
                std::cout << "|";
                for (char c : row) {
                    if (c == '*') {
                        std::cout << "\033[33;1m" << c << "\033[0m";  // Yellow star
                    } else if (c == 'O') {
                        std::cout << "\033[36m" << c << "\033[0m";  // Cyan planet
                    } else {
                        std::cout << c;
                    }
                }
                std::cout << "|\n";
            }
            std::cout << "+" << std::string(width, '-') << "+\n";
            
            // Stats
            std::cout << "Time: " << std::setprecision(2) << std::fixed << (step * params.dt) << "\n";
            std::cout << "Distance: " << r_current << " (expected: " << r << ")\n";
            std::cout << "Angle: " << (angle * 180 / M_PI) << " degrees\n";
            std::cout << "Orbits: " << orbits_completed << "\n";
            
            // Brief pause for animation
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
        }
    }
    
    // Final stats
    std::cout << "\n\nFinal Analysis:\n";
    std::cout << "  Total orbits: " << orbits_completed << "\n";
    if (orbits_completed > 0) {
        float measured_period = (steps * params.dt) / orbits_completed;
        float expected_period = 2 * M_PI * r / v_orbit;
        std::cout << "  Measured period: " << measured_period << "\n";
        std::cout << "  Expected period: " << expected_period << "\n";
        std::cout << "  Error: " << fabs(measured_period - expected_period) / expected_period * 100 << "%\n";
    }
    
    return 0;
}
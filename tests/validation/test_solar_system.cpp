#include "src/backend/SimpleBackend.cpp"
#include <iostream>
#include <chrono>
#include <random>
#include <iomanip>
#include <cmath>
#include <vector>

// Simple ASCII visualization focused on a region
void visualizeSolarSystem(const std::vector<Particle>& particles, float center_x, float center_y, float view_size) {
    int grid_size = 80;
    std::vector<std::vector<char>> grid(grid_size, std::vector<char>(grid_size, ' '));
    
    for (size_t i = 0; i < particles.size(); i++) {
        const auto& p = particles[i];
        
        // Transform to view coordinates
        float x = (p.pos.x - center_x + view_size/2) / view_size * grid_size;
        float y = (p.pos.y - center_y + view_size/2) / view_size * grid_size;
        
        int ix = (int)x;
        int iy = (int)y;
        
        if (ix >= 0 && ix < grid_size && iy >= 0 && iy < grid_size) {
            // Different symbols based on mass
            if (i == 0) {
                grid[iy][ix] = '*';  // Sun
            } else if (i <= 4) {
                grid[iy][ix] = 'o';  // Inner planets
            } else if (i <= 8) {
                grid[iy][ix] = 'O';  // Outer planets
            } else {
                // Asteroids/debris
                if (grid[iy][ix] == ' ') grid[iy][ix] = '.';
                else if (grid[iy][ix] == '.') grid[iy][ix] = ':';
            }
        }
    }
    
    // Print grid
    std::cout << "+" << std::string(grid_size, '-') << "+\n";
    for (const auto& row : grid) {
        std::cout << "|";
        for (char c : row) {
            switch(c) {
                case '*': std::cout << "\033[33;1m" << c << "\033[0m"; break;  // Bright yellow sun
                case 'O': std::cout << "\033[36m" << c << "\033[0m"; break;     // Cyan outer planets
                case 'o': std::cout << "\033[35m" << c << "\033[0m"; break;     // Magenta inner planets
                case ':': std::cout << "\033[32m" << c << "\033[0m"; break;     // Green debris
                case '.': std::cout << "\033[32m" << c << "\033[0m"; break;     // Green debris
                default: std::cout << c;
            }
        }
        std::cout << "|\n";
    }
    std::cout << "+" << std::string(grid_size, '-') << "+\n";
}

// Create a solar system
std::vector<Particle> createSolarSystem(float sun_mass, float box_size) {
    std::vector<Particle> particles;
    
    // Sun at center
    Particle sun;
    sun.pos = {box_size/2, box_size/2};
    sun.vel = {0, 0};
    sun.mass = sun_mass;
    sun.radius = 10.0f;
    particles.push_back(sun);
    
    // Inner planets (Mercury, Venus, Earth, Mars)
    float inner_orbits[] = {40, 70, 100, 150};
    float inner_masses[] = {0.05f, 0.8f, 1.0f, 0.1f};
    
    for (int i = 0; i < 4; i++) {
        Particle planet;
        float r = inner_orbits[i];
        float angle = (rand() / (float)RAND_MAX) * 2 * M_PI;
        
        planet.pos.x = box_size/2 + r * cos(angle);
        planet.pos.y = box_size/2 + r * sin(angle);
        
        // Circular orbit velocity: v = sqrt(GM/r)
        float v_orbit = sqrt(sun_mass / r);
        planet.vel.x = -v_orbit * sin(angle);
        planet.vel.y = v_orbit * cos(angle);
        
        planet.mass = inner_masses[i];
        planet.radius = 2.0f;
        particles.push_back(planet);
    }
    
    // Outer planets (Jupiter, Saturn, Uranus, Neptune)
    float outer_orbits[] = {250, 400, 600, 800};
    float outer_masses[] = {20.0f, 10.0f, 5.0f, 5.0f};
    
    for (int i = 0; i < 4; i++) {
        Particle planet;
        float r = outer_orbits[i];
        float angle = (rand() / (float)RAND_MAX) * 2 * M_PI;
        
        planet.pos.x = box_size/2 + r * cos(angle);
        planet.pos.y = box_size/2 + r * sin(angle);
        
        float v_orbit = sqrt(sun_mass / r);
        planet.vel.x = -v_orbit * sin(angle);
        planet.vel.y = v_orbit * cos(angle);
        
        planet.mass = outer_masses[i];
        planet.radius = 5.0f;
        particles.push_back(planet);
    }
    
    // Asteroid belt between Mars and Jupiter
    for (int i = 0; i < 200; i++) {
        Particle asteroid;
        float r = 180 + (rand() / (float)RAND_MAX) * 50;  // 180-230 AU
        float angle = (rand() / (float)RAND_MAX) * 2 * M_PI;
        
        asteroid.pos.x = box_size/2 + r * cos(angle);
        asteroid.pos.y = box_size/2 + r * sin(angle);
        
        float v_orbit = sqrt(sun_mass / r) * (0.9f + (rand() / (float)RAND_MAX) * 0.2f);
        asteroid.vel.x = -v_orbit * sin(angle);
        asteroid.vel.y = v_orbit * cos(angle);
        
        asteroid.mass = 0.001f;
        asteroid.radius = 0.5f;
        particles.push_back(asteroid);
    }
    
    return particles;
}

void printOrbitalInfo(const std::vector<Particle>& particles, float sun_x, float sun_y) {
    std::cout << "\nOrbital Information:\n";
    std::cout << std::setw(10) << "Body" << std::setw(12) << "Distance" 
              << std::setw(12) << "Velocity" << std::setw(15) << "Orbital V" << "\n";
    std::cout << std::string(50, '-') << "\n";
    
    const char* names[] = {"Sun", "Mercury", "Venus", "Earth", "Mars", 
                           "Jupiter", "Saturn", "Uranus", "Neptune"};
    
    for (size_t i = 0; i < std::min(size_t(9), particles.size()); i++) {
        float dx = particles[i].pos.x - sun_x;
        float dy = particles[i].pos.y - sun_y;
        float r = sqrt(dx*dx + dy*dy);
        float v = sqrt(particles[i].vel.x * particles[i].vel.x + 
                      particles[i].vel.y * particles[i].vel.y);
        
        // Expected orbital velocity for circular orbit
        float v_expected = (i == 0) ? 0 : sqrt(particles[0].mass / r);
        
        std::cout << std::setw(10) << names[i] 
                  << std::setw(12) << std::fixed << std::setprecision(1) << r
                  << std::setw(12) << std::fixed << std::setprecision(2) << v;
        if (i > 0) {
            std::cout << std::setw(15) << std::fixed << std::setprecision(2) 
                      << v_expected << " (expected)";
        }
        std::cout << "\n";
    }
}

int main() {
    std::cout << "=== Solar System Orbital Test ===\n\n";
    std::cout << "Creating a Sol-like system to test gravitational dynamics\n\n";
    
    // Parameters for solar system
    SimulationParams params;
    params.box_size = 2000.0f;     // Large box for orbits
    params.gravity_constant = 100.0f;  // Scaled for reasonable orbit speeds
    params.softening = 1.0f;       // Small softening
    params.dt = 0.01f;              // Small timestep for accuracy
    params.grid_size = 256;         // Good resolution
    
    float sun_mass = 1000.0f;      // Massive sun
    
    // Create solar system
    auto particles = createSolarSystem(sun_mass, params.box_size);
    std::cout << "Created solar system with " << particles.size() << " bodies\n";
    std::cout << "  - 1 Sun (mass=" << sun_mass << ")\n";
    std::cout << "  - 4 inner planets\n";
    std::cout << "  - 4 outer planets\n";
    std::cout << "  - " << (particles.size() - 9) << " asteroids\n\n";
    
    // Print initial state
    std::cout << "Initial configuration:\n";
    visualizeSolarSystem(particles, params.box_size/2, params.box_size/2, 2000);
    printOrbitalInfo(particles, params.box_size/2, params.box_size/2);
    
    // Create backend - use Barnes-Hut for better accuracy with small N
    auto backend = std::make_unique<SimpleBackend>();
    backend->setAlgorithm(ForceAlgorithm::BARNES_HUT);  // More accurate for small systems
    backend->initialize(particles.size(), params);
    backend->setParticles(particles);
    
    // Run simulation with intermediate outputs
    int total_steps = 10000;
    int output_interval = 2000;
    
    std::cout << "\n\nRunning simulation for " << total_steps << " steps...\n";
    std::cout << "(Simulating " << total_steps * params.dt << " time units)\n\n";
    
    for (int step = 0; step < total_steps; step++) {
        backend->step(params.dt);
        
        // Show intermediate results
        if ((step + 1) % output_interval == 0 || step == total_steps - 1) {
            backend->getParticles(particles);
            
            std::cout << "\n" << std::string(80, '=') << "\n";
            std::cout << "After " << (step + 1) << " steps (t=" 
                      << (step + 1) * params.dt << "):\n";
            std::cout << std::string(80, '-') << "\n";
            
            // Zoom levels for different views
            if (step < total_steps/3) {
                // Full system view
                visualizeSolarSystem(particles, params.box_size/2, params.box_size/2, 2000);
            } else if (step < 2*total_steps/3) {
                // Inner system view
                std::cout << "[Zoomed to inner system]\n";
                visualizeSolarSystem(particles, params.box_size/2, params.box_size/2, 500);
            } else {
                // Full system again
                std::cout << "[Full system view]\n";
                visualizeSolarSystem(particles, params.box_size/2, params.box_size/2, 2000);
            }
            
            printOrbitalInfo(particles, params.box_size/2, params.box_size/2);
            
            // Check for stability
            float sun_drift = sqrt(particles[0].pos.x * particles[0].pos.x + 
                                 particles[0].pos.y * particles[0].pos.y);
            if (sun_drift > 10) {
                std::cout << "\n⚠ Warning: Sun has drifted by " << sun_drift << " units\n";
            }
            
            // Check Earth's orbit (index 3)
            if (particles.size() > 3) {
                float dx = particles[3].pos.x - particles[0].pos.x;
                float dy = particles[3].pos.y - particles[0].pos.y;
                float r = sqrt(dx*dx + dy*dy);
                float expected_period = 2 * M_PI * sqrt(r*r*r / (params.gravity_constant * sun_mass));
                std::cout << "\nEarth orbital radius: " << r 
                          << " (expected period: " << expected_period << " time units)\n";
            }
        }
        
        // Progress indicator
        if (step % 100 == 0) {
            std::cout << "\rProgress: " << (100 * step / total_steps) << "%" << std::flush;
        }
    }
    
    // Final analysis
    std::cout << "\n\n" << std::string(80, '=') << "\n";
    std::cout << "FINAL ANALYSIS\n";
    std::cout << std::string(80, '=') << "\n";
    
    // Count surviving planets
    int planets_in_orbit = 0;
    for (size_t i = 1; i <= 8 && i < particles.size(); i++) {
        float dx = particles[i].pos.x - particles[0].pos.x;
        float dy = particles[i].pos.y - particles[0].pos.y;
        float r = sqrt(dx*dx + dy*dy);
        if (r > 10 && r < 1500) {
            planets_in_orbit++;
        }
    }
    
    std::cout << "Planets still in orbit: " << planets_in_orbit << "/8\n";
    
    if (planets_in_orbit >= 6) {
        std::cout << "✓ System is stable!\n";
    } else {
        std::cout << "✗ System has destabilized\n";
    }
    
    return 0;
}
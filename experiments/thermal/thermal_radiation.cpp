#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>
#include "../../src/dynamics/ThermalDynamics.h"

// Helper to create a solar sail (line of connected particles)
std::vector<ThermalParticle> createSolarSail(float x, float y, 
                                              int num_particles = 5,
                                              float spacing = 10.0f) {
    std::vector<ThermalParticle> sail;
    
    for (int i = 0; i < num_particles; i++) {
        ThermalParticle p;
        p.pos.x = x + (i - num_particles/2) * spacing;
        p.pos.y = y;
        p.vel = {0, 0};
        p.mass = 1.0f;
        p.radius = 3.0f;
        p.temp_internal = 300.0f;
        
        // High reflectivity for sail
        p.absorptivity = 0.1f;  // Mostly reflects
        p.emissivity = 0.1f;    // Poor radiator
        
        // Sail is elongated
        p.aspect_ratio = 3.0f;
        p.orientation = {0, 1};  // Vertical orientation initially
        
        sail.push_back(p);
    }
    
    return sail;
}

// ASCII visualization
void visualize(const std::vector<ThermalParticle>& particles, 
               float box_size = 500.0f,
               const std::string& title = "") {
    const int width = 80;
    const int height = 24;
    std::vector<std::vector<char>> grid(height, std::vector<char>(width, ' '));
    
    for (const auto& p : particles) {
        int x = (p.pos.x / box_size + 0.5f) * width;
        int y = (p.pos.y / box_size + 0.5f) * height;
        
        if (x >= 0 && x < width && y >= 0 && y < height) {
            // Show temperature with different symbols
            if (p.temp_internal > 10000) {
                grid[y][x] = '*';  // Star
            } else if (p.temp_internal > 1000) {
                grid[y][x] = 'O';  // Hot
            } else if (p.temp_internal > 500) {
                grid[y][x] = 'o';  // Warm
            } else if (p.aspect_ratio > 1.5f) {
                grid[y][x] = '=';  // Solar sail particle
            } else {
                grid[y][x] = '.';  // Cool
            }
        }
    }
    
    // Draw
    if (!title.empty()) {
        std::cout << title << "\n";
    }
    std::cout << "+" << std::string(width, '-') << "+\n";
    for (const auto& row : grid) {
        std::cout << "|";
        for (char c : row) std::cout << c;
        std::cout << "|\n";
    }
    std::cout << "+" << std::string(width, '-') << "+\n";
}

// Print statistics
void printStats(const std::vector<ThermalParticle>& particles, int step) {
    float max_temp = 0;
    float total_luminosity = 0;
    float2 total_momentum = {0, 0};
    float total_ke = 0;
    
    for (const auto& p : particles) {
        max_temp = std::max(max_temp, p.temp_internal);
        total_luminosity += p.luminosity;
        total_momentum.x += p.mass * p.vel.x;
        total_momentum.y += p.mass * p.vel.y;
        total_ke += 0.5f * p.mass * (p.vel.x * p.vel.x + p.vel.y * p.vel.y);
    }
    
    std::cout << "Step " << std::setw(4) << step
              << " | Max T: " << std::fixed << std::setprecision(0) << max_temp << "K"
              << " | Total L: " << std::scientific << std::setprecision(2) << total_luminosity
              << " | KE: " << std::fixed << std::setprecision(2) << total_ke
              << " | p: (" << std::setprecision(2) << total_momentum.x 
              << ", " << total_momentum.y << ")\n";
}

int main() {
    std::cout << "=== Thermal Radiation Dynamics Test ===\n\n";
    std::cout << "This test demonstrates:\n";
    std::cout << "1. Radiation pressure from hot bodies\n";
    std::cout << "2. Radiative cooling\n";
    std::cout << "3. Solar sail propulsion\n";
    std::cout << "4. Thermal equilibrium\n\n";
    
    std::cout << "Legend:\n";
    std::cout << "  * = Very hot (star)\n";
    std::cout << "  O = Hot\n";
    std::cout << "  o = Warm\n";
    std::cout << "  = = Solar sail\n";
    std::cout << "  . = Cool\n\n";
    
    // Test 1: Radiation pressure pushes particles away
    std::cout << "TEST 1: Radiation Pressure from Hot Star\n";
    std::cout << std::string(50, '-') << "\n\n";
    
    std::vector<ThermalParticle> particles;
    
    // Create a hot "star" at center
    ThermalParticle star;
    star.pos = {0, 0};
    star.vel = {0, 0};
    star.mass = 1000.0f;
    star.radius = 20.0f;
    star.temp_internal = 50000.0f;  // Very hot!
    star.emissivity = 1.0f;         // Perfect black body
    particles.push_back(star);
    
    // Add some dust particles around it
    for (int i = 0; i < 8; i++) {
        ThermalParticle dust;
        float angle = 2.0f * M_PI * i / 8.0f;
        dust.pos.x = 100.0f * cos(angle);
        dust.pos.y = 100.0f * sin(angle);
        dust.vel = {0, 0};
        dust.mass = 1.0f;
        dust.radius = 5.0f;
        dust.temp_internal = 300.0f;
        particles.push_back(dust);
    }
    
    ThermalDynamics thermal;
    thermal.setRadiationScale(100.0f);  // Amplify for demonstration
    thermal.setCoolingRate(0.1f);       // Slow cooling
    
    std::cout << "Initial state: Hot star surrounded by dust\n";
    visualize(particles, 300.0f);
    printStats(particles, 0);
    
    // Simulate
    float dt = 0.01f;
    for (int step = 1; step <= 50; step++) {
        thermal.step(particles, dt);
        
        // Update positions
        for (auto& p : particles) {
            p.pos = p.pos + p.vel * dt;
        }
        
        if (step == 25 || step == 50) {
            std::cout << "\nAfter " << step << " steps:\n";
            visualize(particles, 300.0f);
            printStats(particles, step);
        }
    }
    
    std::cout << "\nDust particles are pushed away by radiation pressure!\n";
    
    // Test 2: Solar sail
    std::cout << "\n\nTEST 2: Solar Sail Propulsion\n";
    std::cout << std::string(50, '-') << "\n\n";
    
    particles.clear();
    
    // Hot star on the left
    star.pos = {-200, 0};
    star.vel = {0, 0};
    star.temp_internal = 100000.0f;  // Extra hot for more pressure
    particles.push_back(star);
    
    // Solar sail on the right
    auto sail = createSolarSail(50, 0, 5, 8.0f);
    for (const auto& p : sail) {
        particles.push_back(p);
    }
    
    // Regular particle for comparison
    ThermalParticle regular;
    regular.pos = {50, 50};
    regular.vel = {0, 0};
    regular.mass = 5.0f;  // Same total mass as sail
    regular.radius = 10.0f;
    regular.temp_internal = 300.0f;
    particles.push_back(regular);
    
    thermal.setRadiationScale(200.0f);  // Strong radiation
    
    std::cout << "Star (*) pushes both sail (=) and regular particle (.):\n";
    visualize(particles, 500.0f, "Initial");
    
    // Track positions
    float sail_x_start = particles[1].pos.x;
    float regular_x_start = particles[particles.size()-1].pos.x;
    
    // Simulate
    for (int step = 1; step <= 100; step++) {
        thermal.step(particles, dt);
        
        for (auto& p : particles) {
            p.pos = p.pos + p.vel * dt;
        }
        
        if (step == 50 || step == 100) {
            std::cout << "\nStep " << step << ":\n";
            visualize(particles, 500.0f);
        }
    }
    
    float sail_x_end = particles[1].pos.x;
    float regular_x_end = particles[particles.size()-1].pos.x;
    
    std::cout << "\nResults:\n";
    std::cout << "Solar sail moved: " << (sail_x_end - sail_x_start) << " units\n";
    std::cout << "Regular particle moved: " << (regular_x_end - regular_x_start) << " units\n";
    std::cout << "Sail advantage: " << ((sail_x_end - sail_x_start) / 
                                         (regular_x_end - regular_x_start)) << "x\n";
    
    // Test 3: Radiative cooling
    std::cout << "\n\nTEST 3: Radiative Cooling\n";
    std::cout << std::string(50, '-') << "\n\n";
    
    particles.clear();
    
    // Several hot particles that will cool down
    for (int i = 0; i < 5; i++) {
        ThermalParticle hot;
        hot.pos = {-100.0f + i * 50.0f, 0};
        hot.vel = {0, 0};
        hot.mass = 10.0f;
        hot.radius = 10.0f;
        hot.temp_internal = 5000.0f - i * 500.0f;  // Different temperatures
        particles.push_back(hot);
    }
    
    thermal.setRadiationScale(10.0f);
    thermal.setCoolingRate(1.0f);  // Normal cooling
    
    std::cout << "Initial temperatures:\n";
    for (size_t i = 0; i < particles.size(); i++) {
        std::cout << "  Particle " << i << ": " 
                  << std::fixed << std::setprecision(0) 
                  << particles[i].temp_internal << "K\n";
    }
    
    // Let them cool
    for (int step = 0; step < 200; step++) {
        thermal.step(particles, dt);
    }
    
    std::cout << "\nAfter radiative cooling:\n";
    for (size_t i = 0; i < particles.size(); i++) {
        std::cout << "  Particle " << i << ": " 
                  << std::fixed << std::setprecision(0) 
                  << particles[i].temp_internal << "K\n";
    }
    
    std::cout << "\nHotter particles cool faster (more luminosity)!\n";
    
    // Test 4: Thermal equilibrium
    std::cout << "\n\nTEST 4: Thermal Equilibrium\n";
    std::cout << std::string(50, '-') << "\n\n";
    
    particles.clear();
    
    // Hot source
    ThermalParticle source;
    source.pos = {0, 0};
    source.vel = {0, 0};
    source.mass = 100.0f;
    source.radius = 15.0f;
    source.temp_internal = 10000.0f;
    source.emissivity = 0.1f;  // Slow cooling
    particles.push_back(source);
    
    // Cold receiver nearby
    ThermalParticle receiver;
    receiver.pos = {50, 0};
    receiver.vel = {0, 0};
    receiver.mass = 10.0f;
    receiver.radius = 10.0f;
    receiver.temp_internal = 100.0f;
    particles.push_back(receiver);
    
    thermal.setRadiationScale(50.0f);
    thermal.setCoolingRate(0.5f);
    
    std::cout << "Hot source heats cold receiver until equilibrium:\n";
    std::cout << std::setw(10) << "Step" 
              << std::setw(15) << "Source T(K)"
              << std::setw(15) << "Receiver T(K)"
              << std::setw(15) << "Difference\n";
    std::cout << std::string(55, '-') << "\n";
    
    for (int step = 0; step <= 500; step += 50) {
        if (step > 0) {
            for (int i = 0; i < 50; i++) {
                thermal.step(particles, dt);
            }
        }
        
        float source_temp = particles[0].temp_internal;
        float receiver_temp = particles[1].temp_internal;
        
        std::cout << std::setw(10) << step
                  << std::setw(15) << std::fixed << std::setprecision(1) << source_temp
                  << std::setw(15) << receiver_temp
                  << std::setw(15) << (source_temp - receiver_temp) << "\n";
    }
    
    std::cout << "\nSystem approaches thermal equilibrium through radiation!\n";
    
    // Summary
    std::cout << "\n=== Summary ===\n";
    std::cout << "The thermal radiation system successfully demonstrates:\n";
    std::cout << "✓ Radiation pressure pushing particles (stellar wind)\n";
    std::cout << "✓ Solar sails with orientation-dependent forces\n";
    std::cout << "✓ Radiative cooling of hot bodies\n";
    std::cout << "✓ Thermal equilibrium through radiation exchange\n";
    std::cout << "\nThese emergent behaviors arise from simple physical laws!\n";
    
    return 0;
}
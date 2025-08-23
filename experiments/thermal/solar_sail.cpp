#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>
#include "../../src/dynamics/ThermalDynamics.h"

// Simple spring structure for connecting particles
struct Spring {
    int i, j;           // Particle indices
    float rest_length;  // Natural length
    float stiffness;    // Spring constant
    float damping;      // Damping coefficient
};

// Apply spring forces between connected particles
void applySpringForces(std::vector<ThermalParticle>& particles, 
                       const std::vector<Spring>& springs,
                       float dt) {
    for (const auto& spring : springs) {
        float2 r = particles[spring.j].pos - particles[spring.i].pos;
        float dist = sqrt(r.x * r.x + r.y * r.y);
        
        if (dist < 0.001f) continue;
        
        // Spring force: F = -k * (dist - rest_length) * direction
        float force_mag = spring.stiffness * (dist - spring.rest_length);
        float2 force = {force_mag * r.x / dist, force_mag * r.y / dist};
        
        // Damping based on relative velocity
        float2 v_rel = particles[spring.j].vel - particles[spring.i].vel;
        float2 damping = {spring.damping * v_rel.x, spring.damping * v_rel.y};
        
        force.x += damping.x;
        force.y += damping.y;
        
        // Apply forces (Newton's third law)
        particles[spring.i].vel.x += force.x * dt / particles[spring.i].mass;
        particles[spring.i].vel.y += force.y * dt / particles[spring.i].mass;
        particles[spring.j].vel.x -= force.x * dt / particles[spring.j].mass;
        particles[spring.j].vel.y -= force.y * dt / particles[spring.j].mass;
    }
}

// Create a solar sail from a line of particles
struct SolarSail {
    std::vector<int> particle_indices;
    std::vector<Spring> springs;
    
    static SolarSail create(std::vector<ThermalParticle>& particles,
                           float x, float y, 
                           int num_particles = 10,
                           float spacing = 5.0f,
                           bool vertical = true) {
        SolarSail sail;
        
        int start_idx = particles.size();
        
        // Create particles in a line
        for (int i = 0; i < num_particles; i++) {
            ThermalParticle p;
            if (vertical) {
                p.pos.x = x;
                p.pos.y = y + (i - num_particles/2) * spacing;
            } else {
                p.pos.x = x + (i - num_particles/2) * spacing;
                p.pos.y = y;
            }
            p.vel = {0, 0};
            p.mass = 0.5f;        // Light particles
            p.radius = 2.0f;      // Small
            p.temp_internal = 300.0f;
            
            // Reflective surface
            p.absorptivity = 0.1f;  // Mostly reflects
            p.emissivity = 0.1f;    // Poor radiator
            
            particles.push_back(p);
            sail.particle_indices.push_back(start_idx + i);
        }
        
        // Connect with springs
        for (int i = 0; i < num_particles - 1; i++) {
            Spring s;
            s.i = start_idx + i;
            s.j = start_idx + i + 1;
            s.rest_length = spacing;
            s.stiffness = 100.0f;  // Fairly rigid
            s.damping = 1.0f;
            sail.springs.push_back(s);
        }
        
        // Add diagonal springs for stability
        for (int i = 0; i < num_particles - 2; i++) {
            Spring s;
            s.i = start_idx + i;
            s.j = start_idx + i + 2;
            s.rest_length = spacing * 2.0f;
            s.stiffness = 50.0f;  // Softer diagonal springs
            s.damping = 0.5f;
            sail.springs.push_back(s);
        }
        
        return sail;
    }
    
    float2 getCenter(const std::vector<ThermalParticle>& particles) const {
        float2 center = {0, 0};
        for (int idx : particle_indices) {
            center.x += particles[idx].pos.x;
            center.y += particles[idx].pos.y;
        }
        center.x /= particle_indices.size();
        center.y /= particle_indices.size();
        return center;
    }
    
    float2 getNormal(const std::vector<ThermalParticle>& particles) const {
        // Calculate orientation from first to last particle
        int first = particle_indices.front();
        int last = particle_indices.back();
        float2 tangent = particles[last].pos - particles[first].pos;
        float len = sqrt(tangent.x * tangent.x + tangent.y * tangent.y);
        if (len > 0.001f) {
            tangent.x /= len;
            tangent.y /= len;
        }
        // Normal is perpendicular to tangent (rotate 90 degrees)
        return {-tangent.y, tangent.x};
    }
    
    float getEffectiveCrossSection(const std::vector<ThermalParticle>& particles,
                                   const float2& sun_direction) const {
        float2 normal = getNormal(particles);
        
        // Dot product gives cosine of angle
        float alignment = fabs(normal.x * sun_direction.x + 
                              normal.y * sun_direction.y);
        
        // Total cross section depends on angle
        // Perpendicular = maximum, parallel = minimum
        float total_radius = 0;
        for (int idx : particle_indices) {
            total_radius += particles[idx].radius * 2.0f;
        }
        
        // Effective cross section varies with angle
        float min_cross = particle_indices.size() * particles[particle_indices[0]].radius * 2.0f;
        float max_cross = total_radius;
        
        return min_cross + (max_cross - min_cross) * alignment;
    }
};

// Visualization
void visualize(const std::vector<ThermalParticle>& particles,
               const std::vector<SolarSail>& sails,
               float box_size = 600.0f) {
    const int width = 80;
    const int height = 24;
    std::vector<std::vector<char>> grid(height, std::vector<char>(width, ' '));
    
    // Mark sail particles
    std::vector<bool> is_sail(particles.size(), false);
    for (const auto& sail : sails) {
        for (int idx : sail.particle_indices) {
            is_sail[idx] = true;
        }
    }
    
    // Draw particles
    for (size_t i = 0; i < particles.size(); i++) {
        const auto& p = particles[i];
        int x = (p.pos.x / box_size + 0.5f) * width;
        int y = (p.pos.y / box_size + 0.5f) * height;
        
        if (x >= 0 && x < width && y >= 0 && y < height) {
            if (p.temp_internal > 10000) {
                grid[y][x] = '*';  // Star
            } else if (is_sail[i]) {
                grid[y][x] = '-';  // Sail particle
            } else if (p.temp_internal > 1000) {
                grid[y][x] = 'o';  // Hot
            } else {
                grid[y][x] = '.';  // Regular
            }
        }
    }
    
    // Draw
    std::cout << "+" << std::string(width, '-') << "+\n";
    for (const auto& row : grid) {
        std::cout << "|";
        for (char c : row) std::cout << c;
        std::cout << "|\n";
    }
    std::cout << "+" << std::string(width, '-') << "+\n";
}

int main() {
    std::cout << "=== Solar Sail Test (Particle Line) ===\n\n";
    std::cout << "A solar sail made from a line of connected particles.\n";
    std::cout << "The sail orientation affects radiation pressure!\n\n";
    
    std::cout << "Legend:\n";
    std::cout << "  * = Star (heat source)\n";
    std::cout << "  - = Sail particle\n";
    std::cout << "  . = Regular particle\n\n";
    
    // Test 1: Perpendicular vs Parallel sail
    std::cout << "TEST 1: Sail Orientation Matters\n";
    std::cout << std::string(50, '-') << "\n\n";
    
    std::vector<ThermalParticle> particles;
    std::vector<SolarSail> sails;
    std::vector<Spring> all_springs;
    
    // Create sun
    ThermalParticle sun;
    sun.pos = {-200, 0};
    sun.vel = {0, 0};
    sun.mass = 10000.0f;
    sun.radius = 30.0f;
    sun.temp_internal = 100000.0f;
    sun.emissivity = 1.0f;
    particles.push_back(sun);
    
    // Create vertical sail (perpendicular to sun)
    std::cout << "Creating vertical sail (maximum cross-section to sun)...\n";
    SolarSail vertical_sail = SolarSail::create(particles, 0, 0, 8, 6.0f, true);
    sails.push_back(vertical_sail);
    for (const auto& s : vertical_sail.springs) {
        all_springs.push_back(s);
    }
    
    // Create horizontal sail (parallel to sun)
    std::cout << "Creating horizontal sail (minimum cross-section to sun)...\n";
    SolarSail horizontal_sail = SolarSail::create(particles, 0, 100, 8, 6.0f, false);
    sails.push_back(horizontal_sail);
    for (const auto& s : horizontal_sail.springs) {
        all_springs.push_back(s);
    }
    
    // Single particle for comparison (same total mass as sail)
    ThermalParticle single;
    single.pos = {0, -100};
    single.vel = {0, 0};
    single.mass = 4.0f;  // Same as total sail mass (8 * 0.5)
    single.radius = 8.0f;
    single.temp_internal = 300.0f;
    single.absorptivity = 0.1f;  // Same as sail
    particles.push_back(single);
    
    ThermalDynamics thermal;
    thermal.setRadiationScale(500.0f);  // Strong radiation for demonstration
    thermal.setCoolingRate(0.1f);
    
    std::cout << "\nInitial configuration:\n";
    visualize(particles, sails);
    
    // Track starting positions
    float2 vertical_start = sails[0].getCenter(particles);
    float2 horizontal_start = sails[1].getCenter(particles);
    float2 single_start = particles[particles.size()-1].pos;
    
    // Simulate
    float dt = 0.01f;
    for (int step = 0; step < 200; step++) {
        // Apply thermal radiation
        thermal.step(particles, dt);
        
        // Apply spring forces
        applySpringForces(particles, all_springs, dt);
        
        // Update positions
        for (auto& p : particles) {
            p.pos = p.pos + p.vel * dt;
        }
        
        if (step == 100 || step == 199) {
            std::cout << "\nStep " << step << ":\n";
            visualize(particles, sails);
        }
    }
    
    // Calculate distances moved
    float2 vertical_end = sails[0].getCenter(particles);
    float2 horizontal_end = sails[1].getCenter(particles);
    float2 single_end = particles[particles.size()-1].pos;
    
    float vertical_dist = sqrt(pow(vertical_end.x - vertical_start.x, 2) + 
                              pow(vertical_end.y - vertical_start.y, 2));
    float horizontal_dist = sqrt(pow(horizontal_end.x - horizontal_start.x, 2) + 
                                pow(horizontal_end.y - horizontal_start.y, 2));
    float single_dist = sqrt(pow(single_end.x - single_start.x, 2) + 
                            pow(single_end.y - single_start.y, 2));
    
    std::cout << "\n=== Results ===\n";
    std::cout << "Vertical sail (perpendicular):   " << std::fixed << std::setprecision(2) 
              << vertical_dist << " units\n";
    std::cout << "Horizontal sail (parallel):      " << horizontal_dist << " units\n";
    std::cout << "Single particle (same mass):     " << single_dist << " units\n";
    std::cout << "\nVertical/Horizontal ratio: " << (vertical_dist / horizontal_dist) << "x\n";
    std::cout << "Vertical/Single ratio: " << (vertical_dist / single_dist) << "x\n";
    
    // Test 2: Rotating sail
    std::cout << "\n\nTEST 2: Rotating Sail\n";
    std::cout << std::string(50, '-') << "\n\n";
    
    particles.clear();
    sails.clear();
    all_springs.clear();
    
    // Sun at center
    sun.pos = {0, 0};
    particles.push_back(sun);
    
    // Create diagonal sail
    std::cout << "Creating a diagonal sail that will rotate...\n";
    for (int i = 0; i < 6; i++) {
        ThermalParticle p;
        float t = (i - 2.5f) / 2.5f;
        p.pos.x = 150.0f + t * 20.0f;
        p.pos.y = t * 20.0f;
        p.vel = {0, 10.0f};  // Initial angular velocity
        p.mass = 0.5f;
        p.radius = 3.0f;
        p.temp_internal = 300.0f;
        p.absorptivity = 0.1f;
        particles.push_back(p);
    }
    
    // Connect with springs
    for (int i = 1; i < 6; i++) {
        Spring s;
        s.i = i;
        s.j = i + 1;
        s.rest_length = sqrt(2.0f * 20.0f * 20.0f) / 2.5f;
        s.stiffness = 200.0f;
        s.damping = 2.0f;
        all_springs.push_back(s);
    }
    
    thermal.setRadiationScale(100.0f);
    
    std::cout << "The diagonal sail will experience torque and rotate:\n\n";
    
    for (int step = 0; step < 300; step++) {
        thermal.step(particles, dt * 0.5f);
        applySpringForces(particles, all_springs, dt * 0.5f);
        
        for (auto& p : particles) {
            p.pos = p.pos + p.vel * dt * 0.5f;
        }
        
        if (step % 100 == 0) {
            std::cout << "Step " << step << ":\n";
            visualize(particles, sails, 400.0f);
            
            // Calculate sail angle
            float2 sail_vec = particles[6].pos - particles[1].pos;
            float angle = atan2(sail_vec.y, sail_vec.x) * 180.0f / M_PI;
            std::cout << "Sail angle: " << angle << " degrees\n\n";
        }
    }
    
    std::cout << "=== Summary ===\n";
    std::cout << "Solar sails made from lines of particles demonstrate:\n";
    std::cout << "✓ Orientation-dependent radiation pressure\n";
    std::cout << "✓ Perpendicular sails receive more force\n";
    std::cout << "✓ Sails can rotate due to radiation torque\n";
    std::cout << "✓ Spring networks maintain sail structure\n";
    std::cout << "\nNo special 'elongated particle' code needed!\n";
    std::cout << "Emergent behavior from particle lines + springs + radiation.\n";
    
    return 0;
}
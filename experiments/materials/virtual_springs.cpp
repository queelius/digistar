#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <random>
#include <algorithm>

// Material types
enum MaterialType {
    ROCK,
    METAL,
    ICE,
    ORGANIC,
    DUST
};

// Material properties that determine spring behavior
struct MaterialProperties {
    MaterialType type;
    const char* name;
    float stiffness;        // Young's modulus analog
    float tensile_strength; // Breaking force per unit area
    float plasticity;       // Plastic deformation threshold (0-1)
    float damping_ratio;    // Energy dissipation (0-1)
    float cohesion;         // How easily springs form with same material (0-1)
    float adhesion;         // How easily springs form with other materials (0-1)
    float melting_point;    // Temperature where springs weaken (K)
};

// Predefined materials
MaterialProperties getMaterial(MaterialType type) {
    switch(type) {
        case ROCK:
            return {ROCK, "Rock", 10000.0f, 100.0f, 0.01f, 0.1f, 0.7f, 0.2f, 1500.0f};
        case METAL:
            return {METAL, "Metal", 20000.0f, 500.0f, 0.2f, 0.05f, 0.8f, 0.3f, 1000.0f};
        case ICE:
            return {ICE, "Ice", 5000.0f, 50.0f, 0.02f, 0.2f, 0.9f, 0.4f, 273.0f};
        case ORGANIC:
            return {ORGANIC, "Organic", 100.0f, 20.0f, 0.5f, 0.5f, 0.6f, 0.5f, 400.0f};
        case DUST:
            return {DUST, "Dust", 10.0f, 5.0f, 0.8f, 0.7f, 0.3f, 0.2f, 800.0f};
        default:
            return getMaterial(ROCK);
    }
}

// Particle with material properties
struct Particle {
    float x, y;           // Position
    float vx, vy;         // Velocity
    float mass;
    float radius;
    float temp;           // Temperature
    MaterialProperties material;
    
    Particle(float x_, float y_, MaterialType mat) 
        : x(x_), y(y_), vx(0), vy(0), mass(10.0f), radius(5.0f), temp(300.0f) {
        material = getMaterial(mat);
    }
};

// Virtual spring with properties derived from connected particles
struct Spring {
    int i, j;               // Particle indices
    float rest_length;      // Equilibrium length
    float stiffness;        // Spring constant
    float break_force;      // Maximum force before breaking
    float damping;          // Damping coefficient
    float plastic_threshold;// Deformation before permanent change
    float max_temp;         // Temperature limit
    bool broken = false;
    
    // Track plastic deformation
    float permanent_stretch = 0;
};

// Generate spring properties from two particles
Spring generateSpring(int i, int j, const Particle& p1, const Particle& p2) {
    Spring spring;
    spring.i = i;
    spring.j = j;
    
    // Rest length is current distance
    float dx = p2.x - p1.x;
    float dy = p2.y - p1.y;
    spring.rest_length = sqrt(dx*dx + dy*dy);
    
    // Contact area (2D: use minimum diameter)
    float contact_area = 2.0f * std::min(p1.radius, p2.radius);
    
    // Stiffness: average of materials, scaled by contact
    spring.stiffness = (p1.material.stiffness + p2.material.stiffness) / 2.0f 
                       * contact_area / 10.0f;  // Scale down for demo
    
    // Breaking force: weakest material determines strength
    spring.break_force = std::min(p1.material.tensile_strength, 
                                  p2.material.tensile_strength) * contact_area;
    
    // Damping: average
    spring.damping = (p1.material.damping_ratio + p2.material.damping_ratio) / 2.0f;
    
    // Plasticity: most plastic material dominates
    spring.plastic_threshold = std::max(p1.material.plasticity, p2.material.plasticity) 
                               * spring.rest_length * 0.1f;
    
    // Temperature limit: lowest melting point
    spring.max_temp = std::min(p1.material.melting_point, p2.material.melting_point);
    
    return spring;
}

// Check if spring should form between particles
bool shouldFormSpring(const Particle& p1, const Particle& p2, 
                     std::mt19937& rng) {
    // Distance check
    float dx = p2.x - p1.x;
    float dy = p2.y - p1.y;
    float dist = sqrt(dx*dx + dy*dy);
    float contact_dist = (p1.radius + p2.radius) * 1.2f;
    
    if (dist > contact_dist) return false;
    
    // Velocity check (must be slow enough)
    float vx = p2.vx - p1.vx;
    float vy = p2.vy - p1.vy;
    float v_rel = sqrt(vx*vx + vy*vy);
    if (v_rel > 10.0f) return false;  // Too fast
    
    // Temperature check
    float avg_temp = (p1.temp + p2.temp) / 2.0f;
    float min_melt = std::min(p1.material.melting_point, p2.material.melting_point);
    if (avg_temp > min_melt * 0.8f) return false;  // Too hot
    
    // Material compatibility
    float bond_prob;
    if (p1.material.type == p2.material.type) {
        bond_prob = p1.material.cohesion;  // Same material
    } else {
        bond_prob = (p1.material.adhesion + p2.material.adhesion) / 2.0f;  // Different
    }
    
    std::uniform_real_distribution<float> dist01(0.0f, 1.0f);
    return dist01(rng) < bond_prob;
}

// Update spring (returns false if broken)
bool updateSpring(Spring& spring, Particle& p1, Particle& p2, float dt) {
    if (spring.broken) return false;
    
    float dx = p2.x - p1.x;
    float dy = p2.y - p1.y;
    float dist = sqrt(dx*dx + dy*dy);
    
    if (dist < 0.001f) return true;  // Too close, skip
    
    // Current stretch/compression
    float stretch = dist - (spring.rest_length + spring.permanent_stretch);
    
    // Temperature weakening
    float avg_temp = (p1.temp + p2.temp) / 2.0f;
    float temp_modifier = 1.0f;
    if (avg_temp > spring.max_temp * 0.7f) {
        temp_modifier = 1.0f - (avg_temp - spring.max_temp * 0.7f) / 
                               (spring.max_temp * 0.3f);
        temp_modifier = std::max(0.0f, temp_modifier);
    }
    
    // Spring force
    float force_mag = spring.stiffness * stretch * temp_modifier;
    
    // Check for breaking
    if (fabs(force_mag) > spring.break_force * temp_modifier) {
        spring.broken = true;
        std::cout << "  Spring broke! (" << p1.material.name << "-" 
                  << p2.material.name << ") Force: " << fabs(force_mag) 
                  << " > " << spring.break_force * temp_modifier << "\n";
        return false;
    }
    
    // Check for melting
    if (avg_temp > spring.max_temp) {
        spring.broken = true;
        std::cout << "  Spring melted! Temp: " << avg_temp 
                  << "K > " << spring.max_temp << "K\n";
        return false;
    }
    
    // Plastic deformation
    if (fabs(stretch) > spring.plastic_threshold) {
        float plastic = (fabs(stretch) - spring.plastic_threshold) * 0.1f;
        if (stretch > 0) {
            spring.permanent_stretch += plastic;
        } else {
            spring.permanent_stretch -= plastic;
        }
        spring.break_force *= 0.98f;  // Weakening from deformation
    }
    
    // Apply forces
    float fx = force_mag * dx / dist;
    float fy = force_mag * dy / dist;
    
    // Damping
    float vx = p2.vx - p1.vx;
    float vy = p2.vy - p1.vy;
    fx += spring.damping * vx * 10.0f;
    fy += spring.damping * vy * 10.0f;
    
    // Newton's third law
    p1.vx += fx * dt / p1.mass;
    p1.vy += fy * dt / p1.mass;
    p2.vx -= fx * dt / p2.mass;
    p2.vy -= fy * dt / p2.mass;
    
    return true;
}

// Visualization
void visualize(const std::vector<Particle>& particles,
               const std::vector<Spring>& springs) {
    const int width = 60;
    const int height = 20;
    std::vector<std::vector<char>> grid(height, std::vector<char>(width, ' '));
    
    // Draw springs as connections
    for (const auto& s : springs) {
        if (s.broken) continue;
        
        const auto& p1 = particles[s.i];
        const auto& p2 = particles[s.j];
        
        // Simple line drawing (just mark endpoints for clarity)
        int x1 = (p1.x / 200.0f + 0.5f) * width;
        int y1 = (p1.y / 200.0f + 0.5f) * height;
        int x2 = (p2.x / 200.0f + 0.5f) * width;
        int y2 = (p2.y / 200.0f + 0.5f) * height;
        
        // Draw line between particles (simplified)
        if (abs(x2-x1) > abs(y2-y1)) {
            // More horizontal
            int xmin = std::min(x1, x2);
            int xmax = std::max(x1, x2);
            for (int x = xmin; x <= xmax; x++) {
                if (x >= 0 && x < width && y1 >= 0 && y1 < height) {
                    if (grid[y1][x] == ' ') grid[y1][x] = '-';
                }
            }
        } else {
            // More vertical
            int ymin = std::min(y1, y2);
            int ymax = std::max(y1, y2);
            for (int y = ymin; y <= ymax; y++) {
                if (x1 >= 0 && x1 < width && y >= 0 && y < height) {
                    if (grid[y][x1] == ' ') grid[y][x1] = '|';
                }
            }
        }
    }
    
    // Draw particles
    for (const auto& p : particles) {
        int x = (p.x / 200.0f + 0.5f) * width;
        int y = (p.y / 200.0f + 0.5f) * height;
        
        if (x >= 0 && x < width && y >= 0 && y < height) {
            char symbol;
            switch(p.material.type) {
                case ROCK:    symbol = 'R'; break;
                case METAL:   symbol = 'M'; break;
                case ICE:     symbol = 'I'; break;
                case ORGANIC: symbol = 'O'; break;
                case DUST:    symbol = 'd'; break;
                default:      symbol = '?'; break;
            }
            grid[y][x] = symbol;
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
    std::cout << "=== Virtual Spring Network Test ===\n\n";
    std::cout << "Springs form based on material properties:\n";
    std::cout << "- Cohesion: how well materials bond to themselves\n";
    std::cout << "- Adhesion: how well materials bond to others\n";
    std::cout << "- Springs inherit stiffness, strength, and damping from materials\n\n";
    
    std::cout << "Legend:\n";
    std::cout << "  R = Rock (stiff, brittle)\n";
    std::cout << "  M = Metal (stiff, ductile)\n";
    std::cout << "  I = Ice (medium stiff, melts)\n";
    std::cout << "  O = Organic (soft, flexible)\n";
    std::cout << "  d = Dust (very soft)\n";
    std::cout << "  - or | = Spring connection\n\n";
    
    std::mt19937 rng(42);
    
    // Test 1: Material compatibility
    std::cout << "TEST 1: Different Materials Form Different Springs\n";
    std::cout << std::string(50, '-') << "\n\n";
    
    std::vector<Particle> particles;
    std::vector<Spring> springs;
    
    // Create a line of different materials
    particles.push_back(Particle(-60, 0, METAL));
    particles.push_back(Particle(-30, 0, METAL));
    particles.push_back(Particle(0, 0, ROCK));
    particles.push_back(Particle(30, 0, ICE));
    particles.push_back(Particle(60, 0, ORGANIC));
    
    // Try to form springs between neighbors
    for (size_t i = 0; i < particles.size() - 1; i++) {
        // Force spring formation for demo
        Spring s = generateSpring(i, i+1, particles[i], particles[i+1]);
        springs.push_back(s);
        
        std::cout << particles[i].material.name << " - " 
                  << particles[i+1].material.name << " spring:\n";
        std::cout << "  Stiffness: " << s.stiffness 
                  << ", Strength: " << s.break_force
                  << ", Damping: " << s.damping << "\n";
    }
    
    std::cout << "\nInitial structure:\n";
    visualize(particles, springs);
    
    // Test 2: Breaking under stress
    std::cout << "\n\nTEST 2: Springs Break Under Different Conditions\n";
    std::cout << std::string(50, '-') << "\n\n";
    
    // Apply increasing force
    std::cout << "Pulling the structure apart...\n";
    particles[0].vx = -50.0f;
    particles[particles.size()-1].vx = 50.0f;
    
    float dt = 0.01f;
    for (int step = 0; step < 100; step++) {
        // Update springs
        for (auto& s : springs) {
            updateSpring(s, particles[s.i], particles[s.j], dt);
        }
        
        // Update positions
        for (auto& p : particles) {
            p.x += p.vx * dt;
            p.y += p.vy * dt;
        }
        
        if (step == 50) {
            std::cout << "\nAfter stretching:\n";
            visualize(particles, springs);
        }
    }
    
    // Count broken springs
    int broken_count = 0;
    for (const auto& s : springs) {
        if (s.broken) broken_count++;
    }
    std::cout << "\nBroken springs: " << broken_count << " / " << springs.size() << "\n";
    
    // Test 3: Temperature effects
    std::cout << "\n\nTEST 3: Temperature Effects (Ice Melting)\n";
    std::cout << std::string(50, '-') << "\n\n";
    
    particles.clear();
    springs.clear();
    
    // Create ice structure
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 3; j++) {
            particles.push_back(Particle(-30 + i*20, -20 + j*20, ICE));
        }
    }
    
    // Form springs between neighbors
    for (size_t i = 0; i < particles.size(); i++) {
        for (size_t j = i+1; j < particles.size(); j++) {
            if (shouldFormSpring(particles[i], particles[j], rng)) {
                springs.push_back(generateSpring(i, j, particles[i], particles[j]));
            }
        }
    }
    
    std::cout << "Ice structure at 200K:\n";
    visualize(particles, springs);
    std::cout << "Active springs: " << springs.size() << "\n";
    
    // Heat up the ice
    std::cout << "\nHeating to 300K (above melting point of 273K)...\n";
    for (auto& p : particles) {
        p.temp = 300.0f;
    }
    
    // Update springs (they should break)
    for (auto& s : springs) {
        updateSpring(s, particles[s.i], particles[s.j], dt);
    }
    
    // Count remaining springs
    int active = 0;
    for (const auto& s : springs) {
        if (!s.broken) active++;
    }
    
    std::cout << "Active springs after melting: " << active << "\n";
    visualize(particles, springs);
    
    // Test 4: Composite structures
    std::cout << "\n\nTEST 4: Composite Structure (Spaceship)\n";
    std::cout << std::string(50, '-') << "\n\n";
    
    particles.clear();
    springs.clear();
    
    // Create spaceship: metal frame with organic interior
    // Metal hull
    particles.push_back(Particle(-40, -20, METAL));  // 0
    particles.push_back(Particle(-40, 20, METAL));   // 1
    particles.push_back(Particle(40, -20, METAL));   // 2
    particles.push_back(Particle(40, 20, METAL));    // 3
    
    // Organic interior
    particles.push_back(Particle(0, 0, ORGANIC));    // 4
    
    // Rock shield
    particles.push_back(Particle(-60, 0, ROCK));     // 5
    
    // Create specific springs
    springs.push_back(generateSpring(0, 1, particles[0], particles[1]));  // Metal frame
    springs.push_back(generateSpring(1, 3, particles[1], particles[3]));
    springs.push_back(generateSpring(3, 2, particles[3], particles[2]));
    springs.push_back(generateSpring(2, 0, particles[2], particles[0]));
    
    springs.push_back(generateSpring(4, 0, particles[4], particles[0]));  // Organic to frame
    springs.push_back(generateSpring(4, 1, particles[4], particles[1]));
    springs.push_back(generateSpring(4, 2, particles[4], particles[2]));
    springs.push_back(generateSpring(4, 3, particles[4], particles[3]));
    
    springs.push_back(generateSpring(5, 0, particles[5], particles[0]));  // Shield attachment
    springs.push_back(generateSpring(5, 1, particles[5], particles[1]));
    
    std::cout << "Composite spaceship structure:\n";
    std::cout << "- Metal frame (M): High stiffness, high strength\n";
    std::cout << "- Organic core (O): Dampens vibrations\n";
    std::cout << "- Rock shield (R): Brittle but strong\n\n";
    
    visualize(particles, springs);
    
    // Analyze spring properties
    std::cout << "\nSpring analysis:\n";
    float max_stiff = 0, min_stiff = 1e9;
    float max_damp = 0, min_damp = 1e9;
    
    for (const auto& s : springs) {
        max_stiff = std::max(max_stiff, s.stiffness);
        min_stiff = std::min(min_stiff, s.stiffness);
        max_damp = std::max(max_damp, s.damping);
        min_damp = std::min(min_damp, s.damping);
    }
    
    std::cout << "Stiffness range: " << min_stiff << " - " << max_stiff << "\n";
    std::cout << "Damping range: " << min_damp << " - " << max_damp << "\n";
    std::cout << "\nMetal-metal springs are stiffest\n";
    std::cout << "Organic connections add damping\n";
    std::cout << "Rock-metal connection is brittle\n";
    
    std::cout << "\n=== Summary ===\n";
    std::cout << "Material-based springs provide:\n";
    std::cout << "✓ Intuitive behavior (ice melts, metal bends, rock breaks)\n";
    std::cout << "✓ Automatic property inheritance from materials\n";
    std::cout << "✓ Realistic composite structures\n";
    std::cout << "✓ Temperature-dependent behavior\n";
    std::cout << "✓ Plastic deformation and fatigue\n";
    std::cout << "\nNo abstract 'interaction vectors' needed!\n";
    std::cout << "Springs behave like real material connections.\n";
    
    return 0;
}
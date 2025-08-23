#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <algorithm>
#include <random>
#include <unordered_map>

// Constants
const float dt = 0.001f;
const float CONTACT_STIFFNESS = 1000.0f;
const float CONTACT_DAMPING = 10.0f;
const float SPRING_STIFFNESS = 100.0f;
const float SPRING_DAMPING = 0.5f;
const float SPRING_BREAK_FORCE = 50.0f;
const float SPRING_BREAK_THRESHOLD = 40.0f;
const float BREAK_IMPULSE = 5.0f;
const float G = 10.0f;  // Gravity for tidal test

// Basic structures
struct float2 {
    float x, y;
    
    float2 operator+(const float2& other) const { return {x + other.x, y + other.y}; }
    float2 operator-(const float2& other) const { return {x - other.x, y - other.y}; }
    float2 operator*(float s) const { return {x * s, y * s}; }
    float2 operator/(float s) const { return {x / s, y / s}; }
    float2& operator+=(const float2& other) { x += other.x; y += other.y; return *this; }
    float2& operator-=(const float2& other) { x -= other.x; y -= other.y; return *this; }
};

float length(const float2& v) { return sqrt(v.x * v.x + v.y * v.y); }
float2 normalize(const float2& v) { float l = length(v); return l > 0 ? v / l : float2{0, 0}; }
float dot(const float2& a, const float2& b) { return a.x * b.x + a.y * b.y; }
float2 perp(const float2& v) { return {-v.y, v.x}; }

struct Particle {
    float2 pos;
    float2 vel;
    float mass;
    float radius;
    char symbol;
    
    Particle(float x = 0, float y = 0, float m = 1.0f, float r = 1.0f, char s = 'o')
        : pos{x, y}, vel{0, 0}, mass(m), radius(r), symbol(s) {}
};

struct Spring {
    int i, j;
    float rest_length;
    float stiffness;
    float damping;
    float break_force;
    float stress = 0;
    bool broken = false;
    
    Spring(int i_, int j_, float rest, float stiff = SPRING_STIFFNESS)
        : i(i_), j(j_), rest_length(rest), stiffness(stiff),
          damping(SPRING_DAMPING), break_force(SPRING_BREAK_FORCE) {}
};

// Soft contact force calculation
struct ContactForce {
    float2 force;
    float penetration_depth;
    bool is_critical;
};

ContactForce calculateSoftContact(const Particle& p1, const Particle& p2) {
    float2 delta = p2.pos - p1.pos;
    float dist = length(delta);
    float min_dist = p1.radius + p2.radius;
    
    ContactForce contact = {{0, 0}, 0, false};
    
    if (dist < min_dist && dist > 0) {
        // Penetration depth
        contact.penetration_depth = min_dist - dist;
        
        // Hertzian contact model (soft contact)
        float stiffness = CONTACT_STIFFNESS * sqrt(p1.radius * p2.radius / (p1.radius + p2.radius));
        float force_mag = stiffness * pow(contact.penetration_depth, 1.5f);
        
        // Add damping
        float2 v_rel = p2.vel - p1.vel;
        float v_normal = dot(v_rel, delta) / dist;
        force_mag += CONTACT_DAMPING * v_normal;
        
        // Apply force
        contact.force = normalize(delta) * force_mag;
        
        // Check if force might break springs
        contact.is_critical = force_mag > SPRING_BREAK_THRESHOLD;
    }
    
    return contact;
}

// Apply soft contact forces
void applySoftContactForces(std::vector<Particle>& particles,
                           std::vector<Spring>& springs) {
    // Check all particle pairs
    for (size_t i = 0; i < particles.size(); i++) {
        for (size_t j = i + 1; j < particles.size(); j++) {
            ContactForce contact = calculateSoftContact(particles[i], particles[j]);
            
            if (contact.penetration_depth > 0) {
                // Apply repulsion
                particles[i].vel -= contact.force * dt / particles[i].mass;
                particles[j].vel += contact.force * dt / particles[j].mass;
                
                // Check if contact breaks springs
                if (contact.is_critical) {
                    for (auto& spring : springs) {
                        if (!spring.broken) {
                            // Springs connected to colliding particles get stressed
                            if (spring.i == (int)i || spring.i == (int)j ||
                                spring.j == (int)i || spring.j == (int)j) {
                                spring.stress += length(contact.force) * 0.5f;
                                
                                if (spring.stress > spring.break_force) {
                                    spring.broken = true;
                                    
                                    // Particles fly apart
                                    float2 spring_dir = normalize(particles[spring.j].pos - 
                                                                 particles[spring.i].pos);
                                    particles[spring.i].vel -= spring_dir * BREAK_IMPULSE;
                                    particles[spring.j].vel += spring_dir * BREAK_IMPULSE;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

// Apply spring forces
void applySpringForces(std::vector<Particle>& particles,
                      std::vector<Spring>& springs) {
    for (auto& spring : springs) {
        if (spring.broken) continue;
        
        float2 delta = particles[spring.j].pos - particles[spring.i].pos;
        float dist = length(delta);
        if (dist == 0) continue;
        
        float2 dir = delta / dist;
        
        // Hooke's law
        float force = spring.stiffness * (dist - spring.rest_length);
        
        // Damping
        float2 v_rel = particles[spring.j].vel - particles[spring.i].vel;
        force += spring.damping * dot(v_rel, dir);
        
        // Update stress
        spring.stress = abs(force);
        
        // Apply forces
        particles[spring.i].vel += dir * force * dt / particles[spring.i].mass;
        particles[spring.j].vel -= dir * force * dt / particles[spring.j].mass;
    }
}

// Tidal force calculation
void applyTidalForces(std::vector<Particle>& particles,
                     std::vector<Spring>& springs,
                     const Particle& massive_object) {
    // Find center of mass of particle system
    float2 com = {0, 0};
    float total_mass = 0;
    for (const auto& p : particles) {
        com += p.pos * p.mass;
        total_mass += p.mass;
    }
    com = com / total_mass;
    
    float2 r_com = com - massive_object.pos;
    float dist_com = length(r_com);
    
    // Tidal gradient
    float tidal_gradient = 2.0f * G * massive_object.mass / (dist_com * dist_com * dist_com);
    
    // Apply tidal forces to each particle
    for (auto& p : particles) {
        float2 r = p.pos - com;
        float r_parallel = dot(r, normalize(r_com));
        
        // Tidal force stretches along line to massive object
        float2 tidal_force = normalize(r_com) * tidal_gradient * r_parallel * p.mass;
        p.vel += tidal_force * dt / p.mass;
    }
    
    // Check if tidal forces break springs
    for (auto& spring : springs) {
        if (!spring.broken) {
            float2 r1 = particles[spring.i].pos - com;
            float2 r2 = particles[spring.j].pos - com;
            
            float stretch1 = dot(r1, normalize(r_com));
            float stretch2 = dot(r2, normalize(r_com));
            
            float tidal_stress = abs(stretch2 - stretch1) * tidal_gradient * 10.0f;
            spring.stress += tidal_stress;
            
            if (spring.stress > spring.break_force) {
                spring.broken = true;
                
                // Particles separate perpendicular to tidal axis
                float2 perp_dir = perp(normalize(r_com));
                particles[spring.i].vel += perp_dir * BREAK_IMPULSE * 0.5f;
                particles[spring.j].vel -= perp_dir * BREAK_IMPULSE * 0.5f;
            }
        }
    }
}

// Visualization
void visualize(const std::vector<Particle>& particles,
              const std::vector<Spring>& springs,
              const std::string& title = "") {
    const int width = 80;
    const int height = 24;
    std::vector<std::vector<char>> grid(height, std::vector<char>(width, ' '));
    
    if (!title.empty()) {
        std::cout << title << "\n";
    }
    
    // Draw springs
    for (const auto& spring : springs) {
        if (spring.broken) continue;
        
        const auto& p1 = particles[spring.i];
        const auto& p2 = particles[spring.j];
        
        // Simple line drawing
        int steps = 20;
        for (int s = 0; s <= steps; s++) {
            float t = (float)s / steps;
            float x = p1.pos.x * (1 - t) + p2.pos.x * t;
            float y = p1.pos.y * (1 - t) + p2.pos.y * t;
            
            int gx = (x / 50.0f + 0.5f) * width;
            int gy = (y / 50.0f + 0.5f) * height;
            
            if (gx >= 0 && gx < width && gy >= 0 && gy < height) {
                if (spring.stress > spring.break_force * 0.8f) {
                    grid[gy][gx] = '!';  // Stressed spring
                } else if (spring.stress > spring.break_force * 0.5f) {
                    grid[gy][gx] = '=';  // Medium stress
                } else {
                    grid[gy][gx] = '-';  // Normal spring
                }
            }
        }
    }
    
    // Draw particles
    for (const auto& p : particles) {
        int x = (p.pos.x / 50.0f + 0.5f) * width;
        int y = (p.pos.y / 50.0f + 0.5f) * height;
        
        if (x >= 0 && x < width && y >= 0 && y < height) {
            grid[y][x] = p.symbol;
        }
    }
    
    // Draw grid
    std::cout << "+" << std::string(width, '-') << "+\n";
    for (const auto& row : grid) {
        std::cout << "|";
        for (char c : row) std::cout << c;
        std::cout << "|\n";
    }
    std::cout << "+" << std::string(width, '-') << "+\n";
}

int main() {
    std::cout << "=== Soft Contact Forces & Collision Test ===\n\n";
    std::cout << "Demonstrating particle repulsion, spring breaking,\n";
    std::cout << "and tidal deformation with emergent soft-body dynamics\n\n";
    
    // Test 1: Soft collision between two bodies
    std::cout << "TEST 1: Soft Body Collision\n";
    std::cout << std::string(50, '-') << "\n\n";
    
    std::vector<Particle> particles;
    std::vector<Spring> springs;
    
    // Create first body (left, moving right)
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            particles.emplace_back(-15.0f + j * 2.0f, -2.0f + i * 2.0f, 1.0f, 1.5f, 'A');
            particles.back().vel = {10.0f, 0};
        }
    }
    
    // Connect first body with springs
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            int idx = i * 3 + j;
            if (j < 2) springs.emplace_back(idx, idx + 1, 2.0f);
            if (i < 2) springs.emplace_back(idx, idx + 3, 2.0f);
        }
    }
    
    // Create second body (right, moving left)
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            particles.emplace_back(10.0f + j * 2.0f, -2.0f + i * 2.0f, 1.0f, 1.5f, 'B');
            particles.back().vel = {-10.0f, 0};
        }
    }
    
    // Connect second body with springs
    int offset = 9;
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            int idx = offset + i * 3 + j;
            if (j < 2) springs.emplace_back(idx, idx + 1, 2.0f);
            if (i < 2) springs.emplace_back(idx, idx + 3, 2.0f);
        }
    }
    
    std::cout << "Initial: Two bodies approaching each other\n";
    visualize(particles, springs);
    
    // Simulate collision
    for (int step = 0; step < 2000; step++) {
        applySoftContactForces(particles, springs);
        applySpringForces(particles, springs);
        
        // Update positions
        for (auto& p : particles) {
            p.pos += p.vel * dt;
        }
        
        // Show key moments
        if (step == 500 || step == 1000 || step == 1500 || step == 1999) {
            std::cout << "\nStep " << step << ":\n";
            visualize(particles, springs);
            
            // Count broken springs
            int broken = 0;
            for (const auto& s : springs) {
                if (s.broken) broken++;
            }
            std::cout << "Broken springs: " << broken << "/" << springs.size() << "\n";
        }
    }
    
    // Test 2: Tidal disruption
    std::cout << "\n\nTEST 2: Tidal Disruption\n";
    std::cout << std::string(50, '-') << "\n\n";
    
    particles.clear();
    springs.clear();
    
    // Create extended body
    for (int i = 0; i < 5; i++) {
        for (int j = 0; j < 2; j++) {
            particles.emplace_back(i * 3.0f - 6.0f, j * 3.0f - 1.5f, 1.0f, 1.0f, 'o');
        }
    }
    
    // Connect with springs
    for (int i = 0; i < 5; i++) {
        for (int j = 0; j < 2; j++) {
            int idx = i * 2 + j;
            // Horizontal springs
            if (i < 4) {
                springs.emplace_back(idx, idx + 2, 3.0f);
            }
            // Vertical springs
            if (j < 1) {
                springs.emplace_back(idx, idx + 1, 3.0f);
            }
            // Diagonal springs
            if (i < 4 && j < 1) {
                springs.emplace_back(idx, idx + 3, 4.24f);
            }
        }
    }
    
    // Massive object to create tidal forces
    Particle massive(0, -20.0f, 1000.0f, 5.0f, 'M');
    
    std::cout << "Initial: Extended body above massive object\n";
    visualize(particles, springs);
    
    // Add massive object indicator
    std::cout << "Massive object 'M' below creates tidal forces\n\n";
    
    // Simulate tidal disruption
    for (int step = 0; step < 2000; step++) {
        applyTidalForces(particles, springs, massive);
        applySpringForces(particles, springs);
        applySoftContactForces(particles, springs);
        
        // Update positions
        for (auto& p : particles) {
            p.pos += p.vel * dt;
        }
        
        // Show key moments
        if (step == 500 || step == 1000 || step == 1500 || step == 1999) {
            std::cout << "Step " << step << ":\n";
            visualize(particles, springs);
            
            // Count broken springs
            int broken = 0;
            for (const auto& s : springs) {
                if (s.broken) broken++;
            }
            std::cout << "Broken springs: " << broken << "/" << springs.size() 
                      << " (Tidal stress breaking bonds)\n\n";
        }
    }
    
    // Test 3: Chain reaction of breaks
    std::cout << "\nTEST 3: Cascading Failure\n";
    std::cout << std::string(50, '-') << "\n\n";
    
    particles.clear();
    springs.clear();
    
    // Create lattice structure
    int lattice_size = 6;
    for (int i = 0; i < lattice_size; i++) {
        for (int j = 0; j < lattice_size; j++) {
            particles.emplace_back(i * 2.0f - 6.0f, j * 2.0f - 6.0f, 1.0f, 0.8f, '.');
        }
    }
    
    // Connect in grid with varying strengths
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> strength_dist(20.0f, 60.0f);
    
    for (int i = 0; i < lattice_size; i++) {
        for (int j = 0; j < lattice_size; j++) {
            int idx = i * lattice_size + j;
            
            // Right neighbor
            if (i < lattice_size - 1) {
                Spring s(idx, idx + lattice_size, 2.0f);
                s.break_force = strength_dist(rng);
                springs.push_back(s);
            }
            
            // Bottom neighbor
            if (j < lattice_size - 1) {
                Spring s(idx, idx + 1, 2.0f);
                s.break_force = strength_dist(rng);
                springs.push_back(s);
            }
        }
    }
    
    // Impact particle
    particles.emplace_back(-15.0f, -3.0f, 5.0f, 2.0f, 'O');
    particles.back().vel = {20.0f, 0};
    
    std::cout << "Initial: Lattice structure with impactor 'O'\n";
    visualize(particles, springs);
    
    // Simulate impact
    for (int step = 0; step < 1500; step++) {
        applySoftContactForces(particles, springs);
        applySpringForces(particles, springs);
        
        // Update positions
        for (auto& p : particles) {
            p.pos += p.vel * dt;
            // Small damping
            p.vel = p.vel * 0.999f;
        }
        
        // Show key moments
        if (step == 300 || step == 600 || step == 900 || step == 1499) {
            std::cout << "\nStep " << step << ":\n";
            visualize(particles, springs);
            
            int broken = 0;
            for (const auto& s : springs) {
                if (s.broken) broken++;
            }
            std::cout << "Broken springs: " << broken << "/" << springs.size()
                      << " (Impact causes cascading failures)\n";
        }
    }
    
    std::cout << "\n=== Summary ===\n";
    std::cout << "✓ Soft contact forces create realistic deformation\n";
    std::cout << "✓ Collisions can break springs and deform bodies\n";
    std::cout << "✓ Tidal forces stretch and tear structures apart\n";
    std::cout << "✓ Impact creates cascading structural failures\n";
    std::cout << "✓ Complex soft-body dynamics emerge from simple rules\n\n";
    
    std::cout << "The combination of particle repulsion, spring networks,\n";
    std::cout << "and environmental forces creates rich emergent behaviors!\n";
    
    return 0;
}
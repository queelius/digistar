/**
 * @file composite_collision.cpp
 * @brief Spring-based composite body collision demonstration
 * 
 * Shows two composite bodies (made of particles connected by springs) colliding.
 * Demonstrates:
 * - Force distribution through spring networks
 * - Local deformation at contact points
 * - Spring breaking under high stress
 * - Different material behaviors from spring properties
 */

#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>
#include <chrono>
#include <thread>
#include <algorithm>
#include <unordered_map>
#include <random>

// ============================================================================
// Configuration
// ============================================================================

namespace Config {
    // World
    constexpr float WORLD_WIDTH = 100.0f;   // meters
    constexpr float WORLD_HEIGHT = 60.0f;   // meters
    
    // Composite body structure
    constexpr int COMPOSITE_WIDTH = 5;      // particles
    constexpr int COMPOSITE_HEIGHT = 4;     // particles
    constexpr float PARTICLE_SPACING = 2.0f; // meters
    constexpr float PARTICLE_RADIUS = 0.8f;  // meters
    constexpr float PARTICLE_MASS = 10.0f;   // kg
    
    // Spring properties (material properties)
    namespace Material {
        // Rigid material (steel-like)
        constexpr float RIGID_STIFFNESS = 50000.0f;    // N/m
        constexpr float RIGID_DAMPING = 500.0f;        // N⋅s/m
        constexpr float RIGID_BREAK_STRAIN = 0.5f;     // 50% extension before break
        
        // Soft material (rubber-like)
        constexpr float SOFT_STIFFNESS = 5000.0f;
        constexpr float SOFT_DAMPING = 200.0f;
        constexpr float SOFT_BREAK_STRAIN = 2.0f;      // 200% extension
    }
    
    // Contact forces
    constexpr float CONTACT_STIFFNESS = 100000.0f;  // N/m^1.5
    constexpr float CONTACT_DAMPING = 2000.0f;      // N⋅s/m
    
    // Simulation
    constexpr float TIME_STEP = 0.00005f;   // 50 microseconds (very small for stiff springs)
    constexpr float SIMULATION_TIME = 5.0f;
    
    // Collision setup
    constexpr float INITIAL_SEPARATION = 50.0f;  // meters
    constexpr float APPROACH_SPEED = 10.0f;      // m/s
    constexpr float VERTICAL_OFFSET = 0.0f;      // head-on collision
    
    // Display
    constexpr int SCREEN_WIDTH = 120;
    constexpr int SCREEN_HEIGHT = 40;
    constexpr int FPS = 30;
}

// ============================================================================
// Core Structures
// ============================================================================

struct Vector2 {
    float x, y;
    
    Vector2 operator+(const Vector2& other) const { return {x + other.x, y + other.y}; }
    Vector2 operator-(const Vector2& other) const { return {x - other.x, y - other.y}; }
    Vector2 operator*(float scalar) const { return {x * scalar, y * scalar}; }
    Vector2 operator/(float scalar) const { return {x / scalar, y / scalar}; }
    Vector2& operator+=(const Vector2& other) { x += other.x; y += other.y; return *this; }
    Vector2& operator-=(const Vector2& other) { x -= other.x; y -= other.y; return *this; }
    
    float length() const { return std::sqrt(x * x + y * y); }
    float length_squared() const { return x * x + y * y; }
    Vector2 normalized() const { 
        float len = length();
        return len > 0 ? (*this) / len : Vector2{0, 0};
    }
    float dot(const Vector2& other) const { return x * other.x + y * other.y; }
};

struct Particle {
    Vector2 position;
    Vector2 velocity;
    Vector2 force;
    float mass;
    float radius;
    int composite_id;  // Which composite body this belongs to
    bool is_contact;   // Is this particle in contact?
    
    Particle(Vector2 pos, float m = Config::PARTICLE_MASS, float r = Config::PARTICLE_RADIUS) 
        : position(pos), velocity{0, 0}, force{0, 0}, mass(m), radius(r), 
          composite_id(-1), is_contact(false) {}
};

struct Spring {
    int particle1_idx;
    int particle2_idx;
    float rest_length;
    float stiffness;
    float damping;
    float break_strain;
    float current_strain;
    bool broken;
    
    Spring(int p1, int p2, float rest_len, float stiff, float damp, float break_str)
        : particle1_idx(p1), particle2_idx(p2), rest_length(rest_len),
          stiffness(stiff), damping(damp), break_strain(break_str),
          current_strain(0), broken(false) {}
};

struct CompositeBody {
    std::vector<int> particle_indices;
    Vector2 center_of_mass;
    Vector2 velocity;
    float total_mass;
    float bounding_radius;
    std::string name;
    
    void update_properties(const std::vector<Particle>& particles) {
        center_of_mass = {0, 0};
        total_mass = 0;
        
        for (int idx : particle_indices) {
            center_of_mass += particles[idx].position * particles[idx].mass;
            total_mass += particles[idx].mass;
        }
        
        if (total_mass > 0) {
            center_of_mass = center_of_mass / total_mass;
        }
        
        // Calculate bounding radius
        bounding_radius = 0;
        for (int idx : particle_indices) {
            float dist = (particles[idx].position - center_of_mass).length() + particles[idx].radius;
            bounding_radius = std::max(bounding_radius, dist);
        }
        
        // Calculate average velocity
        velocity = {0, 0};
        for (int idx : particle_indices) {
            velocity += particles[idx].velocity;
        }
        velocity = velocity / particle_indices.size();
    }
};

// ============================================================================
// Physics System
// ============================================================================

class PhysicsSystem {
private:
    std::vector<Particle> particles;
    std::vector<Spring> springs;
    std::vector<CompositeBody> composites;
    
    void calculate_spring_forces() {
        for (auto& spring : springs) {
            if (spring.broken) continue;
            
            Particle& p1 = particles[spring.particle1_idx];
            Particle& p2 = particles[spring.particle2_idx];
            
            Vector2 delta = p2.position - p1.position;
            float distance = delta.length();
            
            if (distance > 0) {
                // Calculate strain
                spring.current_strain = (distance - spring.rest_length) / spring.rest_length;
                
                // Check for breaking
                if (std::abs(spring.current_strain) > spring.break_strain) {
                    spring.broken = true;
                    std::cout << "Spring broke! Strain: " << spring.current_strain << "\n";
                    continue;
                }
                
                // Spring force (Hooke's law)
                Vector2 direction = delta.normalized();
                float spring_force = spring.stiffness * (distance - spring.rest_length);
                
                // Damping force
                Vector2 relative_velocity = p2.velocity - p1.velocity;
                float damping_force = spring.damping * relative_velocity.dot(direction);
                
                // Total force
                Vector2 total_force = direction * (spring_force + damping_force);
                
                // Apply forces
                p1.force += total_force;
                p2.force -= total_force;
            }
        }
    }
    
    void calculate_contact_forces() {
        // Reset contact flags
        for (auto& p : particles) {
            p.is_contact = false;
        }
        
        // Check composite-composite collisions
        for (size_t i = 0; i < composites.size(); i++) {
            for (size_t j = i + 1; j < composites.size(); j++) {
                check_composite_collision(i, j);
            }
        }
    }
    
    void check_composite_collision(size_t comp1_idx, size_t comp2_idx) {
        CompositeBody& comp1 = composites[comp1_idx];
        CompositeBody& comp2 = composites[comp2_idx];
        
        // Broad phase: bounding sphere check
        Vector2 delta = comp2.center_of_mass - comp1.center_of_mass;
        float distance = delta.length();
        
        if (distance > comp1.bounding_radius + comp2.bounding_radius) {
            return;  // No collision possible
        }
        
        // Find contact region
        float penetration = comp1.bounding_radius + comp2.bounding_radius - distance;
        Vector2 contact_normal = delta.normalized();
        Vector2 contact_point = comp1.center_of_mass + contact_normal * comp1.bounding_radius;
        float contact_radius = penetration * 3.0f;  // Contact zone size
        
        // Find particles in contact zone
        std::vector<int> contact_particles1;
        std::vector<int> contact_particles2;
        
        for (int idx : comp1.particle_indices) {
            float dist_to_contact = (particles[idx].position - contact_point).length();
            if (dist_to_contact < contact_radius + particles[idx].radius) {
                contact_particles1.push_back(idx);
                particles[idx].is_contact = true;
            }
        }
        
        for (int idx : comp2.particle_indices) {
            float dist_to_contact = (particles[idx].position - contact_point).length();
            if (dist_to_contact < contact_radius + particles[idx].radius) {
                contact_particles2.push_back(idx);
                particles[idx].is_contact = true;
            }
        }
        
        // If no actual particle contact, no collision
        if (contact_particles1.empty() || contact_particles2.empty()) {
            return;
        }
        
        // Calculate contact force magnitude (Hertzian model)
        float effective_radius = (comp1.bounding_radius * comp2.bounding_radius) / 
                                (comp1.bounding_radius + comp2.bounding_radius);
        float stiffness = Config::CONTACT_STIFFNESS * std::sqrt(effective_radius);
        float force_magnitude = stiffness * std::pow(penetration, 1.5f);
        
        // Add damping
        Vector2 relative_velocity = comp2.velocity - comp1.velocity;
        float v_normal = relative_velocity.dot(contact_normal);
        if (v_normal < 0) {  // Approaching
            force_magnitude += Config::CONTACT_DAMPING * std::abs(v_normal);
        }
        
        // Distribute forces among contact particles
        Vector2 force = contact_normal * force_magnitude;
        
        if (!contact_particles1.empty()) {
            Vector2 force_per_particle = force / contact_particles1.size();
            for (int idx : contact_particles1) {
                particles[idx].force -= force_per_particle;
            }
        }
        
        if (!contact_particles2.empty()) {
            Vector2 force_per_particle = force / contact_particles2.size();
            for (int idx : contact_particles2) {
                particles[idx].force += force_per_particle;
            }
        }
    }
    
public:
    void create_composite(const std::string& name, Vector2 center, Vector2 velocity, 
                         bool use_soft_material) {
        CompositeBody composite;
        composite.name = name;
        composite.total_mass = 0;
        
        int start_idx = particles.size();
        
        // Create grid of particles
        for (int y = 0; y < Config::COMPOSITE_HEIGHT; y++) {
            for (int x = 0; x < Config::COMPOSITE_WIDTH; x++) {
                Vector2 pos = center + Vector2{
                    (x - Config::COMPOSITE_WIDTH/2.0f) * Config::PARTICLE_SPACING,
                    (y - Config::COMPOSITE_HEIGHT/2.0f) * Config::PARTICLE_SPACING
                };
                
                particles.emplace_back(pos);
                particles.back().velocity = velocity;
                particles.back().composite_id = composites.size();
                composite.particle_indices.push_back(particles.size() - 1);
            }
        }
        
        // Create springs connecting neighbors
        float stiffness = use_soft_material ? Config::Material::SOFT_STIFFNESS 
                                            : Config::Material::RIGID_STIFFNESS;
        float damping = use_soft_material ? Config::Material::SOFT_DAMPING 
                                          : Config::Material::RIGID_DAMPING;
        float break_strain = use_soft_material ? Config::Material::SOFT_BREAK_STRAIN 
                                               : Config::Material::RIGID_BREAK_STRAIN;
        
        for (int y = 0; y < Config::COMPOSITE_HEIGHT; y++) {
            for (int x = 0; x < Config::COMPOSITE_WIDTH; x++) {
                int idx = start_idx + y * Config::COMPOSITE_WIDTH + x;
                
                // Connect to right neighbor
                if (x < Config::COMPOSITE_WIDTH - 1) {
                    int right_idx = idx + 1;
                    float rest_len = (particles[idx].position - particles[right_idx].position).length();
                    springs.emplace_back(idx, right_idx, rest_len, stiffness, damping, break_strain);
                }
                
                // Connect to bottom neighbor
                if (y < Config::COMPOSITE_HEIGHT - 1) {
                    int bottom_idx = idx + Config::COMPOSITE_WIDTH;
                    float rest_len = (particles[idx].position - particles[bottom_idx].position).length();
                    springs.emplace_back(idx, bottom_idx, rest_len, stiffness, damping, break_strain);
                }
                
                // Diagonal springs for shear resistance
                if (x < Config::COMPOSITE_WIDTH - 1 && y < Config::COMPOSITE_HEIGHT - 1) {
                    int diag_idx = idx + Config::COMPOSITE_WIDTH + 1;
                    float rest_len = (particles[idx].position - particles[diag_idx].position).length();
                    springs.emplace_back(idx, diag_idx, rest_len, stiffness * 0.5f, damping, break_strain);
                }
                
                if (x > 0 && y < Config::COMPOSITE_HEIGHT - 1) {
                    int diag_idx = idx + Config::COMPOSITE_WIDTH - 1;
                    float rest_len = (particles[idx].position - particles[diag_idx].position).length();
                    springs.emplace_back(idx, diag_idx, rest_len, stiffness * 0.5f, damping, break_strain);
                }
            }
        }
        
        composite.update_properties(particles);
        composites.push_back(composite);
    }
    
    void step(float dt) {
        // Clear forces
        for (auto& p : particles) {
            p.force = {0, 0};
        }
        
        // Calculate all forces
        calculate_spring_forces();
        calculate_contact_forces();
        
        // Update composite properties
        for (auto& comp : composites) {
            comp.update_properties(particles);
        }
        
        // Integrate using semi-implicit Euler (stable for stiff springs)
        for (auto& p : particles) {
            Vector2 acceleration = p.force / p.mass;
            p.velocity += acceleration * dt;
            p.position += p.velocity * dt;
        }
    }
    
    const std::vector<Particle>& get_particles() const { return particles; }
    const std::vector<Spring>& get_springs() const { return springs; }
    const std::vector<CompositeBody>& get_composites() const { return composites; }
    
    int count_broken_springs() const {
        int count = 0;
        for (const auto& spring : springs) {
            if (spring.broken) count++;
        }
        return count;
    }
    
    float calculate_total_energy() const {
        float total_ke = 0;
        float total_pe = 0;
        
        for (const auto& p : particles) {
            total_ke += 0.5f * p.mass * p.velocity.length_squared();
        }
        
        for (const auto& spring : springs) {
            if (!spring.broken) {
                Vector2 delta = particles[spring.particle2_idx].position - 
                              particles[spring.particle1_idx].position;
                float extension = delta.length() - spring.rest_length;
                total_pe += 0.5f * spring.stiffness * extension * extension;
            }
        }
        
        return total_ke + total_pe;
    }
};

// ============================================================================
// Visualization
// ============================================================================

class Visualizer {
private:
    std::vector<std::vector<char>> display;
    
public:
    Visualizer() {
        display.resize(Config::SCREEN_HEIGHT, 
                      std::vector<char>(Config::SCREEN_WIDTH, ' '));
    }
    
    void render(const PhysicsSystem& physics, float time) {
        // Clear
        for (auto& row : display) {
            std::fill(row.begin(), row.end(), ' ');
        }
        
        // Draw center lines
        int center_x = Config::SCREEN_WIDTH / 2;
        int center_y = Config::SCREEN_HEIGHT / 2;
        for (int x = 0; x < Config::SCREEN_WIDTH; x++) {
            display[center_y][x] = '-';
        }
        for (int y = 0; y < Config::SCREEN_HEIGHT; y++) {
            display[y][center_x] = '|';
        }
        display[center_y][center_x] = '+';
        
        // Draw springs
        for (const auto& spring : physics.get_springs()) {
            if (spring.broken) continue;
            
            const auto& particles = physics.get_particles();
            Vector2 p1 = particles[spring.particle1_idx].position;
            Vector2 p2 = particles[spring.particle2_idx].position;
            
            // Simple line drawing (just endpoints for now)
            int x1 = (int)((p1.x + Config::WORLD_WIDTH/2) / Config::WORLD_WIDTH * Config::SCREEN_WIDTH);
            int y1 = (int)((Config::WORLD_HEIGHT/2 - p1.y) / Config::WORLD_HEIGHT * Config::SCREEN_HEIGHT);
            int x2 = (int)((p2.x + Config::WORLD_WIDTH/2) / Config::WORLD_WIDTH * Config::SCREEN_WIDTH);
            int y2 = (int)((Config::WORLD_HEIGHT/2 - p2.y) / Config::WORLD_HEIGHT * Config::SCREEN_HEIGHT);
            
            // Draw connection indicator
            if (x1 >= 0 && x1 < Config::SCREEN_WIDTH && y1 >= 0 && y1 < Config::SCREEN_HEIGHT) {
                display[y1][x1] = '.';
            }
            if (x2 >= 0 && x2 < Config::SCREEN_WIDTH && y2 >= 0 && y2 < Config::SCREEN_HEIGHT) {
                display[y2][x2] = '.';
            }
        }
        
        // Draw particles
        for (const auto& particle : physics.get_particles()) {
            int x = (int)((particle.position.x + Config::WORLD_WIDTH/2) / Config::WORLD_WIDTH * Config::SCREEN_WIDTH);
            int y = (int)((Config::WORLD_HEIGHT/2 - particle.position.y) / Config::WORLD_HEIGHT * Config::SCREEN_HEIGHT);
            
            if (x >= 0 && x < Config::SCREEN_WIDTH && y >= 0 && y < Config::SCREEN_HEIGHT) {
                char symbol = 'o';
                if (particle.is_contact) symbol = 'X';  // Contact particle
                else if (particle.composite_id == 0) symbol = 'A';  // Composite 1
                else if (particle.composite_id == 1) symbol = 'B';  // Composite 2
                
                display[y][x] = symbol;
            }
        }
        
        // Output
        std::cout << "\033[H";
        std::cout << "+" << std::string(Config::SCREEN_WIDTH, '-') << "+\n";
        for (const auto& row : display) {
            std::cout << "|";
            for (char c : row) std::cout << c;
            std::cout << "|\n";
        }
        std::cout << "+" << std::string(Config::SCREEN_WIDTH, '-') << "+\n";
        
        // Info panel
        const auto& composites = physics.get_composites();
        std::cout << std::fixed << std::setprecision(2);
        std::cout << "Time: " << time << "s | ";
        std::cout << "Broken springs: " << physics.count_broken_springs() << "/" << physics.get_springs().size() << " | ";
        std::cout << "Energy: " << std::scientific << std::setprecision(2) << physics.calculate_total_energy() << " J | ";
        
        if (composites.size() >= 2) {
            float separation = (composites[1].center_of_mass - composites[0].center_of_mass).length();
            std::cout << "Separation: " << std::fixed << std::setprecision(1) << separation << "m";
        }
        std::cout << "\n";
    }
};

// ============================================================================
// Main Simulation
// ============================================================================

int main() {
    std::cout << "=== Composite Body Collision Test ===\n";
    std::cout << "Spring-connected particles forming deformable bodies\n\n";
    
    std::cout << "Setup:\n";
    std::cout << "  Body 1 (A): Rigid material (steel-like)\n";
    std::cout << "  Body 2 (B): Soft material (rubber-like)\n";
    std::cout << "  Approach speed: " << Config::APPROACH_SPEED << " m/s each\n";
    std::cout << "  Contact forces distributed to touching particles only\n";
    std::cout << "  Springs can break under high strain\n\n";
    
    std::cout << "Starting in 2 seconds...\n";
    std::this_thread::sleep_for(std::chrono::seconds(2));
    
    std::cout << "\033[2J\033[H";  // Clear screen
    
    PhysicsSystem physics;
    Visualizer visualizer;
    
    // Create two composite bodies
    physics.create_composite(
        "Rigid Body",
        Vector2{-Config::INITIAL_SEPARATION/2, Config::VERTICAL_OFFSET/2},
        Vector2{Config::APPROACH_SPEED, 0},
        false  // Use rigid material
    );
    
    physics.create_composite(
        "Soft Body",
        Vector2{Config::INITIAL_SEPARATION/2, -Config::VERTICAL_OFFSET/2},
        Vector2{-Config::APPROACH_SPEED, 0},
        true   // Use soft material
    );
    
    float time = 0;
    float dt = Config::TIME_STEP;
    int frame = 0;
    int display_counter = 0;
    auto last_frame = std::chrono::steady_clock::now();
    
    while (time < Config::SIMULATION_TIME) {
        // Physics update
        physics.step(dt);
        time += dt;
        frame++;
        
        // Display at target FPS
        display_counter++;
        if (display_counter >= (int)(1.0f / (Config::FPS * dt))) {
            display_counter = 0;
            
            auto now = std::chrono::steady_clock::now();
            auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - last_frame).count();
            if (elapsed < 1000 / Config::FPS) {
                std::this_thread::sleep_for(std::chrono::milliseconds(1000 / Config::FPS - elapsed));
            }
            
            visualizer.render(physics, time);
            last_frame = std::chrono::steady_clock::now();
        }
    }
    
    // Final summary
    std::cout << "\n=== Collision Summary ===\n";
    std::cout << "Total broken springs: " << physics.count_broken_springs() << "/" << physics.get_springs().size() << "\n";
    std::cout << "Final energy: " << physics.calculate_total_energy() << " J\n";
    
    const auto& composites = physics.get_composites();
    if (composites.size() >= 2) {
        std::cout << "\nComposite 1 (Rigid):\n";
        std::cout << "  Final velocity: (" << composites[0].velocity.x << ", " << composites[0].velocity.y << ") m/s\n";
        std::cout << "  Deformation: Minimal (stiff springs)\n";
        
        std::cout << "\nComposite 2 (Soft):\n";
        std::cout << "  Final velocity: (" << composites[1].velocity.x << ", " << composites[1].velocity.y << ") m/s\n";
        std::cout << "  Deformation: Significant (soft springs)\n";
    }
    
    std::cout << "\nNote how forces propagate differently through rigid vs soft materials!\n";
    
    return 0;
}
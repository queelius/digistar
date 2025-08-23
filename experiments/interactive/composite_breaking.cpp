/**
 * @file composite_breaking.cpp
 * @brief Asymmetric composite bodies with spring breaking demonstration
 * 
 * Shows:
 * - Funny-shaped composite bodies (L-shape vs diamond)
 * - Glancing collision with vertical offset
 * - Spring breaking under stress
 * - False positive filtering (bounding spheres overlap but shapes don't)
 * - Fragmentation and deformation
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
#include <set>
#include <functional>

// ============================================================================
// Configuration
// ============================================================================

namespace Config {
    // World
    constexpr float WORLD_WIDTH = 120.0f;
    constexpr float WORLD_HEIGHT = 80.0f;
    
    // Particle properties
    constexpr float PARTICLE_SPACING = 2.5f;
    constexpr float PARTICLE_RADIUS = 1.0f;
    constexpr float PARTICLE_MASS = 15.0f;
    
    // Spring properties - semi-rigid with breaking
    constexpr float SPRING_STIFFNESS = 30000.0f;     // Semi-rigid
    constexpr float SPRING_DAMPING = 300.0f;
    constexpr float SPRING_BREAK_STRAIN = 0.15f;     // 15% - more brittle overall
    constexpr float DIAGONAL_STIFFNESS_FACTOR = 0.7f; // Diagonals slightly weaker
    
    // Contact forces
    constexpr float CONTACT_STIFFNESS = 200000.0f;
    constexpr float CONTACT_DAMPING = 3000.0f;
    
    // Simulation
    constexpr float TIME_STEP = 0.00002f;    // 20 microseconds
    constexpr float SIMULATION_TIME = 6.0f;
    
    // Collision setup
    constexpr float INITIAL_SEPARATION = 60.0f;
    constexpr float APPROACH_SPEED = 25.0f;      // Very fast for breaking
    constexpr float VERTICAL_OFFSET = 3.0f;      // Almost direct hit
    
    // Display
    constexpr int SCREEN_WIDTH = 140;
    constexpr int SCREEN_HEIGHT = 45;
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
    int composite_id;
    bool is_contact;
    int fragment_id;  // Track which fragment after breaking
    
    Particle(Vector2 pos, float m = Config::PARTICLE_MASS, float r = Config::PARTICLE_RADIUS) 
        : position(pos), velocity{0, 0}, force{0, 0}, mass(m), radius(r), 
          composite_id(-1), is_contact(false), fragment_id(-1) {}
};

struct Spring {
    int particle1_idx;
    int particle2_idx;
    float rest_length;
    float stiffness;
    float damping;
    float break_strain;
    float current_strain;
    float max_strain_experienced;
    bool broken;
    
    Spring(int p1, int p2, float rest_len, float stiff, float damp, float break_str)
        : particle1_idx(p1), particle2_idx(p2), rest_length(rest_len),
          stiffness(stiff), damping(damp), break_strain(break_str),
          current_strain(0), max_strain_experienced(0), broken(false) {}
};

struct CompositeBody {
    std::vector<int> particle_indices;
    Vector2 center_of_mass;
    Vector2 velocity;
    float total_mass;
    float bounding_radius;
    std::string name;
    char symbol;
    
    void update_properties(const std::vector<Particle>& particles) {
        if (particle_indices.empty()) return;
        
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
// Shape Creation Functions
// ============================================================================

std::vector<Vector2> create_L_shape() {
    // L-shaped body (like a Tetris piece)
    //  ##
    //  #
    //  #
    //  ###
    std::vector<Vector2> positions;
    
    // Vertical part
    for (int y = 0; y < 4; y++) {
        positions.push_back({0, y * Config::PARTICLE_SPACING});
    }
    
    // Horizontal parts
    positions.push_back({Config::PARTICLE_SPACING, 0});
    positions.push_back({2 * Config::PARTICLE_SPACING, 0});
    positions.push_back({Config::PARTICLE_SPACING, 3 * Config::PARTICLE_SPACING});
    
    return positions;
}

std::vector<Vector2> create_diamond_shape() {
    // Diamond/rhombus shape
    //    #
    //   ###
    //  #####
    //   ###
    //    #
    std::vector<Vector2> positions;
    
    int widths[] = {1, 3, 5, 3, 1};
    for (int y = 0; y < 5; y++) {
        int width = widths[y];
        for (int x = 0; x < width; x++) {
            float x_offset = (x - width/2.0f) * Config::PARTICLE_SPACING;
            float y_offset = (y - 2) * Config::PARTICLE_SPACING;
            positions.push_back({x_offset, y_offset});
        }
    }
    
    return positions;
}

// ============================================================================
// Physics System
// ============================================================================

class PhysicsSystem {
private:
    std::vector<Particle> particles;
    std::vector<Spring> springs;
    std::vector<CompositeBody> composites;
    int spring_break_count = 0;
    float collision_energy_lost = 0;
    
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
                spring.max_strain_experienced = std::max(spring.max_strain_experienced, 
                                                        std::abs(spring.current_strain));
                
                // Check for breaking
                if (std::abs(spring.current_strain) > spring.break_strain) {
                    spring.broken = true;
                    spring_break_count++;
                    
                    // Add some separation velocity when spring breaks
                    Vector2 break_impulse = delta.normalized() * 5.0f;  // Stronger impulse
                    p1.velocity -= break_impulse;
                    p2.velocity += break_impulse;
                    
                    // Visual feedback
                    std::cout << "SNAP! Spring " << spring.particle1_idx << "-" << spring.particle2_idx 
                              << " broke at strain: " << spring.current_strain << "\n";
                    
                    continue;
                }
                
                // Spring force
                Vector2 direction = delta.normalized();
                float spring_force = spring.stiffness * (distance - spring.rest_length);
                
                // Damping
                Vector2 relative_velocity = p2.velocity - p1.velocity;
                float damping_force = spring.damping * relative_velocity.dot(direction);
                
                Vector2 total_force = direction * (spring_force + damping_force);
                
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
        
        // Broad phase
        Vector2 delta = comp2.center_of_mass - comp1.center_of_mass;
        float distance = delta.length();
        
        if (distance > comp1.bounding_radius + comp2.bounding_radius) {
            return;  // No collision possible
        }
        
        // Find potential contact region
        float penetration = comp1.bounding_radius + comp2.bounding_radius - distance;
        Vector2 contact_normal = delta.normalized();
        Vector2 contact_point = comp1.center_of_mass + contact_normal * (comp1.bounding_radius - penetration/2);
        float contact_radius = std::max(10.0f, penetration * 4.0f);
        
        // Find particles actually in contact zone
        std::vector<int> contact_particles1;
        std::vector<int> contact_particles2;
        
        // Check actual particle-particle collisions (not just contact zone)
        for (int idx1 : comp1.particle_indices) {
            for (int idx2 : comp2.particle_indices) {
                Vector2 p_delta = particles[idx2].position - particles[idx1].position;
                float p_dist = p_delta.length();
                float min_dist = particles[idx1].radius + particles[idx2].radius;
                
                if (p_dist < min_dist) {
                    // Actual contact!
                    contact_particles1.push_back(idx1);
                    contact_particles2.push_back(idx2);
                    particles[idx1].is_contact = true;
                    particles[idx2].is_contact = true;
                    
                    // Apply contact force directly between these particles
                    float overlap = min_dist - p_dist;
                    Vector2 normal = p_delta.normalized();
                    
                    // Hertzian contact
                    float effective_radius = (particles[idx1].radius * particles[idx2].radius) / 
                                           (particles[idx1].radius + particles[idx2].radius);
                    float force_magnitude = Config::CONTACT_STIFFNESS * std::sqrt(effective_radius) * 
                                          std::pow(overlap, 1.5f);
                    
                    // Damping
                    Vector2 v_rel = particles[idx2].velocity - particles[idx1].velocity;
                    float v_normal = v_rel.dot(normal);
                    if (v_normal < 0) {
                        force_magnitude += Config::CONTACT_DAMPING * std::abs(v_normal);
                        collision_energy_lost += 0.5f * std::abs(v_normal) * Config::TIME_STEP;
                    }
                    
                    Vector2 force = normal * force_magnitude;
                    particles[idx1].force -= force;
                    particles[idx2].force += force;
                }
            }
        }
        
        // If bounding spheres overlapped but no actual particle contact, it was a false positive!
        // This naturally filters out non-colliding funny shapes
    }
    
    void detect_fragments() {
        // Use union-find to detect connected components after spring breaking
        std::vector<int> parent(particles.size());
        for (size_t i = 0; i < particles.size(); i++) {
            parent[i] = i;
        }
        
        std::function<int(int)> find = [&](int x) {
            if (parent[x] != x) parent[x] = find(parent[x]);
            return parent[x];
        };
        
        auto unite = [&](int x, int y) {
            int px = find(x), py = find(y);
            if (px != py) parent[px] = py;
        };
        
        // Unite particles connected by unbroken springs
        for (const auto& spring : springs) {
            if (!spring.broken) {
                unite(spring.particle1_idx, spring.particle2_idx);
            }
        }
        
        // Assign fragment IDs
        std::unordered_map<int, int> fragment_map;
        int fragment_count = 0;
        for (size_t i = 0; i < particles.size(); i++) {
            int root = find(i);
            if (fragment_map.find(root) == fragment_map.end()) {
                fragment_map[root] = fragment_count++;
            }
            particles[i].fragment_id = fragment_map[root];
        }
    }
    
public:
    void create_L_composite(const std::string& name, Vector2 center, Vector2 velocity) {
        CompositeBody composite;
        composite.name = name;
        composite.symbol = 'L';
        
        auto shape = create_L_shape();
        int start_idx = particles.size();
        
        // Create particles
        for (const auto& offset : shape) {
            particles.emplace_back(center + offset);
            particles.back().velocity = velocity;
            particles.back().composite_id = composites.size();
            composite.particle_indices.push_back(particles.size() - 1);
        }
        
        // Create springs between all nearby particles
        for (size_t i = 0; i < shape.size(); i++) {
            for (size_t j = i + 1; j < shape.size(); j++) {
                int idx1 = start_idx + i;
                int idx2 = start_idx + j;
                float dist = (particles[idx1].position - particles[idx2].position).length();
                
                // Connect if close enough (neighbors or diagonals)
                if (dist < Config::PARTICLE_SPACING * 2.2f) {
                    float stiffness = Config::SPRING_STIFFNESS;
                    float break_strain = Config::SPRING_BREAK_STRAIN;
                    
                    // Make the "elbow" of the L EXTREMELY BRITTLE
                    // These connect the vertical and horizontal parts
                    if ((i == 0 && j == 3) || (i == 0 && j == 4) || (i == 0 && j == 5) || 
                        (i == 0 && j == 6) || (i == 3 && j == 4) || (i == 3 && j == 5)) {
                        stiffness *= 2.0f;  // Very stiff - transmits forces well
                        break_strain = 0.02f;  // EXTREMELY BRITTLE - breaks at just 2% strain!
                    }
                    
                    if (dist > Config::PARTICLE_SPACING * 1.5f) {
                        // Diagonal spring - slightly weaker
                        stiffness *= Config::DIAGONAL_STIFFNESS_FACTOR;
                    }
                    springs.emplace_back(idx1, idx2, dist, stiffness, 
                                       Config::SPRING_DAMPING, break_strain);
                }
            }
        }
        
        composite.update_properties(particles);
        composites.push_back(composite);
    }
    
    void create_diamond_composite(const std::string& name, Vector2 center, Vector2 velocity) {
        CompositeBody composite;
        composite.name = name;
        composite.symbol = 'D';
        
        auto shape = create_diamond_shape();
        int start_idx = particles.size();
        
        // Create particles
        for (const auto& offset : shape) {
            particles.emplace_back(center + offset);
            particles.back().velocity = velocity;
            particles.back().composite_id = composites.size();
            composite.particle_indices.push_back(particles.size() - 1);
        }
        
        // Create springs
        for (size_t i = 0; i < shape.size(); i++) {
            for (size_t j = i + 1; j < shape.size(); j++) {
                int idx1 = start_idx + i;
                int idx2 = start_idx + j;
                float dist = (particles[idx1].position - particles[idx2].position).length();
                
                if (dist < Config::PARTICLE_SPACING * 2.2f) {
                    float stiffness = Config::SPRING_STIFFNESS;
                    if (dist > Config::PARTICLE_SPACING * 1.5f) {
                        stiffness *= Config::DIAGONAL_STIFFNESS_FACTOR;
                    }
                    springs.emplace_back(idx1, idx2, dist, stiffness, 
                                       Config::SPRING_DAMPING, Config::SPRING_BREAK_STRAIN);
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
        
        // Calculate forces
        calculate_spring_forces();
        calculate_contact_forces();
        
        // Detect fragments
        detect_fragments();
        
        // Update composites
        for (auto& comp : composites) {
            comp.update_properties(particles);
        }
        
        // Integrate
        for (auto& p : particles) {
            Vector2 acceleration = p.force / p.mass;
            p.velocity += acceleration * dt;
            
            // No global damping - we're in space!
            // Spring damping handles internal vibrations only
            
            p.position += p.velocity * dt;
        }
    }
    
    const std::vector<Particle>& get_particles() const { return particles; }
    const std::vector<Spring>& get_springs() const { return springs; }
    const std::vector<CompositeBody>& get_composites() const { return composites; }
    int get_spring_break_count() const { return spring_break_count; }
    int get_fragment_count() const {
        std::set<int> fragments;
        for (const auto& p : particles) {
            fragments.insert(p.fragment_id);
        }
        return fragments.size();
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
        
        // Draw springs (faint)
        for (const auto& spring : physics.get_springs()) {
            if (spring.broken) continue;
            
            const auto& particles = physics.get_particles();
            Vector2 p1 = particles[spring.particle1_idx].position;
            Vector2 p2 = particles[spring.particle2_idx].position;
            
            int x1 = (int)((p1.x + Config::WORLD_WIDTH/2) / Config::WORLD_WIDTH * Config::SCREEN_WIDTH);
            int y1 = (int)((Config::WORLD_HEIGHT/2 - p1.y) / Config::WORLD_HEIGHT * Config::SCREEN_HEIGHT);
            
            if (x1 >= 0 && x1 < Config::SCREEN_WIDTH && y1 >= 0 && y1 < Config::SCREEN_HEIGHT) {
                if (display[y1][x1] == ' ') display[y1][x1] = '.';
            }
        }
        
        // Draw particles
        for (const auto& particle : physics.get_particles()) {
            int x = (int)((particle.position.x + Config::WORLD_WIDTH/2) / Config::WORLD_WIDTH * Config::SCREEN_WIDTH);
            int y = (int)((Config::WORLD_HEIGHT/2 - particle.position.y) / Config::WORLD_HEIGHT * Config::SCREEN_HEIGHT);
            
            if (x >= 0 && x < Config::SCREEN_WIDTH && y >= 0 && y < Config::SCREEN_HEIGHT) {
                char symbol = 'o';
                
                if (particle.is_contact) {
                    symbol = 'X';  // Collision!
                } else if (particle.fragment_id >= 0) {
                    // Different symbols for different fragments
                    if (particle.composite_id == 0) {
                        symbol = (particle.fragment_id == 0) ? 'L' : 'l';
                    } else {
                        symbol = (particle.fragment_id == 0) ? 'D' : 
                                (particle.fragment_id == 1) ? 'd' : 
                                ('a' + particle.fragment_id);
                    }
                }
                
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
        
        // Info
        std::cout << std::fixed << std::setprecision(2);
        std::cout << "Time: " << time << "s | ";
        std::cout << "Broken: " << physics.get_spring_break_count() << "/" << physics.get_springs().size() << " | ";
        std::cout << "Fragments: " << physics.get_fragment_count() << " | ";
        
        const auto& composites = physics.get_composites();
        if (composites.size() >= 2) {
            std::cout << "Sep: " << (composites[1].center_of_mass - composites[0].center_of_mass).length() << "m | ";
            std::cout << "V1: " << composites[0].velocity.length() << " V2: " << composites[1].velocity.length();
        }
        std::cout << "\n";
        
        std::cout << "Legend: L=L-shape intact, l=L-fragment, D=Diamond intact, d=Diamond fragment, X=Contact\n";
    }
};

// ============================================================================
// Main
// ============================================================================

int main() {
    std::cout << "=== Asymmetric Composite Collision with Breaking ===\n\n";
    
    std::cout << "Setup:\n";
    std::cout << "  L-shaped body approaching from left\n";
    std::cout << "  Diamond-shaped body approaching from right\n";
    std::cout << "  Vertical offset for glancing collision\n";
    std::cout << "  Springs will break at 30% strain\n";
    std::cout << "  Watch for fragmentation!\n\n";
    
    std::cout << "Starting in 2 seconds...\n";
    std::this_thread::sleep_for(std::chrono::seconds(2));
    
    std::cout << "\033[2J\033[H";
    
    PhysicsSystem physics;
    Visualizer visualizer;
    
    // Create L-shaped body (left, moving right and slightly up)
    physics.create_L_composite(
        "L-Shape",
        Vector2{-Config::INITIAL_SEPARATION/2, -Config::VERTICAL_OFFSET/2},
        Vector2{Config::APPROACH_SPEED, Config::APPROACH_SPEED * 0.1f}
    );
    
    // Create diamond body (right, moving left and slightly down)
    physics.create_diamond_composite(
        "Diamond",
        Vector2{Config::INITIAL_SEPARATION/2, Config::VERTICAL_OFFSET/2},
        Vector2{-Config::APPROACH_SPEED, -Config::APPROACH_SPEED * 0.1f}
    );
    
    float time = 0;
    float dt = Config::TIME_STEP;
    int frame = 0;
    int display_counter = 0;
    auto last_frame = std::chrono::steady_clock::now();
    
    bool collision_happened = false;
    int max_fragments = 2;
    
    while (time < Config::SIMULATION_TIME) {
        physics.step(dt);
        time += dt;
        frame++;
        
        // Check for fragmentation
        if (physics.get_fragment_count() > max_fragments) {
            max_fragments = physics.get_fragment_count();
            collision_happened = true;
        }
        
        // Display
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
    
    // Summary
    std::cout << "\n=== Collision Summary ===\n";
    std::cout << "Springs broken: " << physics.get_spring_break_count() << "/" << physics.get_springs().size() << "\n";
    std::cout << "Final fragment count: " << physics.get_fragment_count() << "\n";
    
    if (collision_happened) {
        std::cout << "\nCollision resulted in fragmentation!\n";
        std::cout << "The glancing blow and asymmetric shapes caused stress concentrations.\n";
        std::cout << "Springs broke where forces exceeded material limits.\n";
    } else {
        std::cout << "\nBodies bounced without breaking (try higher speeds or closer impact).\n";
    }
    
    std::cout << "\nNote: Bounding spheres may have overlapped without actual collision\n";
    std::cout << "      (false positives automatically filtered by particle-level checks)\n";
    
    return 0;
}
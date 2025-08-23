/**
 * @file solar_system_collision.cpp
 * @brief Solar system simulation with soft contact forces and collision detection
 * 
 * Implements:
 * - Sparse spatial grid for efficient collision detection
 * - Hertzian soft contact model
 * - Two planets on collision course in Mars orbit
 * - Energy dissipation through collisions
 * 
 * Based on SOFT_CONTACT_FORCES.md and SPATIAL_INDEXING_DESIGN.md
 */

#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>
#include <cstdlib>
#include <chrono>
#include <thread>
#include <cstring>
#include <algorithm>
#include <random>
#include <memory>
#include <string>
#include <map>
#include <array>
#include <unordered_map>
#include <termios.h>
#include <unistd.h>
#include <fcntl.h>
#include <omp.h>

// ============================================================================
// Configuration
// ============================================================================

namespace Config {
    namespace Simulation {
        constexpr double G = 4.0 * M_PI * M_PI;  // G in AU³/M☉·year²
        constexpr double TIME_STEP = 0.00001;    // years
        constexpr double SOFTENING = 1e-6;       // Gravity softening
        constexpr int OMP_THREADS = 4;
    }
    
    namespace Collision {
        constexpr float CONTACT_STIFFNESS = 1000.0f;  // Hertzian stiffness
        constexpr float CONTACT_DAMPING = 10.0f;      // Velocity damping
        constexpr float RESTITUTION = 0.6f;           // Coefficient of restitution
        constexpr float MIN_COLLISION_VELOCITY = 0.001f; // Ignore very slow collisions
    }
    
    namespace SpatialGrid {
        constexpr float CELL_SIZE_FACTOR = 3.0f;  // Cell size = max_radius * factor
        constexpr float WORLD_SIZE = 1000.0f;     // AU
        constexpr int EXPECTED_PARTICLES_PER_CELL = 10;
    }
    
    namespace Display {
        constexpr int SCREEN_WIDTH = 120;
        constexpr int SCREEN_HEIGHT = 40;
        constexpr int FRAME_SKIP = 5;
        constexpr int TARGET_FPS = 20;
        constexpr float DEFAULT_ZOOM = 0.5f;
    }
}

// ============================================================================
// Core Data Structures
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

struct ContactInfo {
    Vector2 force;
    float penetration_depth;
    Vector2 contact_normal;
    float impulse_magnitude;
};

struct CelestialBody {
    Vector2 position;
    Vector2 velocity;
    Vector2 force;
    float mass;
    float radius;
    std::string name;
    char symbol;
    
    // Collision tracking
    uint64_t current_cell_idx;  // Which spatial grid cell
    float collision_energy_lost;  // Track energy dissipation
    
    // Visual properties
    bool is_colliding;
    float temperature;  // Visual effect from collision heat
    
    CelestialBody() : position{0,0}, velocity{0,0}, force{0,0}, 
                      mass(1.0f), radius(0.01f), name(""), symbol('.'),
                      current_cell_idx(0), collision_energy_lost(0),
                      is_colliding(false), temperature(0) {}
};

// ============================================================================
// Sparse Spatial Grid for Collision Detection
// ============================================================================

class SparseGrid {
private:
    std::unordered_map<uint64_t, std::vector<int>> cells;
    float cell_size;
    float world_size;
    int grid_resolution;
    
    uint64_t hash_cell(int x, int y) const {
        // Handle negative coordinates and wraparound
        x = ((x % grid_resolution) + grid_resolution) % grid_resolution;
        y = ((y % grid_resolution) + grid_resolution) % grid_resolution;
        return (uint64_t(x) << 32) | uint64_t(y);
    }
    
    std::pair<int, int> get_cell_coords(const Vector2& pos) const {
        int x = static_cast<int>(std::floor(pos.x / cell_size));
        int y = static_cast<int>(std::floor(pos.y / cell_size));
        return {x, y};
    }
    
public:
    SparseGrid() : cell_size(1.0f), world_size(Config::SpatialGrid::WORLD_SIZE) {
        // Adaptive cell size based on largest expected body
        cell_size = 0.1f;  // Start with reasonable size for planets
        grid_resolution = static_cast<int>(world_size / cell_size);
    }
    
    void set_cell_size(float max_radius) {
        cell_size = max_radius * Config::SpatialGrid::CELL_SIZE_FACTOR;
        grid_resolution = static_cast<int>(world_size / cell_size);
        
        // Ensure even division for toroidal space
        grid_resolution = (grid_resolution / 2) * 2;  // Make even
        cell_size = world_size / grid_resolution;
    }
    
    void clear() {
        cells.clear();
    }
    
    void build(std::vector<CelestialBody>& bodies) {
        clear();
        cells.reserve(bodies.size() / Config::SpatialGrid::EXPECTED_PARTICLES_PER_CELL);
        
        for (size_t i = 0; i < bodies.size(); i++) {
            auto [x, y] = get_cell_coords(bodies[i].position);
            uint64_t key = hash_cell(x, y);
            cells[key].push_back(i);
            bodies[i].current_cell_idx = key;
        }
    }
    
    void incremental_update(std::vector<CelestialBody>& bodies) {
        // Only update bodies that changed cells (~1% per frame)
        for (size_t i = 0; i < bodies.size(); i++) {
            auto [x, y] = get_cell_coords(bodies[i].position);
            uint64_t new_key = hash_cell(x, y);
            uint64_t old_key = bodies[i].current_cell_idx;
            
            if (new_key != old_key) {
                // Remove from old cell
                auto& old_cell = cells[old_key];
                old_cell.erase(std::remove(old_cell.begin(), old_cell.end(), i), old_cell.end());
                
                // Add to new cell
                cells[new_key].push_back(i);
                bodies[i].current_cell_idx = new_key;
            }
        }
    }
    
    std::vector<int> get_neighbors(const Vector2& pos, float radius) {
        std::vector<int> neighbors;
        
        auto [cx, cy] = get_cell_coords(pos);
        int cell_radius = static_cast<int>(std::ceil(radius / cell_size));
        
        // Check all cells within radius
        for (int dy = -cell_radius; dy <= cell_radius; dy++) {
            for (int dx = -cell_radius; dx <= cell_radius; dx++) {
                uint64_t key = hash_cell(cx + dx, cy + dy);
                
                auto it = cells.find(key);
                if (it != cells.end()) {
                    neighbors.insert(neighbors.end(), it->second.begin(), it->second.end());
                }
            }
        }
        
        return neighbors;
    }
    
    void get_collision_pairs(std::vector<CelestialBody>& bodies, 
                            std::vector<std::pair<int, int>>& pairs) {
        pairs.clear();
        
        // For each occupied cell
        for (const auto& [key, indices] : cells) {
            // Check within cell
            for (size_t i = 0; i < indices.size(); i++) {
                for (size_t j = i + 1; j < indices.size(); j++) {
                    int idx1 = indices[i];
                    int idx2 = indices[j];
                    
                    // Quick distance check
                    Vector2 diff = bodies[idx2].position - bodies[idx1].position;
                    float dist_sq = diff.length_squared();
                    float sum_radii = bodies[idx1].radius + bodies[idx2].radius;
                    
                    if (dist_sq < sum_radii * sum_radii) {
                        pairs.push_back({idx1, idx2});
                    }
                }
            }
            
            // Check with neighboring cells (avoid double-checking)
            auto [cx, cy] = get_cell_coords(bodies[indices[0]].position);
            
            // Only check "forward" neighbors to avoid duplicates
            for (int dy = -1; dy <= 1; dy++) {
                for (int dx = -1; dx <= 1; dx++) {
                    if (dx == 0 && dy == 0) continue;  // Skip self
                    if (dx < 0 || (dx == 0 && dy < 0)) continue;  // Skip "backward" neighbors
                    
                    uint64_t neighbor_key = hash_cell(cx + dx, cy + dy);
                    auto neighbor_it = cells.find(neighbor_key);
                    
                    if (neighbor_it != cells.end()) {
                        // Check all pairs between cells
                        for (int idx1 : indices) {
                            for (int idx2 : neighbor_it->second) {
                                Vector2 diff = bodies[idx2].position - bodies[idx1].position;
                                float dist_sq = diff.length_squared();
                                float sum_radii = bodies[idx1].radius + bodies[idx2].radius;
                                
                                if (dist_sq < sum_radii * sum_radii) {
                                    pairs.push_back({idx1, idx2});
                                }
                            }
                        }
                    }
                }
            }
        }
    }
};

// ============================================================================
// Soft Contact Physics
// ============================================================================

class CollisionSystem {
private:
    SparseGrid spatial_grid;
    std::vector<std::pair<int, int>> collision_pairs;
    float total_energy_lost;
    
    ContactInfo calculate_contact(const CelestialBody& b1, const CelestialBody& b2) {
        ContactInfo contact = {{0, 0}, 0, {0, 0}, 0};
        
        Vector2 delta = b2.position - b1.position;
        float dist = delta.length();
        float min_dist = b1.radius + b2.radius;
        
        if (dist < min_dist && dist > 0) {
            // Penetration depth
            contact.penetration_depth = min_dist - dist;
            contact.contact_normal = delta.normalized();
            
            // Hertzian contact model (power 1.5 for spheres)
            float effective_radius = (b1.radius * b2.radius) / (b1.radius + b2.radius);
            float stiffness = Config::Collision::CONTACT_STIFFNESS * std::sqrt(effective_radius);
            float force_magnitude = stiffness * std::pow(contact.penetration_depth, 1.5f);
            
            // Add damping to prevent oscillation
            Vector2 v_rel = b2.velocity - b1.velocity;
            float v_normal = v_rel.dot(contact.contact_normal);
            
            // Only add damping if bodies are approaching
            if (v_normal < 0) {
                force_magnitude += Config::Collision::CONTACT_DAMPING * std::abs(v_normal);
            }
            
            contact.force = contact.contact_normal * force_magnitude;
            contact.impulse_magnitude = force_magnitude;
        }
        
        return contact;
    }
    
public:
    CollisionSystem() : total_energy_lost(0) {}
    
    void initialize(std::vector<CelestialBody>& bodies) {
        // Find maximum radius for cell size
        float max_radius = 0;
        for (const auto& body : bodies) {
            max_radius = std::max(max_radius, body.radius);
        }
        
        spatial_grid.set_cell_size(max_radius);
        spatial_grid.build(bodies);
    }
    
    void update(std::vector<CelestialBody>& bodies, float dt) {
        // Update spatial grid incrementally
        spatial_grid.incremental_update(bodies);
        
        // Find collision pairs
        spatial_grid.get_collision_pairs(bodies, collision_pairs);
        
        // Reset collision flags
        for (auto& body : bodies) {
            body.is_colliding = false;
        }
        
        // Process collisions
        #pragma omp parallel for reduction(+:total_energy_lost)
        for (size_t i = 0; i < collision_pairs.size(); i++) {
            int idx1 = collision_pairs[i].first;
            int idx2 = collision_pairs[i].second;
            
            ContactInfo contact = calculate_contact(bodies[idx1], bodies[idx2]);
            
            if (contact.penetration_depth > 0) {
                // Apply forces
                #pragma omp atomic
                bodies[idx1].force.x -= contact.force.x;
                #pragma omp atomic
                bodies[idx1].force.y -= contact.force.y;
                #pragma omp atomic
                bodies[idx2].force.x += contact.force.x;
                #pragma omp atomic
                bodies[idx2].force.y += contact.force.y;
                
                // Mark as colliding for visualization
                bodies[idx1].is_colliding = true;
                bodies[idx2].is_colliding = true;
                
                // Calculate energy loss (for tracking)
                Vector2 v_rel = bodies[idx2].velocity - bodies[idx1].velocity;
                float v_normal = v_rel.dot(contact.contact_normal);
                float reduced_mass = (bodies[idx1].mass * bodies[idx2].mass) / 
                                   (bodies[idx1].mass + bodies[idx2].mass);
                float energy_lost = 0.5f * reduced_mass * v_normal * v_normal * 
                                   (1.0f - Config::Collision::RESTITUTION * Config::Collision::RESTITUTION);
                
                total_energy_lost += energy_lost;
                
                // Heat generation from collision
                #pragma omp atomic
                bodies[idx1].temperature += energy_lost / bodies[idx1].mass;
                #pragma omp atomic
                bodies[idx2].temperature += energy_lost / bodies[idx2].mass;
            }
        }
    }
    
    int get_collision_count() const { return collision_pairs.size(); }
    float get_energy_lost() const { return total_energy_lost; }
    void reset_energy_tracking() { total_energy_lost = 0; }
};

// ============================================================================
// Physics Engine with Collisions
// ============================================================================

class PhysicsEngine {
private:
    CollisionSystem collision_system;
    
    void compute_gravity(std::vector<CelestialBody>& bodies) {
        #pragma omp parallel for
        for (size_t i = 0; i < bodies.size(); i++) {
            bodies[i].force = {0, 0};
            
            for (size_t j = 0; j < bodies.size(); j++) {
                if (i == j) continue;
                
                Vector2 r = bodies[j].position - bodies[i].position;
                float dist_sq = r.length_squared() + Config::Simulation::SOFTENING;
                float dist = std::sqrt(dist_sq);
                
                float force_mag = Config::Simulation::G * bodies[i].mass * bodies[j].mass / dist_sq;
                bodies[i].force += r.normalized() * force_mag;
            }
        }
    }
    
public:
    void initialize(std::vector<CelestialBody>& bodies) {
        collision_system.initialize(bodies);
    }
    
    void integrate(std::vector<CelestialBody>& bodies, float dt) {
        // Compute all forces
        compute_gravity(bodies);
        collision_system.update(bodies, dt);
        
        // Velocity Verlet integration
        #pragma omp parallel for
        for (size_t i = 0; i < bodies.size(); i++) {
            // Update velocity (half step)
            Vector2 accel = bodies[i].force / bodies[i].mass;
            bodies[i].velocity += accel * (dt * 0.5f);
            
            // Update position
            bodies[i].position += bodies[i].velocity * dt;
            
            // Cool down temperature
            bodies[i].temperature *= 0.99f;
        }
        
        // Recompute forces at new positions
        compute_gravity(bodies);
        collision_system.update(bodies, dt);
        
        // Update velocity (second half step)
        #pragma omp parallel for
        for (size_t i = 0; i < bodies.size(); i++) {
            Vector2 accel = bodies[i].force / bodies[i].mass;
            bodies[i].velocity += accel * (dt * 0.5f);
        }
    }
    
    int get_collision_count() const { return collision_system.get_collision_count(); }
    float get_energy_lost() const { return collision_system.get_energy_lost(); }
};

// ============================================================================
// System Builder with Collision Scenario
// ============================================================================

class SystemBuilder {
public:
    void build_collision_scenario(std::vector<CelestialBody>& bodies) {
        bodies.clear();
        std::mt19937 rng(42);
        
        // Sun
        CelestialBody sun;
        sun.position = {0, 0};
        sun.velocity = {0, 0};
        sun.mass = 1.0f;
        sun.radius = 0.05f;
        sun.name = "Sun";
        sun.symbol = '*';
        bodies.push_back(sun);
        
        // Mercury
        bodies.push_back(create_planet("Mercury", 0.387f, 0.0000165f, 0.002f, '.'));
        
        // Venus
        bodies.push_back(create_planet("Venus", 0.723f, 0.0000245f, 0.006f, 'v'));
        
        // Earth with Moon
        auto earth = create_planet("Earth", 1.0f, 0.00003f, 0.006f, 'E');
        bodies.push_back(earth);
        
        // Add Moon
        CelestialBody moon = create_planet("Moon", 1.0026f, 0.0000000369f, 0.0017f, 'm');
        moon.velocity = earth.velocity;  // Start with Earth's velocity
        bodies.push_back(moon);
        
        // === COLLISION SCENARIO: Two planets in Mars orbit ===
        
        // Mars-A: Original position
        float mars_orbit = 1.524f;
        float mars_velocity = std::sqrt(Config::Simulation::G / mars_orbit);
        
        CelestialBody marsA;
        marsA.position = {mars_orbit, 0};
        marsA.velocity = {0, mars_velocity};
        marsA.mass = 0.0000033f;  // Mars mass
        marsA.radius = 0.01f;      // Larger radius for better collision
        marsA.name = "Mars-A";
        marsA.symbol = 'A';
        bodies.push_back(marsA);
        
        // Mars-B: 60 degrees behind, will catch up
        float angle = -60.0f * M_PI / 180.0f;  // Behind Mars-A
        CelestialBody marsB;
        marsB.position = {mars_orbit * std::cos(angle), mars_orbit * std::sin(angle)};
        marsB.velocity = {-mars_velocity * std::sin(angle), mars_velocity * std::cos(angle)};
        marsB.mass = 0.0000033f;  // Same mass
        marsB.radius = 0.01f;      // Same radius
        marsB.name = "Mars-B";
        marsB.symbol = 'B';
        
        // Give Mars-B slightly higher velocity to catch up
        marsB.velocity = marsB.velocity * 1.02f;  // 2% faster
        bodies.push_back(marsB);
        
        // Jupiter
        bodies.push_back(create_planet("Jupiter", 5.203f, 0.000954f, 0.05f, 'J'));
        
        // Saturn with rings
        auto saturn = create_planet("Saturn", 9.537f, 0.000286f, 0.04f, 'S');
        bodies.push_back(saturn);
        
        // Create Saturn's rings
        std::uniform_real_distribution<float> ring_dist(0.08f, 0.15f);
        std::uniform_real_distribution<float> angle_dist(0, 2 * M_PI);
        
        for (int i = 0; i < 1000; i++) {  // Fewer particles for performance
            float r = saturn.radius + ring_dist(rng);
            float theta = angle_dist(rng);
            
            float x = saturn.position.x + r * std::cos(theta);
            float y = saturn.position.y + r * std::sin(theta);
            
            // Orbital velocity for ring particle
            float v_orbit = std::sqrt(Config::Simulation::G * saturn.mass / r);
            float vx = saturn.velocity.x - v_orbit * std::sin(theta);
            float vy = saturn.velocity.y + v_orbit * std::cos(theta);
            
            CelestialBody ring_particle;
            ring_particle.position = {x, y};
            ring_particle.velocity = {vx, vy};
            ring_particle.mass = 1e-20f;
            ring_particle.radius = 0.0001f;
            ring_particle.symbol = ':';
            bodies.push_back(ring_particle);
        }
        
        // Add some asteroids
        std::uniform_real_distribution<float> ast_a(2.2f, 3.3f);
        std::uniform_real_distribution<float> ast_angle(0, 2 * M_PI);
        
        for (int i = 0; i < 500; i++) {
            float a = ast_a(rng);
            float theta = ast_angle(rng);
            float r = a;
            
            float x = r * std::cos(theta);
            float y = r * std::sin(theta);
            
            float v = std::sqrt(Config::Simulation::G / r);
            float vx = -v * std::sin(theta);
            float vy = v * std::cos(theta);
            
            CelestialBody asteroid;
            asteroid.position = {x, y};
            asteroid.velocity = {vx, vy};
            asteroid.mass = 1e-12f;
            asteroid.radius = 0.0001f;
            asteroid.symbol = '.';
            bodies.push_back(asteroid);
        }
        
        std::cout << "System built with " << bodies.size() << " bodies\n";
        std::cout << "Mars-A and Mars-B on collision course!\n";
    }
    
private:
    CelestialBody create_planet(const std::string& name, float au, float mass, 
                                float radius, char symbol) {
        float v = std::sqrt(Config::Simulation::G / au);
        CelestialBody planet;
        planet.position = {au, 0};
        planet.velocity = {0, v};
        planet.mass = mass;
        planet.radius = radius;
        planet.name = name;
        planet.symbol = symbol;
        return planet;
    }
};

// ============================================================================
// Visualization
// ============================================================================

class Visualizer {
private:
    std::vector<std::vector<char>> display;
    Vector2 camera_pos;
    float zoom;
    
public:
    Visualizer() : zoom(Config::Display::DEFAULT_ZOOM), camera_pos({0, 0}) {
        display.resize(Config::Display::SCREEN_HEIGHT,
                      std::vector<char>(Config::Display::SCREEN_WIDTH, ' '));
    }
    
    void render(const std::vector<CelestialBody>& bodies, const PhysicsEngine& physics) {
        // Clear display
        for (auto& row : display) {
            std::fill(row.begin(), row.end(), ' ');
        }
        
        // Draw bodies
        for (const auto& body : bodies) {
            int sx = (body.position.x - camera_pos.x) * zoom + Config::Display::SCREEN_WIDTH / 2;
            int sy = (body.position.y - camera_pos.y) * zoom + Config::Display::SCREEN_HEIGHT / 2;
            
            if (sx >= 0 && sx < Config::Display::SCREEN_WIDTH &&
                sy >= 0 && sy < Config::Display::SCREEN_HEIGHT) {
                
                // Show collision state
                if (body.is_colliding) {
                    display[sy][sx] = 'X';  // Collision marker
                } else if (body.temperature > 0.1f) {
                    display[sy][sx] = 'o';  // Hot from recent collision
                } else {
                    display[sy][sx] = body.symbol;
                }
            }
        }
        
        // Draw to terminal
        std::cout << "\033[H";
        
        // Header
        std::cout << "+" << std::string(Config::Display::SCREEN_WIDTH, '-') << "+\n";
        
        // Display grid
        for (const auto& row : display) {
            std::cout << "|";
            for (char c : row) {
                std::cout << c;
            }
            std::cout << "|\n";
        }
        
        // Footer
        std::cout << "+" << std::string(Config::Display::SCREEN_WIDTH, '-') << "+\n";
        
        // Stats
        std::cout << "Collisions: " << physics.get_collision_count() 
                  << " | Energy Lost: " << std::fixed << std::setprecision(6) 
                  << physics.get_energy_lost()
                  << " | Zoom: " << zoom
                  << " | X: Mars collision! " << std::string(20, ' ') << "\n";
    }
    
    void set_camera(const Vector2& pos) { camera_pos = pos; }
    void set_zoom(float z) { zoom = z; }
};

// ============================================================================
// Main Simulation
// ============================================================================

class Simulation {
private:
    std::vector<CelestialBody> bodies;
    PhysicsEngine physics;
    SystemBuilder builder;
    Visualizer visualizer;
    float time;
    float dt;
    
    int find_body(const std::string& name) {
        for (size_t i = 0; i < bodies.size(); i++) {
            if (bodies[i].name == name) return i;
        }
        return -1;
    }
    
public:
    Simulation() : time(0), dt(Config::Simulation::TIME_STEP) {}
    
    void initialize() {
        builder.build_collision_scenario(bodies);
        physics.initialize(bodies);
        
        // Focus on Mars orbit
        visualizer.set_zoom(5.0f);
        visualizer.set_camera({1.5f, 0});  // Mars orbit region
    }
    
    void run(bool test_mode = false) {
        std::cout << "\033[2J\033[H";  // Clear screen
        
        int frame = 0;
        int max_frames = test_mode ? 500 : 100000;  // Limit frames in test mode
        auto last_frame = std::chrono::steady_clock::now();
        
        // Track Mars collision
        int marsA_idx = find_body("Mars-A");
        int marsB_idx = find_body("Mars-B");
        bool collision_happened = false;
        float min_distance = 1000.0f;
        
        while (frame < max_frames) {
            // Physics update
            physics.integrate(bodies, dt);
            time += dt;
            
            // Check Mars distance
            if (marsA_idx >= 0 && marsB_idx >= 0) {
                Vector2 diff = bodies[marsB_idx].position - bodies[marsA_idx].position;
                float dist = diff.length();
                min_distance = std::min(min_distance, dist);
                
                if (dist < 0.02f && !collision_happened) {
                    collision_happened = true;
                    if (!test_mode) {
                        std::cout << "\a";  // Beep on collision
                    }
                }
            }
            
            // Render
            frame++;
            if (frame % Config::Display::FRAME_SKIP == 0) {
                if (!test_mode) {
                    auto now = std::chrono::steady_clock::now();
                    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
                        now - last_frame).count();
                    
                    if (elapsed < 1000 / Config::Display::TARGET_FPS) {
                        std::this_thread::sleep_for(
                            std::chrono::milliseconds(1000 / Config::Display::TARGET_FPS - elapsed));
                    }
                    
                    visualizer.render(bodies, physics);
                    last_frame = std::chrono::steady_clock::now();
                }
                
                // Exit on collision merge
                if (collision_happened && frame > 1000) {
                    std::cout << "\nMars-A and Mars-B have collided!\n";
                    std::cout << "Simulation time: " << time << " years\n";
                    break;
                }
            }
        }
        
        if (test_mode) {
            std::cout << "\nTest Results:\n";
            std::cout << "Frames simulated: " << frame << "\n";
            std::cout << "Simulation time: " << time << " years\n";
            std::cout << "Min Mars distance: " << min_distance << " AU\n";
            std::cout << "Collision happened: " << (collision_happened ? "Yes" : "No") << "\n";
            std::cout << "Total collisions detected: " << physics.get_collision_count() << "\n";
            std::cout << "Energy lost to collisions: " << physics.get_energy_lost() << "\n";
        }
    }
};

// ============================================================================
// Main Function
// ============================================================================

int main(int argc, char* argv[]) {
    bool test_mode = (argc > 1 && std::string(argv[1]) == "--test");
    
    std::cout << "=== Solar System with Soft Contact Collisions ===\n\n";
    std::cout << "Features:\n";
    std::cout << "- Sparse spatial grid for collision detection\n";
    std::cout << "- Hertzian soft contact model\n";
    std::cout << "- Mars-A and Mars-B on collision course\n";
    std::cout << "- Energy dissipation and heat generation\n\n";
    
    if (test_mode) {
        std::cout << "Running in test mode (quick verification)...\n";
    } else {
        std::cout << "Watch for Mars collision (marked with 'X')!\n\n";
        std::cout << "Starting in 3 seconds...\n";
        std::this_thread::sleep_for(std::chrono::seconds(3));
    }
    
    Simulation sim;
    sim.initialize();
    sim.run(test_mode);
    
    return 0;
}
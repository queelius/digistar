/**
 * @file contact_test.cpp
 * @brief Simple two-body collision test with soft contact forces
 * 
 * Two objects separated by 100 meters, moving towards each other at 20 m/s
 * Demonstrates Hertzian contact model and energy dissipation
 */

#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>
#include <chrono>
#include <thread>
#include <algorithm>

// ============================================================================
// Configuration
// ============================================================================

namespace Config {
    // Use meters and seconds for this test
    constexpr float INITIAL_SEPARATION = 100.0f;  // meters
    constexpr float APPROACH_SPEED = 20.0f;       // m/s
    constexpr float OBJECT_RADIUS = 5.0f;         // meters
    constexpr float OBJECT_MASS = 1000.0f;        // kg
    
    // Collision parameters
    constexpr float CONTACT_STIFFNESS = 1000000.0f;  // N/m^1.5 (Hertzian) - increased for less penetration
    constexpr float CONTACT_DAMPING = 5000.0f;       // N⋅s/m - increased damping
    constexpr float RESTITUTION = 0.6f;              // 0 = perfectly inelastic, 1 = perfectly elastic
    
    // Simulation
    constexpr float TIME_STEP = 0.0001f;  // seconds (needs to be small for stiff contacts)
    constexpr float SIMULATION_TIME = 10.0f;  // seconds
    
    // Display
    constexpr int SCREEN_WIDTH = 120;
    constexpr int SCREEN_HEIGHT = 30;
    constexpr float WORLD_WIDTH = 200.0f;  // meters
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
    
    float length() const { return std::sqrt(x * x + y * y); }
    Vector2 normalized() const { 
        float len = length();
        return len > 0 ? (*this) / len : Vector2{0, 0};
    }
    float dot(const Vector2& other) const { return x * other.x + y * other.y; }
};

struct Body {
    Vector2 position;
    Vector2 velocity;
    Vector2 force;
    float mass;
    float radius;
    std::string name;
    
    // Collision state
    bool is_colliding;
    float max_force_experienced;
    float total_energy;
    
    Body(const std::string& n, Vector2 pos, Vector2 vel, float m, float r) 
        : name(n), position(pos), velocity(vel), mass(m), radius(r),
          force{0, 0}, is_colliding(false), max_force_experienced(0) {
        total_energy = 0.5f * mass * velocity.length() * velocity.length();
    }
    
    void update_energy() {
        total_energy = 0.5f * mass * velocity.length() * velocity.length();
    }
};

// ============================================================================
// Contact Force Calculation
// ============================================================================

struct ContactInfo {
    Vector2 force;
    float penetration_depth;
    Vector2 contact_normal;
    float force_magnitude;
    bool active;
};

ContactInfo calculate_contact(const Body& b1, const Body& b2) {
    ContactInfo contact = {{0, 0}, 0, {0, 0}, 0, false};
    
    Vector2 delta = b2.position - b1.position;
    float dist = delta.length();
    float min_dist = b1.radius + b2.radius;
    
    if (dist < min_dist && dist > 0) {
        contact.active = true;
        
        // Penetration depth
        contact.penetration_depth = min_dist - dist;
        contact.contact_normal = delta.normalized();
        
        // Hertzian contact model for spheres
        // F = k * δ^(3/2) where δ is penetration depth
        float effective_radius = (b1.radius * b2.radius) / (b1.radius + b2.radius);
        float stiffness = Config::CONTACT_STIFFNESS * std::sqrt(effective_radius);
        contact.force_magnitude = stiffness * std::pow(contact.penetration_depth, 1.5f);
        
        // Add velocity-dependent damping
        Vector2 v_rel = b2.velocity - b1.velocity;
        float v_normal = v_rel.dot(contact.contact_normal);
        
        // Only add damping if bodies are approaching (v_normal < 0)
        if (v_normal < 0) {
            float damping_force = Config::CONTACT_DAMPING * std::abs(v_normal);
            contact.force_magnitude += damping_force;
        }
        
        contact.force = contact.contact_normal * contact.force_magnitude;
    }
    
    return contact;
}

// ============================================================================
// Visualization
// ============================================================================

class Visualizer {
private:
    std::vector<std::vector<char>> display;
    std::vector<std::pair<float, float>> position_history;
    
public:
    Visualizer() {
        display.resize(Config::SCREEN_HEIGHT, 
                      std::vector<char>(Config::SCREEN_WIDTH, ' '));
    }
    
    void render(const std::vector<Body>& bodies, float time, const ContactInfo& contact) {
        // Clear display
        for (auto& row : display) {
            std::fill(row.begin(), row.end(), ' ');
        }
        
        // Draw ground line
        int ground_y = Config::SCREEN_HEIGHT / 2;
        for (int x = 0; x < Config::SCREEN_WIDTH; x++) {
            display[ground_y][x] = '-';
        }
        
        // Store positions for trail
        if (bodies.size() >= 2) {
            float separation = (bodies[1].position.x - bodies[0].position.x);
            position_history.push_back({bodies[0].position.x, bodies[1].position.x});
            if (position_history.size() > 100) {
                position_history.erase(position_history.begin());
            }
        }
        
        // Draw position trail
        for (size_t i = 0; i < position_history.size(); i++) {
            float alpha = i / (float)position_history.size();
            char trail_char = (alpha < 0.3f) ? '.' : (alpha < 0.7f) ? ':' : '=';
            
            int x1 = (int)((position_history[i].first + Config::WORLD_WIDTH/2) / Config::WORLD_WIDTH * Config::SCREEN_WIDTH);
            int x2 = (int)((position_history[i].second + Config::WORLD_WIDTH/2) / Config::WORLD_WIDTH * Config::SCREEN_WIDTH);
            
            if (x1 >= 0 && x1 < Config::SCREEN_WIDTH) {
                display[ground_y - 1][x1] = trail_char;
            }
            if (x2 >= 0 && x2 < Config::SCREEN_WIDTH) {
                display[ground_y + 1][x2] = trail_char;
            }
        }
        
        // Draw bodies
        for (const auto& body : bodies) {
            int x = (int)((body.position.x + Config::WORLD_WIDTH/2) / Config::WORLD_WIDTH * Config::SCREEN_WIDTH);
            int y = ground_y;
            
            // Offset vertically for visibility
            if (body.name == "Body-1") y -= 2;
            if (body.name == "Body-2") y += 2;
            
            // Draw body with size
            int size = (int)(body.radius / Config::WORLD_WIDTH * Config::SCREEN_WIDTH);
            for (int dx = -size; dx <= size; dx++) {
                int px = x + dx;
                if (px >= 0 && px < Config::SCREEN_WIDTH && y >= 0 && y < Config::SCREEN_HEIGHT) {
                    if (body.is_colliding) {
                        display[y][px] = 'X';  // Collision!
                    } else {
                        display[y][px] = (body.name == "Body-1") ? 'A' : 'B';
                    }
                }
            }
        }
        
        // Draw collision force indicator
        if (contact.active) {
            int cx = Config::SCREEN_WIDTH / 2;
            int cy = ground_y;
            
            // Draw impact star
            if (cy > 0 && cy < Config::SCREEN_HEIGHT - 1) {
                display[cy][cx] = '*';
                if (cx > 0) display[cy][cx-1] = '<';
                if (cx < Config::SCREEN_WIDTH - 1) display[cy][cx+1] = '>';
                display[cy-1][cx] = '^';
                display[cy+1][cx] = 'v';
            }
        }
        
        // Draw to terminal
        std::cout << "\033[H";  // Home cursor
        
        // Header
        std::cout << "+" << std::string(Config::SCREEN_WIDTH, '-') << "+\n";
        
        // Display grid
        for (const auto& row : display) {
            std::cout << "|";
            for (char c : row) {
                std::cout << c;
            }
            std::cout << "|\n";
        }
        
        // Footer
        std::cout << "+" << std::string(Config::SCREEN_WIDTH, '-') << "+\n";
        
        // Information panel
        std::cout << std::fixed << std::setprecision(3);
        std::cout << "Time: " << std::setw(7) << time << "s | ";
        
        if (bodies.size() >= 2) {
            float separation = bodies[1].position.x - bodies[0].position.x;
            float rel_velocity = bodies[1].velocity.x - bodies[0].velocity.x;
            
            std::cout << "Separation: " << std::setw(8) << separation << "m | ";
            std::cout << "Rel.Vel: " << std::setw(8) << rel_velocity << "m/s | ";
            
            if (contact.active) {
                std::cout << "CONTACT! Force: " << std::scientific << std::setprecision(2) 
                          << contact.force_magnitude << "N ";
                std::cout << "Penetration: " << std::fixed << std::setprecision(3) 
                          << contact.penetration_depth << "m";
            } else {
                std::cout << "No contact" << std::string(40, ' ');
            }
        }
        std::cout << "\n";
        
        // Energy tracking
        if (bodies.size() >= 2) {
            float total_ke = bodies[0].total_energy + bodies[1].total_energy;
            float initial_ke = 2 * 0.5f * Config::OBJECT_MASS * Config::APPROACH_SPEED * Config::APPROACH_SPEED;
            float energy_lost = initial_ke - total_ke;
            
            std::cout << "KE: " << std::fixed << std::setprecision(1) << total_ke << "J | ";
            std::cout << "Energy Lost: " << energy_lost << "J | ";
            std::cout << "Max Force: " << std::scientific << std::setprecision(2) 
                      << std::max(bodies[0].max_force_experienced, bodies[1].max_force_experienced) << "N";
        }
        std::cout << std::string(30, ' ') << "\n";
    }
};

// ============================================================================
// Simulation
// ============================================================================

class Simulation {
private:
    std::vector<Body> bodies;
    Visualizer visualizer;
    float time;
    float dt;
    
public:
    Simulation() : time(0), dt(Config::TIME_STEP) {}
    
    void initialize() {
        // Body 1: Moving right
        bodies.emplace_back(
            "Body-1",
            Vector2{-Config::INITIAL_SEPARATION/2, 0},
            Vector2{Config::APPROACH_SPEED, 0},
            Config::OBJECT_MASS,
            Config::OBJECT_RADIUS
        );
        
        // Body 2: Moving left
        bodies.emplace_back(
            "Body-2",
            Vector2{Config::INITIAL_SEPARATION/2, 0},
            Vector2{-Config::APPROACH_SPEED, 0},
            Config::OBJECT_MASS,
            Config::OBJECT_RADIUS
        );
        
        std::cout << "\nInitial conditions:\n";
        std::cout << "  Separation: " << Config::INITIAL_SEPARATION << " m\n";
        std::cout << "  Approach speed: " << Config::APPROACH_SPEED << " m/s each\n";
        std::cout << "  Relative velocity: " << (2 * Config::APPROACH_SPEED) << " m/s\n";
        std::cout << "  Object mass: " << Config::OBJECT_MASS << " kg\n";
        std::cout << "  Object radius: " << Config::OBJECT_RADIUS << " m\n";
        std::cout << "  Contact stiffness: " << Config::CONTACT_STIFFNESS << " N/m^1.5\n";
        std::cout << "  Restitution: " << Config::RESTITUTION << "\n\n";
    }
    
    void run() {
        std::cout << "\033[2J\033[H";  // Clear screen
        
        int frame = 0;
        int display_counter = 0;
        auto last_frame = std::chrono::steady_clock::now();
        
        // Track collision state
        bool collision_started = false;
        bool collision_ended = false;
        float collision_start_time = 0;
        float collision_end_time = 0;
        float min_separation = Config::INITIAL_SEPARATION;
        
        while (time < Config::SIMULATION_TIME) {
            // Calculate contact forces
            ContactInfo contact = calculate_contact(bodies[0], bodies[1]);
            
            // Reset forces
            for (auto& body : bodies) {
                body.force = {0, 0};
                body.is_colliding = false;
            }
            
            // Apply contact forces
            if (contact.active) {
                bodies[0].force = bodies[0].force - contact.force;
                bodies[1].force = bodies[1].force + contact.force;
                bodies[0].is_colliding = true;
                bodies[1].is_colliding = true;
                
                // Track max force
                bodies[0].max_force_experienced = std::max(bodies[0].max_force_experienced, contact.force_magnitude);
                bodies[1].max_force_experienced = std::max(bodies[1].max_force_experienced, contact.force_magnitude);
                
                if (!collision_started) {
                    collision_started = true;
                    collision_start_time = time;
                }
            } else if (collision_started && !collision_ended) {
                collision_ended = true;
                collision_end_time = time;
            }
            
            // Integrate using semi-implicit Euler (stable for stiff contacts)
            for (auto& body : bodies) {
                // Update velocity
                Vector2 acceleration = body.force / body.mass;
                body.velocity += acceleration * dt;
                
                // Update position
                body.position += body.velocity * dt;
                
                // Update energy
                body.update_energy();
            }
            
            // Track minimum separation
            float separation = (bodies[1].position - bodies[0].position).length();
            min_separation = std::min(min_separation, separation);
            
            // Update time
            time += dt;
            frame++;
            
            // Display at target FPS
            display_counter++;
            if (display_counter >= (int)(1.0f / (Config::FPS * dt))) {
                display_counter = 0;
                
                auto now = std::chrono::steady_clock::now();
                auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - last_frame).count();
                
                if (elapsed < 1000 / Config::FPS) {
                    std::this_thread::sleep_for(
                        std::chrono::milliseconds(1000 / Config::FPS - elapsed));
                }
                
                visualizer.render(bodies, time, contact);
                last_frame = std::chrono::steady_clock::now();
            }
            
            // End simulation after bodies separate significantly
            if (collision_ended) {
                float current_separation = bodies[1].position.x - bodies[0].position.x;
                if (current_separation > Config::INITIAL_SEPARATION * 0.8f) {
                    break;
                }
            }
        }
        
        // Final statistics
        std::cout << "\n\n=== Collision Summary ===\n";
        if (collision_started) {
            std::cout << "Collision occurred at t = " << collision_start_time << " s\n";
            if (collision_ended) {
                std::cout << "Collision ended at t = " << collision_end_time << " s\n";
                std::cout << "Contact duration: " << (collision_end_time - collision_start_time) * 1000 << " ms\n";
            }
            
            std::cout << "Minimum separation: " << min_separation << " m\n";
            std::cout << "Maximum penetration: " << (bodies[0].radius + bodies[1].radius - min_separation) << " m\n";
            
            // Calculate coefficient of restitution from velocities
            float v1_final = bodies[0].velocity.x;
            float v2_final = bodies[1].velocity.x;
            float v_sep = std::abs(v2_final - v1_final);
            float v_app = 2 * Config::APPROACH_SPEED;
            float measured_restitution = v_sep / v_app;
            
            std::cout << "\nFinal velocities:\n";
            std::cout << "  Body-1: " << v1_final << " m/s\n";
            std::cout << "  Body-2: " << v2_final << " m/s\n";
            std::cout << "  Separation velocity: " << v_sep << " m/s\n";
            std::cout << "  Measured restitution: " << measured_restitution 
                      << " (configured: " << Config::RESTITUTION << ")\n";
            
            float initial_ke = 2 * 0.5f * Config::OBJECT_MASS * Config::APPROACH_SPEED * Config::APPROACH_SPEED;
            float final_ke = bodies[0].total_energy + bodies[1].total_energy;
            float energy_lost = initial_ke - final_ke;
            
            std::cout << "\nEnergy analysis:\n";
            std::cout << "  Initial KE: " << initial_ke << " J\n";
            std::cout << "  Final KE: " << final_ke << " J\n";
            std::cout << "  Energy lost: " << energy_lost << " J (" 
                      << (100 * energy_lost / initial_ke) << "%)\n";
            
            std::cout << "\nMaximum contact force: " << std::scientific << std::setprecision(3) 
                      << bodies[0].max_force_experienced << " N\n";
        } else {
            std::cout << "No collision occurred (bodies did not make contact)\n";
        }
    }
};

// ============================================================================
// Main
// ============================================================================

int main() {
    std::cout << "=== Soft Contact Force Test ===\n";
    std::cout << "Two bodies approaching head-on\n";
    std::cout << "Hertzian contact model with damping\n\n";
    
    std::cout << "Controls: Press Ctrl+C to exit\n\n";
    std::cout << "Starting simulation in 2 seconds...\n";
    std::this_thread::sleep_for(std::chrono::seconds(2));
    
    Simulation sim;
    sim.initialize();
    sim.run();
    
    return 0;
}
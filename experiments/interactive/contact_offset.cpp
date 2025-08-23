/**
 * @file contact_offset.cpp
 * @brief Offset collision test - demonstrates angular deflection
 * 
 * Two objects with vertical offset, showing 2D collision dynamics
 */

#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>
#include <chrono>
#include <thread>
#include <algorithm>

// Configuration
namespace Config {
    constexpr float INITIAL_SEPARATION = 100.0f;  // meters
    constexpr float APPROACH_SPEED = 20.0f;       // m/s
    constexpr float VERTICAL_OFFSET = 8.0f;       // meters (partial overlap)
    constexpr float OBJECT_RADIUS = 5.0f;         // meters
    constexpr float OBJECT_MASS = 1000.0f;        // kg
    
    constexpr float CONTACT_STIFFNESS = 1000000.0f;
    constexpr float CONTACT_DAMPING = 5000.0f;
    constexpr float TIME_STEP = 0.0001f;
    constexpr float SIMULATION_TIME = 5.0f;
    
    constexpr int SCREEN_WIDTH = 120;
    constexpr int SCREEN_HEIGHT = 40;
    constexpr float WORLD_WIDTH = 200.0f;
    constexpr float WORLD_HEIGHT = 80.0f;
    constexpr int FPS = 30;
}

// Vector2 structure
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
    bool is_colliding;
    
    Body(const std::string& n, Vector2 pos, Vector2 vel, float m, float r) 
        : name(n), position(pos), velocity(vel), mass(m), radius(r),
          force{0, 0}, is_colliding(false) {}
};

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
        contact.penetration_depth = min_dist - dist;
        contact.contact_normal = delta.normalized();
        
        // Hertzian contact
        float effective_radius = (b1.radius * b2.radius) / (b1.radius + b2.radius);
        float stiffness = Config::CONTACT_STIFFNESS * std::sqrt(effective_radius);
        contact.force_magnitude = stiffness * std::pow(contact.penetration_depth, 1.5f);
        
        // Damping
        Vector2 v_rel = b2.velocity - b1.velocity;
        float v_normal = v_rel.dot(contact.contact_normal);
        if (v_normal < 0) {
            contact.force_magnitude += Config::CONTACT_DAMPING * std::abs(v_normal);
        }
        
        contact.force = contact.contact_normal * contact.force_magnitude;
    }
    
    return contact;
}

class Visualizer {
private:
    std::vector<std::vector<char>> display;
    std::vector<Vector2> trail1, trail2;
    
public:
    Visualizer() {
        display.resize(Config::SCREEN_HEIGHT, 
                      std::vector<char>(Config::SCREEN_WIDTH, ' '));
    }
    
    void render(const std::vector<Body>& bodies, float time) {
        // Clear
        for (auto& row : display) {
            std::fill(row.begin(), row.end(), ' ');
        }
        
        // Add to trails
        if (bodies.size() >= 2) {
            trail1.push_back(bodies[0].position);
            trail2.push_back(bodies[1].position);
            if (trail1.size() > 200) trail1.erase(trail1.begin());
            if (trail2.size() > 200) trail2.erase(trail2.begin());
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
        
        // Draw trails
        auto draw_trail = [&](const std::vector<Vector2>& trail, char symbol) {
            for (size_t i = 0; i < trail.size(); i++) {
                float alpha = i / (float)trail.size();
                char c = (alpha < 0.3f) ? '.' : (alpha < 0.7f) ? ':' : symbol;
                
                int x = (int)((trail[i].x + Config::WORLD_WIDTH/2) / Config::WORLD_WIDTH * Config::SCREEN_WIDTH);
                int y = (int)((Config::WORLD_HEIGHT/2 - trail[i].y) / Config::WORLD_HEIGHT * Config::SCREEN_HEIGHT);
                
                if (x >= 0 && x < Config::SCREEN_WIDTH && y >= 0 && y < Config::SCREEN_HEIGHT) {
                    if (display[y][x] == ' ' || display[y][x] == '-' || display[y][x] == '|') {
                        display[y][x] = c;
                    }
                }
            }
        };
        
        draw_trail(trail1, 'a');
        draw_trail(trail2, 'b');
        
        // Draw bodies
        for (const auto& body : bodies) {
            int cx = (int)((body.position.x + Config::WORLD_WIDTH/2) / Config::WORLD_WIDTH * Config::SCREEN_WIDTH);
            int cy = (int)((Config::WORLD_HEIGHT/2 - body.position.y) / Config::WORLD_HEIGHT * Config::SCREEN_HEIGHT);
            
            // Draw circle approximation
            int r_screen = (int)(body.radius / Config::WORLD_WIDTH * Config::SCREEN_WIDTH);
            for (int dy = -r_screen; dy <= r_screen; dy++) {
                for (int dx = -r_screen; dx <= r_screen; dx++) {
                    if (dx*dx + dy*dy <= r_screen*r_screen) {
                        int px = cx + dx;
                        int py = cy + dy;
                        if (px >= 0 && px < Config::SCREEN_WIDTH && py >= 0 && py < Config::SCREEN_HEIGHT) {
                            char c = body.is_colliding ? 'X' : (body.name == "Body-1" ? 'A' : 'B');
                            display[py][px] = c;
                        }
                    }
                }
            }
        }
        
        // Draw
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
        if (bodies.size() >= 2) {
            std::cout << "Pos1: (" << bodies[0].position.x << "," << bodies[0].position.y << ") | ";
            std::cout << "Pos2: (" << bodies[1].position.x << "," << bodies[1].position.y << ") | ";
            std::cout << "Vel1: (" << bodies[0].velocity.x << "," << bodies[0].velocity.y << ") | ";
            std::cout << "Vel2: (" << bodies[1].velocity.x << "," << bodies[1].velocity.y << ")";
        }
        std::cout << "\n";
    }
};

class Simulation {
private:
    std::vector<Body> bodies;
    Visualizer visualizer;
    float time;
    float dt;
    
public:
    Simulation() : time(0), dt(Config::TIME_STEP) {}
    
    void initialize() {
        // Body 1: Moving right, slightly up
        bodies.emplace_back(
            "Body-1",
            Vector2{-Config::INITIAL_SEPARATION/2, -Config::VERTICAL_OFFSET/2},
            Vector2{Config::APPROACH_SPEED, 0},
            Config::OBJECT_MASS,
            Config::OBJECT_RADIUS
        );
        
        // Body 2: Moving left, slightly down
        bodies.emplace_back(
            "Body-2",
            Vector2{Config::INITIAL_SEPARATION/2, Config::VERTICAL_OFFSET/2},
            Vector2{-Config::APPROACH_SPEED, 0},
            Config::OBJECT_MASS,
            Config::OBJECT_RADIUS
        );
        
        std::cout << "\nOffset Collision Setup:\n";
        std::cout << "  Horizontal separation: " << Config::INITIAL_SEPARATION << " m\n";
        std::cout << "  Vertical offset: " << Config::VERTICAL_OFFSET << " m\n";
        std::cout << "  Impact parameter: " << Config::VERTICAL_OFFSET << " m\n";
        std::cout << "  Expected: Glancing collision with angular deflection\n\n";
    }
    
    void run() {
        std::cout << "\033[2J\033[H";
        
        int frame = 0;
        int display_counter = 0;
        auto last_frame = std::chrono::steady_clock::now();
        
        while (time < Config::SIMULATION_TIME) {
            // Calculate contact
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
            }
            
            // Integrate
            for (auto& body : bodies) {
                Vector2 acceleration = body.force / body.mass;
                body.velocity += acceleration * dt;
                body.position += body.velocity * dt;
            }
            
            time += dt;
            frame++;
            
            // Display
            display_counter++;
            if (display_counter >= (int)(1.0f / (Config::FPS * dt))) {
                display_counter = 0;
                
                auto now = std::chrono::steady_clock::now();
                auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - last_frame).count();
                if (elapsed < 1000 / Config::FPS) {
                    std::this_thread::sleep_for(std::chrono::milliseconds(1000 / Config::FPS - elapsed));
                }
                
                visualizer.render(bodies, time);
                last_frame = std::chrono::steady_clock::now();
            }
        }
        
        // Summary
        std::cout << "\n=== Offset Collision Results ===\n";
        std::cout << "Body-1 final velocity: (" << bodies[0].velocity.x << ", " << bodies[0].velocity.y << ") m/s\n";
        std::cout << "Body-2 final velocity: (" << bodies[1].velocity.x << ", " << bodies[1].velocity.y << ") m/s\n";
        std::cout << "\nNote the Y-component velocities from the glancing collision!\n";
        std::cout << "Body-1 deflected: " << (bodies[0].velocity.y > 0 ? "upward" : "downward") << "\n";
        std::cout << "Body-2 deflected: " << (bodies[1].velocity.y > 0 ? "upward" : "downward") << "\n";
    }
};

int main() {
    std::cout << "=== Offset Collision Test ===\n";
    std::cout << "Two bodies with vertical offset\n";
    std::cout << "Demonstrates 2D deflection from glancing collision\n\n";
    
    std::cout << "Starting in 2 seconds...\n";
    std::this_thread::sleep_for(std::chrono::seconds(2));
    
    Simulation sim;
    sim.initialize();
    sim.run();
    
    return 0;
}
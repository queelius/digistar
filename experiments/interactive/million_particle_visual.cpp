// Million-Particle Solar System with Real-Time Visualization
// Interactive ASCII visualization to debug orbital mechanics

#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>
#include <cstdlib>
#include <chrono>
#include <thread>
#include <cstring>
#include <algorithm>
#include <complex>
#include <fftw3.h>
#include <omp.h>
#include <random>
#include <termios.h>
#include <unistd.h>
#include <fcntl.h>

// Simulation units: AU, Solar masses, Years
namespace SimUnits {
    constexpr double G = 4.0 * M_PI * M_PI;  // G in AU³/M☉·year²
    constexpr double TIME_STEP = 0.0001;     // years (~0.876 hours) - smaller for stability
}

// 2D Vector
struct float2 {
    float x, y;
    
    float2() : x(0), y(0) {}
    float2(float x_, float y_) : x(x_), y(y_) {}
    
    float2 operator+(const float2& o) const { return {x + o.x, y + o.y}; }
    float2 operator-(const float2& o) const { return {x - o.x, y - o.y}; }
    float2 operator*(float s) const { return {x * s, y * s}; }
    float2 operator/(float s) const { return {x / s, y / s}; }
    float2& operator+=(const float2& o) { x += o.x; y += o.y; return *this; }
    float2& operator-=(const float2& o) { x -= o.x; y -= o.y; return *this; }
    
    float length() const { return std::sqrt(x * x + y * y); }
    float2 normalized() const { 
        float len = length();
        return len > 0 ? float2(x/len, y/len) : float2(0, 0);
    }
};

// Particle
struct Particle {
    float2 pos;
    float2 vel;
    float2 force;
    float mass;
    uint8_t type;  // 0=star, 1=planet, 2=moon, 3=asteroid, 4=KBO
    
    Particle() : pos(0, 0), vel(0, 0), force(0, 0), mass(0), type(3) {}
    Particle(float2 p, float2 v, float m, uint8_t t) 
        : pos(p), vel(v), force(0, 0), mass(m), type(t) {}
};

// Terminal input handler
class InputHandler {
    struct termios old_tio, new_tio;
    
public:
    InputHandler() {
        tcgetattr(STDIN_FILENO, &old_tio);
        new_tio = old_tio;
        new_tio.c_lflag &= (~ICANON & ~ECHO);
        tcsetattr(STDIN_FILENO, TCSANOW, &new_tio);
        fcntl(STDIN_FILENO, F_SETFL, O_NONBLOCK);
    }
    
    ~InputHandler() {
        tcsetattr(STDIN_FILENO, TCSANOW, &old_tio);
    }
    
    char get_input() {
        char c = 0;
        read(STDIN_FILENO, &c, 1);
        return c;
    }
};

// Enhanced Visualizer with camera controls
class Visualizer {
    float2 camera_pos;
    float zoom;
    int view_mode;  // 0=inner, 1=belt, 2=outer
    bool show_density;
    bool paused;
    int frame_count;
    InputHandler input;
    
    // Density grid for visualization
    static constexpr int DENSITY_GRID = 120;
    std::vector<std::vector<int>> density_map;
    
public:
    Visualizer() : camera_pos(0, 0), zoom(0.5f), view_mode(0), 
                   show_density(true), paused(false), frame_count(0) {
        density_map.resize(DENSITY_GRID, std::vector<int>(DENSITY_GRID, 0));
    }
    
    bool is_paused() const { return paused; }
    
    void handle_input() {
        char c = input.get_input();
        if (c == 0) return;
        
        switch(c) {
            // Camera controls
            case 'w': camera_pos.y += 5.0f / zoom; break;
            case 's': camera_pos.y -= 5.0f / zoom; break;
            case 'a': camera_pos.x -= 5.0f / zoom; break;
            case 'd': camera_pos.x += 5.0f / zoom; break;
            
            // Zoom
            case '+': case '=': zoom *= 1.5f; break;
            case '-': case '_': zoom /= 1.5f; break;
            
            // View modes
            case '1': view_mode = 0; zoom = 2.0f; camera_pos = float2(0, 0); break;  // Inner system
            case '2': view_mode = 1; zoom = 0.5f; camera_pos = float2(0, 0); break;  // Asteroid belt
            case '3': view_mode = 2; zoom = 0.05f; camera_pos = float2(0, 0); break; // Outer system
            
            // Options
            case 'r': camera_pos = float2(0, 0); break;  // Reset
            case 'p': paused = !paused; break;
            case 'g': show_density = !show_density; break;
            case 'q': exit(0); break;
        }
    }
    
    void display(const std::vector<Particle>& particles, float time, float dt) {
        frame_count++;
        handle_input();
        
        // Only update display every few frames for performance
        if (frame_count % 5 != 0) return;
        
        std::cout << "\033[2J\033[H";  // Clear screen
        
        // Header
        std::cout << "=== Million Particle Solar System (Real-Time) ===\n";
        std::cout << "Time: " << std::fixed << std::setprecision(3) << time << " years | ";
        std::cout << "dt: " << dt * 365.25 << " days | ";
        std::cout << "Particles: " << particles.size() << " | ";
        std::cout << (paused ? "PAUSED" : "RUNNING") << "\n";
        
        // View info
        const char* view_names[] = {"Inner System", "Asteroid Belt", "Outer System"};
        std::cout << "View: " << view_names[view_mode] << " | ";
        std::cout << "Zoom: " << std::setprecision(2) << zoom << "x | ";
        std::cout << "Camera: (" << camera_pos.x << ", " << camera_pos.y << ")\n";
        
        // Controls
        std::cout << "Controls: WASD=pan, +-=zoom, 123=views, p=pause, g=density, r=reset, q=quit\n";
        std::cout << "─────────────────────────────────────────────────────────────────────────\n";
        
        // Visualization area
        const int WIDTH = 120;
        const int HEIGHT = 35;
        std::vector<std::vector<char>> screen(HEIGHT, std::vector<char>(WIDTH, ' '));
        std::vector<std::vector<int>> z_buffer(HEIGHT, std::vector<int>(WIDTH, -1));
        
        // Clear density map
        if (show_density) {
            for (auto& row : density_map) {
                std::fill(row.begin(), row.end(), 0);
            }
        }
        
        // Draw particles
        for (size_t i = 0; i < particles.size(); i++) {
            const auto& p = particles[i];
            
            // Transform to screen coordinates
            float2 rel_pos = p.pos - camera_pos;
            int sx = WIDTH/2 + (int)(rel_pos.x * zoom);
            int sy = HEIGHT/2 - (int)(rel_pos.y * zoom * 0.5f);  // Aspect ratio
            
            // Update density map
            if (show_density && p.type >= 3) {  // Only asteroids/KBOs for density
                int dx = (int)((p.pos.x + 100) * DENSITY_GRID / 200.0f);
                int dy = (int)((p.pos.y + 100) * DENSITY_GRID / 200.0f);
                if (dx >= 0 && dx < DENSITY_GRID && dy >= 0 && dy < DENSITY_GRID) {
                    density_map[dy][dx]++;
                }
            }
            
            // Draw particle
            if (sx >= 0 && sx < WIDTH && sy >= 0 && sy < HEIGHT) {
                char symbol = ' ';
                int priority = 0;
                
                switch(p.type) {
                    case 0: symbol = '@'; priority = 10; break;  // Sun
                    case 1: symbol = 'O'; priority = 8; break;   // Planet
                    case 2: symbol = 'o'; priority = 6; break;   // Moon
                    case 3: symbol = '.'; priority = 2; break;   // Asteroid
                    case 4: symbol = ','; priority = 1; break;   // KBO
                }
                
                if (priority > z_buffer[sy][sx]) {
                    screen[sy][sx] = symbol;
                    z_buffer[sy][sx] = priority;
                }
            }
        }
        
        // Overlay density map if enabled
        if (show_density) {
            for (int y = 0; y < HEIGHT; y++) {
                for (int x = 0; x < WIDTH; x++) {
                    if (screen[y][x] == ' ') {
                        // Map screen to density grid
                        float2 world_pos;
                        world_pos.x = camera_pos.x + (x - WIDTH/2) / zoom;
                        world_pos.y = camera_pos.y - (y - HEIGHT/2) / (zoom * 0.5f);
                        
                        int dx = (int)((world_pos.x + 100) * DENSITY_GRID / 200.0f);
                        int dy = (int)((world_pos.y + 100) * DENSITY_GRID / 200.0f);
                        
                        if (dx >= 0 && dx < DENSITY_GRID && dy >= 0 && dy < DENSITY_GRID) {
                            int density = density_map[dy][dx];
                            if (density > 0) {
                                if (density > 100) screen[y][x] = '#';
                                else if (density > 50) screen[y][x] = '*';
                                else if (density > 20) screen[y][x] = '+';
                                else if (density > 10) screen[y][x] = '-';
                                else if (density > 5) screen[y][x] = ':';
                            }
                        }
                    }
                }
            }
        }
        
        // Display screen
        for (const auto& row : screen) {
            for (char c : row) {
                std::cout << c;
            }
            std::cout << '\n';
        }
        
        // Statistics
        std::cout << "─────────────────────────────────────────────────────────────────────────\n";
        
        // Find and show some key bodies
        std::cout << "Key Bodies: ";
        int shown = 0;
        for (size_t i = 0; i < particles.size() && shown < 5; i++) {
            if (particles[i].type <= 1) {  // Star or planet
                float r = particles[i].pos.length();
                float v = particles[i].vel.length();
                std::cout << "r=" << std::setprecision(1) << r << "AU,v=" << v << " | ";
                shown++;
            }
        }
        std::cout << "\n";
    }
};

// Simple PM solver for testing
class SimplePMSolver {
    static constexpr int GRID_SIZE = 512;
    static constexpr float WORLD_SIZE = 100.0f;
    static constexpr float CELL_SIZE = WORLD_SIZE / GRID_SIZE;
    static constexpr float SOFTENING = 0.01f;
    
    std::vector<std::vector<float>> density;
    std::vector<std::vector<float>> potential;
    std::vector<std::vector<float2>> field;
    
public:
    SimplePMSolver() {
        density.resize(GRID_SIZE, std::vector<float>(GRID_SIZE, 0));
        potential.resize(GRID_SIZE, std::vector<float>(GRID_SIZE, 0));
        field.resize(GRID_SIZE, std::vector<float2>(GRID_SIZE));
    }
    
    void compute_forces(std::vector<Particle>& particles) {
        // For debugging, let's use direct N-body for the main bodies
        // and PM only for asteroids/KBOs
        
        // Clear forces
        for (auto& p : particles) {
            p.force = float2(0, 0);
        }
        
        // Direct N-body for planets and moons (high accuracy needed)
        for (size_t i = 0; i < particles.size(); i++) {
            if (particles[i].type > 2) continue;  // Skip asteroids/KBOs
            
            for (size_t j = 0; j < particles.size(); j++) {
                if (i == j) continue;
                if (particles[j].type > 2 && particles[j].mass < 1e-8) continue;  // Skip tiny asteroids
                
                float2 delta = particles[j].pos - particles[i].pos;
                float dist_sq = delta.x * delta.x + delta.y * delta.y;
                dist_sq = std::max(dist_sq, SOFTENING * SOFTENING);
                
                float dist = std::sqrt(dist_sq);
                float force_mag = SimUnits::G * particles[i].mass * particles[j].mass / dist_sq;
                
                particles[i].force += delta.normalized() * force_mag;
            }
        }
        
        // For asteroids/KBOs, just use simple orbital force toward sun
        // (This is a simplification but keeps them stable)
        for (size_t i = 0; i < particles.size(); i++) {
            if (particles[i].type < 3) continue;  // Only asteroids/KBOs
            
            float2 to_sun = float2(0, 0) - particles[i].pos;
            float dist = to_sun.length();
            if (dist > 0.1f) {
                float force_mag = SimUnits::G * particles[i].mass * 1.0f / (dist * dist);
                particles[i].force = to_sun.normalized() * force_mag;
            }
        }
    }
};

// Initialize a smaller test system first
void build_test_system(std::vector<Particle>& particles) {
    particles.clear();
    
    // Sun
    particles.emplace_back(float2(0, 0), float2(0, 0), 1.0f, 0);
    
    // Add just the main planets
    struct Planet {
        const char* name;
        float dist;
        float mass;
    } planets[] = {
        {"Mercury", 0.387f, 1.66e-7f},
        {"Venus", 0.723f, 2.45e-6f},
        {"Earth", 1.000f, 3.00e-6f},
        {"Mars", 1.524f, 3.23e-7f},
        {"Jupiter", 5.203f, 9.55e-4f},
        {"Saturn", 9.537f, 2.86e-4f},
        {"Uranus", 19.191f, 4.37e-5f},
        {"Neptune", 30.069f, 5.15e-5f}
    };
    
    for (const auto& p : planets) {
        float v = std::sqrt(SimUnits::G * 1.0f / p.dist);
        particles.emplace_back(
            float2(p.dist, 0),
            float2(0, v),
            p.mass,
            1
        );
    }
    
    // Add Earth's moon
    float moon_dist = 0.00257f;
    float earth_pos = 1.0f;
    float earth_vel = std::sqrt(SimUnits::G / 1.0f);
    particles.emplace_back(
        float2(earth_pos + moon_dist, 0),
        float2(0, earth_vel + std::sqrt(SimUnits::G * 3.00e-6f / moon_dist)),
        7.34e-8f,
        2
    );
    
    // Add some asteroids in the belt
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> uniform(0, 1);
    
    for (int i = 0; i < 10000; i++) {
        float r = 2.2f + 1.1f * uniform(rng);
        float theta = uniform(rng) * 2 * M_PI;
        float v = std::sqrt(SimUnits::G / r);
        
        particles.emplace_back(
            float2(r * cos(theta), r * sin(theta)),
            float2(-v * sin(theta), v * cos(theta)),
            1e-12f,
            3
        );
    }
    
    // Add some KBOs
    for (int i = 0; i < 5000; i++) {
        float r = 30.0f + 20.0f * uniform(rng);
        float theta = uniform(rng) * 2 * M_PI;
        float v = std::sqrt(SimUnits::G / r);
        
        particles.emplace_back(
            float2(r * cos(theta), r * sin(theta)),
            float2(-v * sin(theta), v * cos(theta)),
            1e-10f,
            4
        );
    }
    
    std::cout << "Test system built: " << particles.size() << " particles\n";
}

int main() {
    // Use fewer threads for interactive performance
    omp_set_num_threads(4);
    
    std::vector<Particle> particles;
    build_test_system(particles);
    
    SimplePMSolver solver;
    Visualizer viz;
    
    float time = 0;
    float dt = SimUnits::TIME_STEP;
    
    std::cout << "\033[2J\033[H";
    std::cout << "Initializing visualization...\n";
    std::this_thread::sleep_for(std::chrono::seconds(1));
    
    // Main loop
    while (true) {
        auto frame_start = std::chrono::high_resolution_clock::now();
        
        if (!viz.is_paused()) {
            // Velocity Verlet integration
            
            // Update positions
            #pragma omp parallel for
            for (size_t i = 0; i < particles.size(); i++) {
                auto& p = particles[i];
                float2 acc = p.force / p.mass;
                p.pos += p.vel * dt + acc * (0.5f * dt * dt);
            }
            
            // Save old forces
            std::vector<float2> old_forces(particles.size());
            #pragma omp parallel for
            for (size_t i = 0; i < particles.size(); i++) {
                old_forces[i] = particles[i].force;
            }
            
            // Compute new forces
            solver.compute_forces(particles);
            
            // Update velocities
            #pragma omp parallel for
            for (size_t i = 0; i < particles.size(); i++) {
                auto& p = particles[i];
                float2 avg_force = (old_forces[i] + p.force) * 0.5f;
                float2 acc = avg_force / p.mass;
                p.vel += acc * dt;
            }
            
            time += dt;
        }
        
        // Display
        viz.display(particles, time, dt);
        
        // Frame rate limiting
        auto frame_end = std::chrono::high_resolution_clock::now();
        auto frame_time = std::chrono::duration_cast<std::chrono::milliseconds>(frame_end - frame_start);
        if (frame_time.count() < 50) {  // Target ~20 FPS
            std::this_thread::sleep_for(std::chrono::milliseconds(50 - frame_time.count()));
        }
    }
    
    return 0;
}
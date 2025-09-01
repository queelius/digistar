#include "SimpleBackend.h"
#include <iostream>
#include <chrono>
#include <random>
#include <cmath>
#include <iomanip>
#include <thread>
#include <atomic>
#include <termios.h>
#include <unistd.h>
#include <fcntl.h>

// Non-blocking keyboard input
class KeyboardInput {
private:
    termios old_tio, new_tio;
    
public:
    KeyboardInput() {
        tcgetattr(STDIN_FILENO, &old_tio);
        new_tio = old_tio;
        new_tio.c_lflag &= ~(ICANON | ECHO);
        tcsetattr(STDIN_FILENO, TCSANOW, &new_tio);
        fcntl(STDIN_FILENO, F_SETFL, O_NONBLOCK);
    }
    
    ~KeyboardInput() {
        tcsetattr(STDIN_FILENO, TCSANOW, &old_tio);
    }
    
    char getKey() {
        char c = 0;
        read(STDIN_FILENO, &c, 1);
        return c;
    }
};

// Interactive ASCII viewer
class InteractiveViewer {
public:
    float center_x, center_y;
    float zoom;
    int width, height;
    std::vector<std::vector<char>> buffer;
    std::atomic<bool> running{true};
    bool show_labels;
    
    InteractiveViewer(int w = 120, int h = 40, float initial_zoom = 0.1f) 
        : width(w), height(h), zoom(initial_zoom), show_labels(true) {
        buffer.resize(height, std::vector<char>(width, ' '));
    }
    
    void clear() {
        for (auto& row : buffer) {
            std::fill(row.begin(), row.end(), ' ');
        }
    }
    
    void plotPoint(float x, float y, char symbol) {
        float vx = (x - center_x) * zoom + width/2;
        float vy = (y - center_y) * zoom + height/2;
        
        int sx = (int)vx;
        int sy = (int)vy;
        
        if (sx >= 0 && sx < width && sy >= 0 && sy < height) {
            // Priority system for overlapping objects
            if (buffer[sy][sx] == ' ' || symbol == '*' || symbol == 'O') {
                buffer[sy][sx] = symbol;
            }
        }
    }
    
    void plotLabel(float x, float y, const std::string& label) {
        if (!show_labels) return;
        
        float vx = (x - center_x) * zoom + width/2;
        float vy = (y - center_y) * zoom + height/2;
        
        int sx = (int)vx + 1;  // Offset label to the right
        int sy = (int)vy;
        
        if (sy >= 0 && sy < height) {
            for (size_t i = 0; i < label.length() && sx + i < width; i++) {
                if (sx + i >= 0) {
                    buffer[sy][sx + i] = label[i];
                }
            }
        }
    }
    
    void render(float sim_time, float fps, size_t body_count, size_t total_particles) {
        std::cout << "\033[2J\033[H" << std::flush;
        
        // Title
        std::cout << "+" << std::string(width-2, '=') << "+\n";
        std::cout << "| Complete Solar System Simulation";
        std::cout << std::string(width - 35, ' ') << "|\n";
        std::cout << "+" << std::string(width-2, '-') << "+\n";
        
        // Render buffer with colors
        for (const auto& row : buffer) {
            std::cout << "|";
            for (char c : row) {
                switch(c) {
                    case '*': std::cout << "\033[33;1m" << c << "\033[0m"; break;  // Sun - bright yellow
                    case 'O': std::cout << "\033[36;1m" << c << "\033[0m"; break;  // Gas giants - bright cyan
                    case 'o': std::cout << "\033[34m" << c << "\033[0m"; break;    // Rocky planets - blue
                    case 'm': std::cout << "\033[37m" << c << "\033[0m"; break;    // Moons - white
                    case '.': std::cout << "\033[32m" << c << "\033[0m"; break;    // Asteroids - green
                    case ',': std::cout << "\033[35m" << c << "\033[0m"; break;    // Kuiper objects - magenta
                    case ':': std::cout << "\033[33m" << c << "\033[0m"; break;    // Dust - dim yellow
                    default: std::cout << c;
                }
            }
            std::cout << "|\n";
        }
        
        // Status bar
        std::cout << "+" << std::string(width-2, '-') << "+\n";
        std::cout << "| Time: " << std::fixed << std::setprecision(1) << sim_time << " days";
        std::cout << " | FPS: " << std::setprecision(0) << fps;
        std::cout << " | Bodies: " << body_count;
        std::cout << " | Total: " << total_particles/1000 << "K";
        std::cout << " | Zoom: " << std::setprecision(3) << zoom;
        int used = 70;
        std::cout << std::string(width - used, ' ') << "|\n";
        
        // Controls
        std::cout << "| WASD=pan  +/-=zoom  R=reset  L=labels  1-9=goto planet  Q=quit  SPACE=pause";
        std::cout << std::string(width - 79, ' ') << "|\n";
        std::cout << "+" << std::string(width-2, '=') << "+\n";
    }
    
    void handleInput(char key, float box_size) {
        float pan_speed = box_size * 0.05f / zoom;
        float zoom_speed = 1.2f;
        
        switch(key) {
            case 'w': case 'W': center_y -= pan_speed; break;
            case 's': case 'S': center_y += pan_speed; break;
            case 'a': case 'A': center_x -= pan_speed; break;
            case 'd': case 'D': center_x += pan_speed; break;
            case '+': case '=': zoom *= zoom_speed; break;
            case '-': case '_': zoom /= zoom_speed; break;
            case 'l': case 'L': show_labels = !show_labels; break;
            case 'r': case 'R': 
                center_x = box_size/2; 
                center_y = box_size/2; 
                zoom = 0.1f;
                break;
            case 'q': case 'Q':
                running = false;
                break;
        }
        
        zoom = std::max(0.001f, std::min(1000.0f, zoom));
    }
};

// Celestial body info
struct CelestialBody {
    std::string name;
    float orbital_radius;  // AU
    float mass;            // relative to Earth
    float radius;          // relative units
    char symbol;
    int num_moons;
    bool is_planet;
};

// Create complete solar system
std::vector<Particle> createCompleteSolarSystem(float box_size, float scale_au) {
    std::vector<Particle> particles;
    std::random_device rd;
    std::mt19937 gen(rd());
    
    float center = box_size / 2;
    
    // Sun
    Particle sun;
    sun.pos = {center, center};
    sun.vel = {0, 0};
    sun.mass = 333000.0f;  // Sun mass in Earth masses
    sun.radius = 50.0f;
    particles.push_back(sun);
    
    // Planets with real orbital radii (scaled)
    std::vector<CelestialBody> planets = {
        {"Mercury", 0.39f, 0.055f, 2.0f, 'o', 0, true},
        {"Venus", 0.72f, 0.815f, 4.0f, 'o', 0, true},
        {"Earth", 1.00f, 1.000f, 4.0f, 'o', 1, true},    // 1 moon
        {"Mars", 1.52f, 0.107f, 3.0f, 'o', 2, true},     // 2 moons
        {"Jupiter", 5.20f, 317.8f, 20.0f, 'O', 79, true}, // 79 moons!
        {"Saturn", 9.54f, 95.2f, 18.0f, 'O', 82, true},   // 82 moons!
        {"Uranus", 19.19f, 14.5f, 10.0f, 'O', 27, true},  // 27 moons
        {"Neptune", 30.07f, 17.1f, 10.0f, 'O', 14, true}, // 14 moons
    };
    
    // Add planets and their moons
    for (const auto& body : planets) {
        float r = body.orbital_radius * scale_au;
        float angle = (rand() / (float)RAND_MAX) * 2 * M_PI;
        
        // Planet
        Particle planet;
        planet.pos.x = center + r * cos(angle);
        planet.pos.y = center + r * sin(angle);
        
        // Orbital velocity: v = sqrt(GM/r)
        float v_orbit = sqrt(sun.mass / r) * 0.1f;  // Scaled velocity
        planet.vel.x = -v_orbit * sin(angle);
        planet.vel.y = v_orbit * cos(angle);
        planet.mass = body.mass;
        planet.radius = body.radius;
        particles.push_back(planet);
        
        // Add moons
        if (body.num_moons > 0) {
            // For performance, limit moons for gas giants
            int moons_to_create = std::min(body.num_moons, body.is_planet ? 10 : body.num_moons);
            
            for (int m = 0; m < moons_to_create; m++) {
                float moon_dist = body.radius * (2.0f + m * 0.5f);  // Moon distances
                float moon_angle = (rand() / (float)RAND_MAX) * 2 * M_PI;
                
                Particle moon;
                moon.pos.x = planet.pos.x + moon_dist * cos(moon_angle);
                moon.pos.y = planet.pos.y + moon_dist * sin(moon_angle);
                
                // Moon orbital velocity around planet
                float v_moon = sqrt(planet.mass / moon_dist) * 0.5f;
                moon.vel.x = planet.vel.x - v_moon * sin(moon_angle);
                moon.vel.y = planet.vel.y + v_moon * cos(moon_angle);
                moon.mass = 0.01f;  // Small mass
                moon.radius = 1.0f;
                particles.push_back(moon);
            }
        }
    }
    
    // Asteroid Belt (between Mars and Jupiter, 2.2 - 3.2 AU)
    std::uniform_real_distribution<float> asteroid_r(2.2f * scale_au, 3.2f * scale_au);
    std::uniform_real_distribution<float> asteroid_angle(0, 2*M_PI);
    
    for (int i = 0; i < 5000; i++) {  // 5000 asteroids
        float r = asteroid_r(gen);
        float angle = asteroid_angle(gen);
        
        Particle asteroid;
        asteroid.pos.x = center + r * cos(angle);
        asteroid.pos.y = center + r * sin(angle);
        
        float v_orbit = sqrt(sun.mass / r) * 0.1f * (0.9f + 0.2f * (rand()/(float)RAND_MAX));
        asteroid.vel.x = -v_orbit * sin(angle);
        asteroid.vel.y = v_orbit * cos(angle);
        asteroid.mass = 0.0001f;
        asteroid.radius = 0.5f;
        particles.push_back(asteroid);
    }
    
    // Kuiper Belt (30 - 50 AU)
    std::uniform_real_distribution<float> kuiper_r(30.0f * scale_au, 50.0f * scale_au);
    
    for (int i = 0; i < 10000; i++) {  // 10000 Kuiper belt objects
        float r = kuiper_r(gen);
        float angle = asteroid_angle(gen);
        
        Particle kbo;
        kbo.pos.x = center + r * cos(angle);
        kbo.pos.y = center + r * sin(angle);
        
        float v_orbit = sqrt(sun.mass / r) * 0.1f * (0.8f + 0.4f * (rand()/(float)RAND_MAX));
        kbo.vel.x = -v_orbit * sin(angle);
        kbo.vel.y = v_orbit * cos(angle);
        kbo.mass = 0.001f;
        kbo.radius = 0.5f;
        particles.push_back(kbo);
    }
    
    // Add Pluto as a special Kuiper belt object
    float pluto_r = 39.5f * scale_au;
    float pluto_angle = (rand() / (float)RAND_MAX) * 2 * M_PI;
    
    Particle pluto;
    pluto.pos.x = center + pluto_r * cos(pluto_angle);
    pluto.pos.y = center + pluto_r * sin(pluto_angle);
    float v_pluto = sqrt(sun.mass / pluto_r) * 0.1f;
    pluto.vel.x = -v_pluto * sin(pluto_angle);
    pluto.vel.y = v_pluto * cos(pluto_angle);
    pluto.mass = 0.002f;
    pluto.radius = 2.0f;
    particles.push_back(pluto);
    
    return particles;
}

int main() {
    std::cout << "=== Complete Solar System Interactive Simulation ===\n\n";
    std::cout << "Simulating:\n";
    std::cout << "  - The Sun\n";
    std::cout << "  - 8 Planets with accurate orbits\n";
    std::cout << "  - Major moons for each planet\n";
    std::cout << "  - 5,000 asteroids in the main belt\n";
    std::cout << "  - 10,000 Kuiper belt objects\n";
    std::cout << "  - Pluto!\n\n";
    
    // Simulation parameters
    SimulationParams params;
    params.box_size = 20000.0f;  // Large box for solar system
    params.gravity_constant = 10.0f;
    params.softening = 1.0f;
    params.dt = 0.01f;  // Much smaller timestep for stability (0.01 days)
    params.grid_size = 512;
    
    float scale_au = 100.0f;  // 1 AU = 100 simulation units
    
    // Create solar system
    std::cout << "Creating solar system...\n";
    auto particles = createCompleteSolarSystem(params.box_size, scale_au);
    size_t n = particles.size();
    std::cout << "Total bodies: " << n << "\n";
    
    // Initialize backend
    std::cout << "Initializing Particle Mesh backend...\n";
    auto backend = std::make_unique<SimpleBackend>();
    backend->setAlgorithm(ForceAlgorithm::PARTICLE_MESH);
    backend->initialize(n, params);
    backend->setParticles(particles);
    
    // Interactive viewer
    InteractiveViewer viewer(120, 40, 0.05f);  // Start zoomed out to see whole system
    viewer.center_x = params.box_size / 2;
    viewer.center_y = params.box_size / 2;
    
    // Keyboard input
    KeyboardInput keyboard;
    
    // Timing
    auto last_fps_update = std::chrono::high_resolution_clock::now();
    float sim_time = 0;
    int frame_count = 0;
    float current_fps = 0;
    bool paused = false;
    
    // Planet names for display
    std::vector<std::string> planet_names = {
        "Sun", "Mercury", "Venus", "Earth", "Mars", 
        "Jupiter", "Saturn", "Uranus", "Neptune"
    };
    
    std::cout << "\nStarting simulation...\n";
    std::this_thread::sleep_for(std::chrono::seconds(2));
    
    // Main loop
    while (viewer.running) {
        auto frame_start = std::chrono::high_resolution_clock::now();
        
        // Handle input
        char key = keyboard.getKey();
        if (key) {
            if (key == ' ') {
                paused = !paused;
            } else if (key >= '0' && key <= '8') {
                // Jump to planet
                int planet_idx = key - '0';
                if (planet_idx < 9) {
                    viewer.center_x = particles[planet_idx].pos.x;
                    viewer.center_y = particles[planet_idx].pos.y;
                    if (planet_idx == 0) {
                        viewer.zoom = 0.05f;  // Sun view
                    } else if (planet_idx <= 4) {
                        viewer.zoom = 0.5f;   // Inner planets
                    } else {
                        viewer.zoom = 0.2f;   // Outer planets
                    }
                }
            } else {
                viewer.handleInput(key, params.box_size);
            }
        }
        
        // Physics update
        if (!paused) {
            backend->step(params.dt);
            sim_time += params.dt;
            
            // Update particles for display
            if (frame_count % 3 == 0) {
                backend->getParticles(particles);
            }
        }
        
        // Render
        viewer.clear();
        
        // Draw all particles
        for (size_t i = 0; i < particles.size(); i++) {
            const auto& p = particles[i];
            char symbol = '.';  // Default asteroid
            
            if (i == 0) {
                symbol = '*';  // Sun
            } else if (i <= 8) {
                symbol = (i >= 5) ? 'O' : 'o';  // Planets
            } else if (i < 100) {
                symbol = 'm';  // Moons
            } else if (i < 5100) {
                symbol = '.';  // Asteroids
            } else {
                symbol = ',';  // Kuiper objects
            }
            
            viewer.plotPoint(p.pos.x, p.pos.y, symbol);
            
            // Labels for major bodies
            if (i < 9 && viewer.show_labels) {
                viewer.plotLabel(p.pos.x, p.pos.y, planet_names[i]);
            }
        }
        
        viewer.render(sim_time, current_fps, 9, n);
        
        // FPS calculation
        frame_count++;
        auto now = std::chrono::high_resolution_clock::now();
        auto fps_duration = std::chrono::duration<float>(now - last_fps_update).count();
        if (fps_duration > 0.5f) {
            current_fps = frame_count / fps_duration;
            frame_count = 0;
            last_fps_update = now;
        }
        
        // Frame limiting
        auto frame_duration = std::chrono::duration<float>(now - frame_start).count();
        float target_frame_time = 1.0f / 30.0f;
        if (frame_duration < target_frame_time) {
            std::this_thread::sleep_for(
                std::chrono::milliseconds((int)((target_frame_time - frame_duration) * 1000))
            );
        }
    }
    
    std::cout << "\n\nSimulation complete!\n";
    std::cout << "Simulated " << sim_time << " days\n";
    
    return 0;
}
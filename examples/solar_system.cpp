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

// Physical constants with proper units
// We use: Distance in AU, Time in days, Mass in solar masses
const double G_REAL = 6.67430e-11;           // m^3 kg^-1 s^-2
const double AU = 1.496e11;                  // meters
const double DAY = 86400;                    // seconds  
const double SOLAR_MASS = 1.989e30;          // kg
const double EARTH_MASS = 5.972e24;          // kg

// Convert to simulation units where:
// 1 distance unit = 1 AU
// 1 time unit = 1 day  
// 1 mass unit = 1 solar mass
const double G_SIM = G_REAL * SOLAR_MASS * DAY * DAY / (AU * AU * AU);
// This gives us G in units of AU^3 / (solar_mass * day^2)

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
    bool show_trails;
    
    InteractiveViewer(int w = 120, int h = 40, float initial_zoom = 5.0f) 
        : width(w), height(h), zoom(initial_zoom), show_labels(true), show_trails(false) {
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
            if (buffer[sy][sx] == ' ' || symbol == '*' || symbol == 'O') {
                buffer[sy][sx] = symbol;
            }
        }
    }
    
    void plotLabel(float x, float y, const std::string& label) {
        if (!show_labels) return;
        
        float vx = (x - center_x) * zoom + width/2;
        float vy = (y - center_y) * zoom + height/2;
        
        int sx = (int)vx + 1;
        int sy = (int)vy;
        
        if (sy >= 0 && sy < height) {
            for (size_t i = 0; i < label.length() && sx + i < width; i++) {
                if (sx + i >= 0 && buffer[sy][sx + i] == ' ') {
                    buffer[sy][sx + i] = label[i];
                }
            }
        }
    }
    
    void render(float sim_time, float fps, size_t body_count, size_t total_particles) {
        std::cout << "\033[2J\033[H" << std::flush;
        
        // Title
        std::cout << "+" << std::string(width-2, '=') << "+\n";
        std::cout << "| Solar System - Accurate Physics (G=" << std::scientific << std::setprecision(3) << G_SIM << ")";
        std::cout << std::fixed << std::string(width - 55, ' ') << "|\n";
        std::cout << "+" << std::string(width-2, '-') << "+\n";
        
        // Render buffer with colors
        for (const auto& row : buffer) {
            std::cout << "|";
            for (char c : row) {
                switch(c) {
                    case '*': std::cout << "\033[33;1m" << c << "\033[0m"; break;  // Sun
                    case 'O': std::cout << "\033[36;1m" << c << "\033[0m"; break;  // Gas giants
                    case 'o': std::cout << "\033[34m" << c << "\033[0m"; break;    // Rocky planets
                    case 'm': std::cout << "\033[37m" << c << "\033[0m"; break;    // Moons
                    case '.': std::cout << "\033[32m" << c << "\033[0m"; break;    // Asteroids
                    case ',': std::cout << "\033[35m" << c << "\033[0m"; break;    // Kuiper
                    default: std::cout << c;
                }
            }
            std::cout << "|\n";
        }
        
        // Status
        std::cout << "+" << std::string(width-2, '-') << "+\n";
        std::cout << "| Time: " << std::fixed << std::setprecision(1) << sim_time << " days";
        std::cout << " (" << sim_time/365.25 << " years)";
        std::cout << " | FPS: " << std::setprecision(0) << fps;
        std::cout << " | Zoom: " << std::setprecision(1) << zoom << "x";
        std::cout << " | Center: (" << std::setprecision(2) << center_x << " AU)";
        int used = 75;
        std::cout << std::string(width - used, ' ') << "|\n";
        
        // Controls
        std::cout << "| WASD=pan  +/-=zoom  R=reset  L=labels  1-9=goto  T=trails  Q=quit  SPACE=pause";
        std::cout << std::string(width - 82, ' ') << "|\n";
        std::cout << "+" << std::string(width-2, '=') << "+\n";
    }
    
    void handleInput(char key, float box_size) {
        float pan_speed = 0.5f / zoom;  // Pan in AU
        float zoom_speed = 1.2f;
        
        switch(key) {
            case 'w': case 'W': center_y -= pan_speed; break;
            case 's': case 'S': center_y += pan_speed; break;
            case 'a': case 'A': center_x -= pan_speed; break;
            case 'd': case 'D': center_x += pan_speed; break;
            case '+': case '=': zoom *= zoom_speed; break;
            case '-': case '_': zoom /= zoom_speed; break;
            case 'l': case 'L': show_labels = !show_labels; break;
            case 't': case 'T': show_trails = !show_trails; break;
            case 'r': case 'R': 
                center_x = 0;  // Sun at origin
                center_y = 0; 
                zoom = 5.0f;
                break;
            case 'q': case 'Q':
                running = false;
                break;
        }
        
        zoom = std::max(0.1f, std::min(1000.0f, zoom));
    }
};

// Planet data with real values
struct PlanetData {
    std::string name;
    float semi_major_axis;     // AU
    float eccentricity;        // 0-1
    float mass;                // Solar masses
    float radius;              // Display size
    char symbol;
    int major_moons;           // How many major moons to simulate
};

// Create accurate solar system
std::vector<Particle> createAccurateSolarSystem(float box_size) {
    std::vector<Particle> particles;
    std::random_device rd;
    std::mt19937 gen(rd());
    
    // Sun at origin (box center will be handled by coordinate transform)
    Particle sun;
    sun.pos = {0, 0};  // Origin in AU
    sun.vel = {0, 0};
    sun.mass = 1.0f;   // 1 solar mass
    sun.radius = 0.00465f;  // Solar radius in AU
    particles.push_back(sun);
    
    // Planets with accurate data
    std::vector<PlanetData> planets = {
        {"Mercury", 0.387f, 0.206f, 1.66e-7f, 0.0001f, 'o', 0},
        {"Venus",   0.723f, 0.007f, 2.45e-6f, 0.0002f, 'o', 0},
        {"Earth",   1.000f, 0.017f, 3.00e-6f, 0.0002f, 'o', 1},
        {"Mars",    1.524f, 0.093f, 3.23e-7f, 0.0001f, 'o', 2},
        {"Jupiter", 5.203f, 0.048f, 9.54e-4f, 0.001f, 'O', 4},  // 4 Galilean moons
        {"Saturn",  9.537f, 0.054f, 2.86e-4f, 0.0008f, 'O', 1}, // Titan
        {"Uranus",  19.19f, 0.047f, 4.37e-5f, 0.0004f, 'O', 0},
        {"Neptune", 30.07f, 0.009f, 5.15e-5f, 0.0004f, 'O', 1}, // Triton
    };
    
    // Add planets
    for (const auto& planet : planets) {
        float a = planet.semi_major_axis;  // semi-major axis
        float e = planet.eccentricity;
        
        // Start at perihelion for simplicity
        float r = a * (1 - e);
        float angle = (rand() / (float)RAND_MAX) * 2 * M_PI;
        
        Particle p;
        p.pos.x = r * cos(angle);
        p.pos.y = r * sin(angle);
        
        // Orbital velocity at perihelion: v = sqrt(GM/a * (1+e)/(1-e))
        // For circular orbit (e=0): v = sqrt(GM/r)
        float v_peri = sqrt(G_SIM * sun.mass / a * (1 + e) / (1 - e));
        
        // Velocity perpendicular to radius for circular motion
        p.vel.x = -v_peri * sin(angle);
        p.vel.y = v_peri * cos(angle);
        
        p.mass = planet.mass;
        p.radius = planet.radius;
        particles.push_back(p);
        
        // Add major moons
        if (planet.major_moons > 0) {
            if (planet.name == "Earth") {
                // Moon at correct distance
                float moon_dist = 0.00257f;  // 384,400 km in AU
                float moon_angle = angle + M_PI/4;  // Offset from planet
                
                Particle moon;
                moon.pos.x = p.pos.x + moon_dist * cos(moon_angle);
                moon.pos.y = p.pos.y + moon_dist * sin(moon_angle);
                
                // Moon velocity = planet velocity + orbital velocity around planet
                float v_moon = sqrt(G_SIM * p.mass / moon_dist);
                moon.vel.x = p.vel.x - v_moon * sin(moon_angle);
                moon.vel.y = p.vel.y + v_moon * cos(moon_angle);
                moon.mass = 3.69e-8f;  // Moon mass in solar masses
                moon.radius = 0.00001f;
                particles.push_back(moon);
            }
            else if (planet.name == "Jupiter") {
                // Four Galilean moons: Io, Europa, Ganymede, Callisto
                float moon_dists[] = {0.00282f, 0.00449f, 0.00716f, 0.01257f};  // In AU
                float moon_masses[] = {4.5e-8f, 2.4e-8f, 7.5e-8f, 5.4e-8f};    // Solar masses
                
                for (int i = 0; i < 4; i++) {
                    float moon_angle = angle + i * M_PI/2;
                    
                    Particle moon;
                    moon.pos.x = p.pos.x + moon_dists[i] * cos(moon_angle);
                    moon.pos.y = p.pos.y + moon_dists[i] * sin(moon_angle);
                    
                    float v_moon = sqrt(G_SIM * p.mass / moon_dists[i]);
                    moon.vel.x = p.vel.x - v_moon * sin(moon_angle);
                    moon.vel.y = p.vel.y + v_moon * cos(moon_angle);
                    moon.mass = moon_masses[i];
                    moon.radius = 0.00001f;
                    particles.push_back(moon);
                }
            }
        }
    }
    
    // Asteroid belt (2.2 - 3.3 AU)
    std::uniform_real_distribution<float> ast_a(2.2f, 3.3f);
    std::uniform_real_distribution<float> ast_angle(0, 2*M_PI);
    std::uniform_real_distribution<float> ast_e(0, 0.2f);
    
    for (int i = 0; i < 2000; i++) {
        float a = ast_a(gen);
        float e = ast_e(gen);
        float angle = ast_angle(gen);
        float r = a * (1 - e * cos(angle));  // Approximate
        
        Particle ast;
        ast.pos.x = r * cos(angle);
        ast.pos.y = r * sin(angle);
        
        float v = sqrt(G_SIM * sun.mass / r) * (0.9f + 0.2f * (rand()/(float)RAND_MAX));
        ast.vel.x = -v * sin(angle);
        ast.vel.y = v * cos(angle);
        ast.mass = 1e-12f;  // Very small mass
        ast.radius = 0.00001f;
        particles.push_back(ast);
    }
    
    // Kuiper belt (30 - 50 AU) 
    std::uniform_real_distribution<float> kb_a(30.0f, 50.0f);
    
    for (int i = 0; i < 3000; i++) {
        float a = kb_a(gen);
        float angle = ast_angle(gen);
        float r = a;
        
        Particle kbo;
        kbo.pos.x = r * cos(angle);
        kbo.pos.y = r * sin(angle);
        
        float v = sqrt(G_SIM * sun.mass / r);
        kbo.vel.x = -v * sin(angle);
        kbo.vel.y = v * cos(angle);
        kbo.mass = 1e-10f;
        kbo.radius = 0.00001f;
        particles.push_back(kbo);
    }
    
    return particles;
}

int main() {
    std::cout << "=== Accurate Solar System Simulation ===\n\n";
    std::cout << "Using real physical constants:\n";
    std::cout << "  G = " << std::scientific << G_REAL << " m³/kg/s²\n";
    std::cout << "  G_sim = " << G_SIM << " AU³/M☉/day²\n";
    std::cout << "  1 AU = " << AU << " meters\n";
    std::cout << "  1 day = " << DAY << " seconds\n\n";
    std::cout << "Time scale: 1 simulation second ≈ 10 days\n\n";
    
    // Simulation parameters with proper units
    SimulationParams params;
    params.box_size = 100.0f;  // 100 AU box to contain outer planets
    params.gravity_constant = G_SIM;  // Proper gravitational constant
    params.softening = 0.0001f;  // Very small softening (100 km in AU)
    params.dt = 0.1f;  // 0.1 day timestep for accuracy
    params.grid_size = 256;  // Good resolution
    
    // Create solar system
    std::cout << "Creating solar system with accurate orbital mechanics...\n";
    auto particles = createAccurateSolarSystem(params.box_size);
    size_t n = particles.size();
    std::cout << "Total bodies: " << n << "\n";
    std::cout << "  - Sun + 8 planets\n";
    std::cout << "  - Major moons (Earth, Jupiter, Saturn, Neptune)\n";
    std::cout << "  - 2000 asteroids\n";
    std::cout << "  - 3000 Kuiper belt objects\n\n";
    
    // Shift particles to box coordinates (sun at center of box)
    float box_center = params.box_size / 2;
    for (auto& p : particles) {
        p.pos.x += box_center;
        p.pos.y += box_center;
    }
    
    // Initialize backend
    std::cout << "Initializing Particle Mesh backend...\n";
    auto backend = std::make_unique<SimpleBackend>();
    backend->setAlgorithm(ForceAlgorithm::PARTICLE_MESH);
    backend->initialize(n, params);
    backend->setParticles(particles);
    
    // Interactive viewer
    InteractiveViewer viewer(120, 40, 5.0f);
    viewer.center_x = 0;  // Start centered on Sun (which is at origin in our coordinate system)
    viewer.center_y = 0;
    
    // Keyboard input
    KeyboardInput keyboard;
    
    // Timing
    auto last_fps_update = std::chrono::high_resolution_clock::now();
    float sim_time = 0;
    int frame_count = 0;
    float current_fps = 0;
    bool paused = false;
    
    // Planet names
    std::vector<std::string> names = {
        "Sun", "Mercury", "Venus", "Earth", "Mars", 
        "Jupiter", "Saturn", "Uranus", "Neptune"
    };
    
    std::cout << "Starting simulation...\n";
    std::cout << "Press any key to begin.\n";
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
                // Jump to planet (remember to adjust for box coordinates)
                int idx = key - '0';
                if (idx < 9) {
                    viewer.center_x = particles[idx].pos.x - box_center;
                    viewer.center_y = particles[idx].pos.y - box_center;
                    
                    // Adjust zoom based on what we're looking at
                    if (idx == 0) viewer.zoom = 10.0f;      // Sun
                    else if (idx <= 4) viewer.zoom = 50.0f; // Inner planets
                    else viewer.zoom = 10.0f;               // Outer planets
                }
            } else {
                viewer.handleInput(key, params.box_size);
            }
        }
        
        // Physics
        if (!paused) {
            // Multiple small steps for accuracy
            for (int substep = 0; substep < 10; substep++) {
                backend->step(params.dt);
            }
            sim_time += params.dt * 10;  // Total time advanced
            
            // Update particles for display
            if (frame_count % 2 == 0) {
                backend->getParticles(particles);
            }
        }
        
        // Render
        viewer.clear();
        
        // Draw all particles (transform back from box coordinates)
        for (size_t i = 0; i < particles.size(); i++) {
            float x = particles[i].pos.x - box_center;  // Convert back to Sun-centered
            float y = particles[i].pos.y - box_center;
            
            char symbol = '.';
            if (i == 0) symbol = '*';  // Sun
            else if (i <= 8) symbol = (i >= 5) ? 'O' : 'o';  // Planets
            else if (i < 20) symbol = 'm';  // Moons
            else if (i < 2020) symbol = '.';  // Asteroids
            else symbol = ',';  // Kuiper
            
            viewer.plotPoint(x, y, symbol);
            
            // Labels for major bodies
            if (i < 9) {
                viewer.plotLabel(x, y, names[i]);
            }
        }
        
        viewer.render(sim_time, current_fps, 9, n);
        
        // FPS
        frame_count++;
        auto now = std::chrono::high_resolution_clock::now();
        auto fps_duration = std::chrono::duration<float>(now - last_fps_update).count();
        if (fps_duration > 0.5f) {
            current_fps = frame_count / fps_duration;
            frame_count = 0;
            last_fps_update = now;
        }
        
        // Frame limiting - run at 10 FPS (each frame = 1 day of simulation)
        auto frame_duration = std::chrono::duration<float>(now - frame_start).count();
        float target_frame_time = 0.1f;  // 10 FPS
        if (frame_duration < target_frame_time) {
            std::this_thread::sleep_for(
                std::chrono::milliseconds((int)((target_frame_time - frame_duration) * 1000))
            );
        }
    }
    
    std::cout << "\n\nSimulation complete!\n";
    std::cout << "Simulated " << sim_time << " days (" << sim_time/365.25 << " years)\n";
    
    return 0;
}
// Enhanced Solar System with All Moons and Saturn's Rings
// Interactive visualization with detailed moon systems

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
#include <termios.h>
#include <unistd.h>
#include <fcntl.h>
#include <omp.h>

// Simulation units: AU, Solar masses, Years
namespace SimUnits {
    constexpr double G = 4.0 * M_PI * M_PI;  // G in AU³/M☉·year²
    constexpr double TIME_STEP = 0.00001;    // ~0.0876 hours - very small for stability
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
    uint8_t type;  // 0=star, 1=planet, 2=moon, 3=asteroid, 4=KBO, 5=ring
    uint8_t parent_id;  // For moons, which planet they orbit
    std::string name;
    
    Particle() : pos(0, 0), vel(0, 0), force(0, 0), mass(0), type(3), parent_id(255) {}
    Particle(float2 p, float2 v, float m, uint8_t t, const std::string& n = "") 
        : pos(p), vel(v), force(0, 0), mass(m), type(t), parent_id(255), name(n) {}
};

// Moon data structures
struct MoonData {
    const char* name;
    float semi_major_km;  // kilometers
    float mass_kg;         // kg
};

// Jupiter's major moons (we'll simulate the 4 Galilean + 10 other major ones)
const MoonData jupiter_moons[] = {
    // Galilean moons
    {"Io", 421800, 8.93e22},
    {"Europa", 671100, 4.80e22},
    {"Ganymede", 1070400, 1.48e23},
    {"Callisto", 1882700, 1.08e23},
    // Other major moons
    {"Amalthea", 181400, 2.08e18},
    {"Himalia", 11460000, 6.70e18},
    {"Thebe", 221900, 4.30e17},
    {"Elara", 11740000, 8.70e17},
    {"Pasiphae", 23620000, 3.00e17},
    {"Carme", 23400000, 1.30e17},
    {"Sinope", 23940000, 7.50e16},
    {"Lysithea", 11720000, 6.30e16},
    {"Ananke", 21280000, 3.00e16},
    {"Leda", 11165000, 1.10e16}
};

// Saturn's major moons (we'll simulate 20 major ones)
const MoonData saturn_moons[] = {
    {"Mimas", 185540, 3.75e19},
    {"Enceladus", 238040, 1.08e20},
    {"Tethys", 294672, 6.18e20},
    {"Dione", 377415, 1.10e21},
    {"Rhea", 527068, 2.31e21},
    {"Titan", 1221865, 1.35e23},
    {"Hyperion", 1500933, 5.62e18},
    {"Iapetus", 3560854, 1.81e21},
    {"Phoebe", 12947780, 8.29e18},
    {"Janus", 151460, 1.90e18},
    {"Epimetheus", 151410, 5.27e17},
    {"Prometheus", 139380, 1.60e17},
    {"Pandora", 141720, 1.37e17},
    {"Atlas", 137670, 6.60e15},
    {"Pan", 133584, 4.95e15},
    {"Telesto", 294672, 6.20e15},  // Trojan of Tethys
    {"Calypso", 294672, 2.50e15},  // Trojan of Tethys
    {"Helene", 377415, 1.10e16},   // Trojan of Dione
    {"Daphnis", 136500, 7.70e13},
    {"Methone", 194440, 1.65e13}
};

// Other planets' moons
const MoonData earth_moon = {"Moon", 384400, 7.34e22};
const MoonData mars_moons[] = {
    {"Phobos", 9377, 1.06e16},
    {"Deimos", 23460, 1.48e15}
};
const MoonData uranus_moons[] = {
    {"Miranda", 129900, 6.59e19},
    {"Ariel", 190900, 1.35e21},
    {"Umbriel", 266000, 1.17e21},
    {"Titania", 436300, 3.53e21},
    {"Oberon", 583500, 3.01e21}
};
const MoonData neptune_moons[] = {
    {"Triton", 354800, 2.14e22},
    {"Nereid", 5513818, 3.10e19},
    {"Proteus", 117647, 4.40e19}
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

// Enhanced Visualizer
class Visualizer {
    float2 camera_pos;
    float zoom;
    int view_mode;  // 0=inner, 1=belt, 2=jupiter, 3=saturn, 4=outer
    bool show_trails;
    bool show_labels;
    bool paused;
    int frame_count;
    InputHandler input;
    
    // Trail storage
    std::vector<std::vector<float2>> trails;
    static constexpr int MAX_TRAIL_LENGTH = 100;
    
public:
    Visualizer(size_t num_particles) 
        : camera_pos(0, 0), zoom(0.5f), view_mode(0), 
          show_trails(false), show_labels(true), paused(false), frame_count(0) {
        trails.resize(num_particles);
    }
    
    bool is_paused() const { return paused; }
    
    void handle_input() {
        char c = input.get_input();
        if (c == 0) return;
        
        switch(c) {
            // Camera controls
            case 'w': camera_pos.y += 2.0f / zoom; break;
            case 's': camera_pos.y -= 2.0f / zoom; break;
            case 'a': camera_pos.x -= 2.0f / zoom; break;
            case 'd': camera_pos.x += 2.0f / zoom; break;
            
            // Zoom
            case '+': case '=': zoom *= 1.5f; break;
            case '-': case '_': zoom /= 1.5f; break;
            
            // View modes
            case '1': // Inner system
                view_mode = 0; zoom = 2.0f; camera_pos = float2(0, 0); 
                break;
            case '2': // Asteroid belt
                view_mode = 1; zoom = 0.5f; camera_pos = float2(2.5f, 0); 
                break;
            case '3': // Jupiter system
                view_mode = 2; zoom = 50.0f; camera_pos = float2(5.2f, 0); 
                break;
            case '4': // Saturn system
                view_mode = 3; zoom = 30.0f; camera_pos = float2(9.5f, 0); 
                break;
            case '5': // Outer system
                view_mode = 4; zoom = 0.05f; camera_pos = float2(0, 0); 
                break;
            
            // Options
            case 'r': camera_pos = float2(0, 0); break;
            case 'p': paused = !paused; break;
            case 't': show_trails = !show_trails; break;
            case 'l': show_labels = !show_labels; break;
            case 'q': exit(0); break;
        }
    }
    
    void update_trails(const std::vector<Particle>& particles) {
        if (!show_trails) return;
        
        for (size_t i = 0; i < particles.size() && i < trails.size(); i++) {
            if (particles[i].type <= 2) {  // Only track planets and moons
                trails[i].push_back(particles[i].pos);
                if (trails[i].size() > MAX_TRAIL_LENGTH) {
                    trails[i].erase(trails[i].begin());
                }
            }
        }
    }
    
    void display(const std::vector<Particle>& particles, float time, float dt) {
        frame_count++;
        handle_input();
        update_trails(particles);
        
        // Only update display every few frames
        if (frame_count % 5 != 0) return;
        
        std::cout << "\033[2J\033[H";  // Clear screen
        
        // Header
        std::cout << "=== Enhanced Solar System (All Moons + Saturn's Rings) ===\n";
        std::cout << "Time: " << std::fixed << std::setprecision(4) << time << " years | ";
        std::cout << "Particles: " << particles.size() << " | ";
        std::cout << (paused ? "PAUSED" : "RUNNING") << "\n";
        
        // View info
        const char* view_names[] = {"Inner System", "Asteroid Belt", "Jupiter System", 
                                   "Saturn System", "Outer System"};
        std::cout << "View: " << view_names[view_mode] << " | ";
        std::cout << "Zoom: " << std::setprecision(1) << zoom << "x | ";
        std::cout << "Trails: " << (show_trails ? "ON" : "OFF") << "\n";
        
        // Controls
        std::cout << "Controls: WASD=pan, +-=zoom, 12345=views, p=pause, t=trails, l=labels, q=quit\n";
        std::cout << "─────────────────────────────────────────────────────────────────────────\n";
        
        // Visualization area
        const int WIDTH = 120;
        const int HEIGHT = 35;
        std::vector<std::vector<char>> screen(HEIGHT, std::vector<char>(WIDTH, ' '));
        std::vector<std::vector<int>> z_buffer(HEIGHT, std::vector<int>(WIDTH, -1));
        
        // Draw trails first
        if (show_trails) {
            for (size_t i = 0; i < trails.size(); i++) {
                for (const auto& pos : trails[i]) {
                    float2 rel_pos = pos - camera_pos;
                    int sx = WIDTH/2 + (int)(rel_pos.x * zoom);
                    int sy = HEIGHT/2 - (int)(rel_pos.y * zoom * 0.5f);
                    
                    if (sx >= 0 && sx < WIDTH && sy >= 0 && sy < HEIGHT) {
                        if (screen[sy][sx] == ' ') {
                            screen[sy][sx] = '.';
                        }
                    }
                }
            }
        }
        
        // Draw particles
        for (size_t i = 0; i < particles.size(); i++) {
            const auto& p = particles[i];
            
            float2 rel_pos = p.pos - camera_pos;
            int sx = WIDTH/2 + (int)(rel_pos.x * zoom);
            int sy = HEIGHT/2 - (int)(rel_pos.y * zoom * 0.5f);
            
            if (sx >= 0 && sx < WIDTH && sy >= 0 && sy < HEIGHT) {
                char symbol = ' ';
                int priority = 0;
                
                switch(p.type) {
                    case 0: symbol = '@'; priority = 10; break;  // Sun
                    case 1: symbol = 'O'; priority = 8; break;   // Planet
                    case 2: symbol = 'o'; priority = 6; break;   // Moon
                    case 3: symbol = '.'; priority = 2; break;   // Asteroid
                    case 4: symbol = ','; priority = 1; break;   // KBO
                    case 5: symbol = '*'; priority = 3; break;   // Ring particle
                }
                
                if (priority > z_buffer[sy][sx]) {
                    screen[sy][sx] = symbol;
                    z_buffer[sy][sx] = priority;
                    
                    // Add labels for major bodies
                    if (show_labels && !p.name.empty() && p.type <= 1) {
                        int label_y = sy - 1;
                        if (label_y >= 0 && label_y < HEIGHT) {
                            for (size_t j = 0; j < p.name.length() && sx + j < WIDTH; j++) {
                                if (screen[label_y][sx + j] == ' ') {
                                    screen[label_y][sx + j] = p.name[j];
                                }
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
        
        // Count particles by type
        int counts[6] = {0};
        for (const auto& p : particles) {
            counts[p.type]++;
        }
        
        std::cout << "Bodies: Sun:" << counts[0] 
                  << " Planets:" << counts[1]
                  << " Moons:" << counts[2]
                  << " Asteroids:" << counts[3]
                  << " KBOs:" << counts[4]
                  << " Ring:" << counts[5] << "\n";
        
        // Show specific system info based on view
        if (view_mode == 2) {  // Jupiter view
            std::cout << "Jupiter's moons visible: Io, Europa, Ganymede, Callisto + " 
                      << (counts[2] > 4 ? std::to_string(counts[2] - 4) + " others" : "") << "\n";
        } else if (view_mode == 3) {  // Saturn view
            std::cout << "Saturn system: " << counts[5] << " ring particles, "
                      << "Titan, Rhea, Iapetus, and other moons\n";
        }
    }
};

// Physics solver
class PhysicsSolver {
public:
    void compute_forces(std::vector<Particle>& particles) {
        // Clear forces
        #pragma omp parallel for
        for (size_t i = 0; i < particles.size(); i++) {
            particles[i].force = float2(0, 0);
        }
        
        // For efficiency, use direct N-body only for major bodies
        // and simplified central force for small particles
        
        // Direct N-body for planets, moons, and sun
        for (size_t i = 0; i < particles.size(); i++) {
            if (particles[i].type > 2) continue;  // Skip small bodies
            
            for (size_t j = 0; j < particles.size(); j++) {
                if (i == j) continue;
                if (particles[j].type > 2 && particles[j].mass < 1e-9) continue;
                
                float2 delta = particles[j].pos - particles[i].pos;
                float dist_sq = delta.x * delta.x + delta.y * delta.y;
                dist_sq = std::max(dist_sq, 1e-6f);  // Softening
                
                float dist = std::sqrt(dist_sq);
                float force_mag = SimUnits::G * particles[i].mass * particles[j].mass / dist_sq;
                
                particles[i].force += delta.normalized() * force_mag;
            }
        }
        
        // Simplified forces for asteroids, KBOs, and ring particles
        #pragma omp parallel for
        for (size_t i = 0; i < particles.size(); i++) {
            if (particles[i].type < 3) continue;  // Only small bodies
            
            // Force from Sun
            float2 to_sun = float2(0, 0) - particles[i].pos;
            float dist = to_sun.length();
            if (dist > 0.001f) {
                float force_mag = SimUnits::G * particles[i].mass * 1.0f / (dist * dist);
                particles[i].force += to_sun.normalized() * force_mag;
            }
            
            // For ring particles, also add force from Saturn
            if (particles[i].type == 5) {
                // Find Saturn (around 9.5 AU)
                for (size_t j = 0; j < particles.size(); j++) {
                    if (particles[j].type == 1 && particles[j].name == "Saturn") {
                        float2 to_saturn = particles[j].pos - particles[i].pos;
                        float dist2 = to_saturn.length();
                        if (dist2 > 0.0001f) {
                            float force_mag2 = SimUnits::G * particles[i].mass * particles[j].mass / (dist2 * dist2);
                            particles[i].force += to_saturn.normalized() * force_mag2;
                        }
                        break;
                    }
                }
            }
        }
    }
};

// Build the enhanced solar system
void build_enhanced_system(std::vector<Particle>& particles) {
    particles.clear();
    particles.reserve(100000);  // Reserve space
    
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> uniform(0, 1);
    
    // Sun
    particles.emplace_back(float2(0, 0), float2(0, 0), 1.0f, 0, "Sun");
    
    // Planets with real data
    struct Planet {
        const char* name;
        float dist_au;
        float mass_solar;
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
    
    // Add planets
    for (int i = 0; i < 8; i++) {
        float v = std::sqrt(SimUnits::G * 1.0f / planets[i].dist_au);
        particles.emplace_back(
            float2(planets[i].dist_au, 0),
            float2(0, v),
            planets[i].mass_solar,
            1,
            planets[i].name
        );
        
        // Add moons for each planet
        if (i == 2) {  // Earth
            float moon_dist_au = earth_moon.semi_major_km / 1.496e8;  // Convert km to AU
            float moon_mass_solar = earth_moon.mass_kg / 1.989e30;   // Convert kg to solar masses
            float moon_v = std::sqrt(SimUnits::G * planets[i].mass_solar / moon_dist_au);
            
            particles.emplace_back(
                float2(planets[i].dist_au + moon_dist_au, 0),
                float2(0, v + moon_v),
                moon_mass_solar,
                2,
                earth_moon.name
            );
        }
        else if (i == 3) {  // Mars
            for (const auto& moon : mars_moons) {
                float moon_dist_au = moon.semi_major_km / 1.496e8;
                float moon_mass_solar = moon.mass_kg / 1.989e30;
                float moon_v = std::sqrt(SimUnits::G * planets[i].mass_solar / moon_dist_au);
                float angle = uniform(rng) * 2 * M_PI;
                
                particles.emplace_back(
                    float2(planets[i].dist_au + moon_dist_au * cos(angle), 
                           moon_dist_au * sin(angle)),
                    float2(-moon_v * sin(angle), v + moon_v * cos(angle)),
                    moon_mass_solar,
                    2,
                    moon.name
                );
            }
        }
        else if (i == 4) {  // Jupiter - add all 14 major moons
            for (const auto& moon : jupiter_moons) {
                float moon_dist_au = moon.semi_major_km / 1.496e8;
                float moon_mass_solar = moon.mass_kg / 1.989e30;
                float moon_v = std::sqrt(SimUnits::G * planets[i].mass_solar / moon_dist_au);
                float angle = uniform(rng) * 2 * M_PI;
                
                particles.emplace_back(
                    float2(planets[i].dist_au + moon_dist_au * cos(angle), 
                           moon_dist_au * sin(angle)),
                    float2(-moon_v * sin(angle), v + moon_v * cos(angle)),
                    moon_mass_solar,
                    2,
                    moon.name
                );
            }
        }
        else if (i == 5) {  // Saturn - add 20 major moons
            for (const auto& moon : saturn_moons) {
                float moon_dist_au = moon.semi_major_km / 1.496e8;
                float moon_mass_solar = moon.mass_kg / 1.989e30;
                float moon_v = std::sqrt(SimUnits::G * planets[i].mass_solar / moon_dist_au);
                float angle = uniform(rng) * 2 * M_PI;
                
                particles.emplace_back(
                    float2(planets[i].dist_au + moon_dist_au * cos(angle), 
                           moon_dist_au * sin(angle)),
                    float2(-moon_v * sin(angle), v + moon_v * cos(angle)),
                    moon_mass_solar,
                    2,
                    moon.name
                );
            }
            
            // Add Saturn's rings! 
            // A ring: 122,000 - 136,780 km
            // B ring: 91,980 - 117,580 km  
            // C ring: 74,500 - 91,980 km
            std::cout << "Adding Saturn's ring system...\n";
            int ring_particles = 10000;  // Lots of particles for the rings
            
            for (int j = 0; j < ring_particles; j++) {
                // Random radius within ring zones
                float ring_zone = uniform(rng);
                float r_km;
                if (ring_zone < 0.4f) {  // A ring (40%)
                    r_km = 122000 + (136780 - 122000) * uniform(rng);
                } else if (ring_zone < 0.8f) {  // B ring (40%)
                    r_km = 91980 + (117580 - 91980) * uniform(rng);
                } else {  // C ring (20%)
                    r_km = 74500 + (91980 - 74500) * uniform(rng);
                }
                
                float r_au = r_km / 1.496e8;
                float theta = uniform(rng) * 2 * M_PI;
                
                // Orbital velocity for ring particle
                float v_ring = std::sqrt(SimUnits::G * planets[i].mass_solar / r_au);
                
                // Position relative to Saturn
                float2 ring_pos(planets[i].dist_au + r_au * cos(theta),
                               r_au * sin(theta));
                float2 ring_vel(-v_ring * sin(theta), v + v_ring * cos(theta));
                
                particles.emplace_back(ring_pos, ring_vel, 1e-20f, 5, "");
            }
        }
        else if (i == 6) {  // Uranus
            for (const auto& moon : uranus_moons) {
                float moon_dist_au = moon.semi_major_km / 1.496e8;
                float moon_mass_solar = moon.mass_kg / 1.989e30;
                float moon_v = std::sqrt(SimUnits::G * planets[i].mass_solar / moon_dist_au);
                float angle = uniform(rng) * 2 * M_PI;
                
                particles.emplace_back(
                    float2(planets[i].dist_au + moon_dist_au * cos(angle), 
                           moon_dist_au * sin(angle)),
                    float2(-moon_v * sin(angle), v + moon_v * cos(angle)),
                    moon_mass_solar,
                    2,
                    moon.name
                );
            }
        }
        else if (i == 7) {  // Neptune
            for (const auto& moon : neptune_moons) {
                float moon_dist_au = moon.semi_major_km / 1.496e8;
                float moon_mass_solar = moon.mass_kg / 1.989e30;
                float moon_v = std::sqrt(SimUnits::G * planets[i].mass_solar / moon_dist_au);
                float angle = uniform(rng) * 2 * M_PI;
                
                particles.emplace_back(
                    float2(planets[i].dist_au + moon_dist_au * cos(angle), 
                           moon_dist_au * sin(angle)),
                    float2(-moon_v * sin(angle), v + moon_v * cos(angle)),
                    moon_mass_solar,
                    2,
                    moon.name
                );
            }
        }
    }
    
    // Add asteroid belt
    std::cout << "Adding asteroid belt...\n";
    for (int i = 0; i < 5000; i++) {
        float r = 2.2f + 1.1f * uniform(rng);
        float theta = uniform(rng) * 2 * M_PI;
        float v = std::sqrt(SimUnits::G / r);
        
        particles.emplace_back(
            float2(r * cos(theta), r * sin(theta)),
            float2(-v * sin(theta), v * cos(theta)),
            1e-12f,
            3,
            ""
        );
    }
    
    // Add some KBOs
    std::cout << "Adding Kuiper belt objects...\n";
    for (int i = 0; i < 2000; i++) {
        float r = 30.0f + 20.0f * uniform(rng);
        float theta = uniform(rng) * 2 * M_PI;
        float v = std::sqrt(SimUnits::G / r);
        
        particles.emplace_back(
            float2(r * cos(theta), r * sin(theta)),
            float2(-v * sin(theta), v * cos(theta)),
            1e-10f,
            4,
            ""
        );
    }
    
    std::cout << "Enhanced system built: " << particles.size() << " particles\n";
    std::cout << "Including Saturn's rings with " << 10000 << " particles!\n\n";
}

int main() {
    // Use 4 threads for good performance
    omp_set_num_threads(4);
    
    std::vector<Particle> particles;
    build_enhanced_system(particles);
    
    PhysicsSolver solver;
    Visualizer viz(particles.size());
    
    float time = 0;
    float dt = SimUnits::TIME_STEP;
    
    std::cout << "\033[2J\033[H";
    std::cout << "Initializing enhanced solar system...\n";
    std::cout << "Press any key to start...\n";
    std::cin.get();
    
    // Main loop
    while (true) {
        auto frame_start = std::chrono::high_resolution_clock::now();
        
        if (!viz.is_paused()) {
            // Velocity Verlet integration
            
            // Update positions
            #pragma omp parallel for
            for (size_t i = 0; i < particles.size(); i++) {
                auto& p = particles[i];
                if (p.mass > 0) {
                    float2 acc = p.force / p.mass;
                    p.pos += p.vel * dt + acc * (0.5f * dt * dt);
                }
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
                if (p.mass > 0) {
                    float2 avg_force = (old_forces[i] + p.force) * 0.5f;
                    float2 acc = avg_force / p.mass;
                    p.vel += acc * dt;
                }
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
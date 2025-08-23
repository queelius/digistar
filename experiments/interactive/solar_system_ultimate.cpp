// Ultimate Solar System Simulation
// Multiple star systems, named asteroids/KBOs, entity tracking

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
#include <map>

// Simulation units: AU, Solar masses, Years
namespace SimUnits {
    constexpr double G = 4.0 * M_PI * M_PI;  // G in AU³/M☉·year²
    constexpr double TIME_STEP = 0.00001;    // ~0.0876 hours
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
    uint8_t star_system;  // 0=Sol, 1=Alpha Centauri, etc.
    std::string name;
    
    Particle() : pos(0, 0), vel(0, 0), force(0, 0), mass(0), type(3), star_system(0) {}
    Particle(float2 p, float2 v, float m, uint8_t t, const std::string& n = "", uint8_t sys = 0) 
        : pos(p), vel(v), force(0, 0), mass(m), type(t), star_system(sys), name(n) {}
};

// Named asteroids data
struct AsteroidData {
    const char* name;
    float semi_major_au;
    float mass_kg;
} named_asteroids[] = {
    {"Ceres", 2.77f, 9.39e20f},      // Dwarf planet
    {"Vesta", 2.36f, 2.59e20f},
    {"Pallas", 2.77f, 2.04e20f},
    {"Hygiea", 3.14f, 8.67e19f},
    {"Juno", 2.67f, 2.67e19f},
    {"Psyche", 2.92f, 2.27e19f},      // Metal asteroid!
    {"Europa", 3.10f, 1.48e19f},      // Not the moon
    {"Davida", 3.17f, 3.84e19f},
    {"Iris", 2.39f, 1.36e19f},
    {"Eunomia", 2.64f, 3.12e19f},
    {"Eros", 1.46f, 6.69e15f},        // Near-Earth asteroid
    {"Itokawa", 1.32f, 3.51e10f},     // Visited by Hayabusa
    {"Bennu", 1.13f, 7.33e10f},       // Visited by OSIRIS-REx
    {"Ryugu", 1.19f, 4.50e11f}        // Visited by Hayabusa2
};

// Named KBOs/Trans-Neptunian Objects
struct KBOData {
    const char* name;
    float semi_major_au;
    float mass_kg;
} named_kbos[] = {
    {"Pluto", 39.48f, 1.31e22f},      // Dwarf planet
    {"Eris", 67.78f, 1.66e22f},       // Most massive dwarf planet
    {"Makemake", 45.79f, 3.1e21f},
    {"Haumea", 43.34f, 4.01e21f},
    {"Gonggong", 67.38f, 1.75e21f},
    {"Quaoar", 43.69f, 1.4e21f},
    {"Sedna", 525.86f, 1e21f},        // Very distant!
    {"Orcus", 39.42f, 6.32e20f},
    {"Salacia", 42.24f, 4.92e20f},
    {"Varuna", 42.92f, 3.7e20f},
    {"Ixion", 39.70f, 5.8e20f},
    {"Varda", 46.20f, 2.66e20f},
    {"2007_OR10", 67.3f, 1.75e21f}
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

// Entity tracker - allows cycling through named objects
class EntityTracker {
    std::vector<size_t> named_entities;  // Indices of named particles
    int current_tracked = -1;
    
public:
    void build_index(const std::vector<Particle>& particles) {
        named_entities.clear();
        for (size_t i = 0; i < particles.size(); i++) {
            if (!particles[i].name.empty()) {
                named_entities.push_back(i);
            }
        }
        std::cout << "Tracking " << named_entities.size() << " named entities\n";
    }
    
    int get_current() const { return current_tracked; }
    
    size_t get_current_index() const {
        if (current_tracked >= 0 && current_tracked < named_entities.size()) {
            return named_entities[current_tracked];
        }
        return SIZE_MAX;
    }
    
    void next() {
        if (named_entities.empty()) return;
        current_tracked = (current_tracked + 1) % named_entities.size();
    }
    
    void previous() {
        if (named_entities.empty()) return;
        current_tracked--;
        if (current_tracked < 0) current_tracked = named_entities.size() - 1;
    }
    
    void track_by_name(const std::vector<Particle>& particles, const std::string& name) {
        for (size_t i = 0; i < named_entities.size(); i++) {
            if (particles[named_entities[i]].name == name) {
                current_tracked = i;
                return;
            }
        }
    }
    
    std::string get_info(const std::vector<Particle>& particles) const {
        size_t idx = get_current_index();
        if (idx == SIZE_MAX) return "No entity tracked";
        
        const auto& p = particles[idx];
        std::stringstream ss;
        ss << "Tracking: " << p.name << " | ";
        ss << "r=" << std::fixed << std::setprecision(2) << p.pos.length() << " AU | ";
        ss << "v=" << std::setprecision(1) << p.vel.length() << " AU/yr | ";
        
        const char* types[] = {"Star", "Planet", "Moon", "Asteroid", "KBO", "Ring"};
        ss << "Type: " << types[p.type];
        
        return ss.str();
    }
};

// Enhanced Visualizer with tracking
class Visualizer {
    float2 camera_pos;
    float zoom;
    int view_mode;  // 0=free, 1=sol, 2=alpha_cen, 3=tracked
    bool show_trails;
    bool show_labels;
    bool show_grid;
    bool paused;
    int frame_count;
    InputHandler input;
    EntityTracker& tracker;
    
    // Trail storage
    std::vector<std::vector<float2>> trails;
    static constexpr int MAX_TRAIL_LENGTH = 200;
    
public:
    Visualizer(size_t num_particles, EntityTracker& t) 
        : camera_pos(0, 0), zoom(0.5f), view_mode(0), 
          show_trails(false), show_labels(true), show_grid(false),
          paused(false), frame_count(0), tracker(t) {
        trails.resize(num_particles);
    }
    
    bool is_paused() const { return paused; }
    
    void handle_input(const std::vector<Particle>& particles) {
        char c = input.get_input();
        if (c == 0) return;
        
        switch(c) {
            // Camera controls
            case 'w': camera_pos.y += 2.0f / zoom; break;
            case 's': camera_pos.y -= 2.0f / zoom; break;
            case 'a': camera_pos.x -= 2.0f / zoom; break;
            case 'd': camera_pos.x += 2.0f / zoom; break;
            
            // Zoom - works in all modes including tracking
            case '+': case '=': zoom *= 1.5f; break;
            case '-': case '_': zoom /= 1.5f; break;
            
            // View modes
            case '1': // Sol system
                view_mode = 1; zoom = 0.5f; camera_pos = float2(0, 0); 
                break;
            case '2': // Alpha Centauri
                view_mode = 2; zoom = 0.5f; camera_pos = float2(300, 0); 
                break;
            case '3': // Tracked entity
                view_mode = 3;
                if (tracker.get_current_index() < particles.size()) {
                    camera_pos = particles[tracker.get_current_index()].pos;
                    // Don't change zoom - let user control it
                }
                break;
            
            // Entity tracking
            case 'n': case 'N':  // Next entity
                tracker.next();
                if (view_mode == 3 && tracker.get_current_index() < particles.size()) {
                    camera_pos = particles[tracker.get_current_index()].pos;
                }
                break;
            case 'b': case 'B':  // Previous entity
                tracker.previous();
                if (view_mode == 3 && tracker.get_current_index() < particles.size()) {
                    camera_pos = particles[tracker.get_current_index()].pos;
                }
                break;
            
            // Options
            case 'r': camera_pos = float2(0, 0); view_mode = 0; break;
            case 'p': paused = !paused; break;
            case 't': show_trails = !show_trails; break;
            case 'l': show_labels = !show_labels; break;
            case 'g': show_grid = !show_grid; break;
            case 'q': exit(0); break;
        }
    }
    
    void update_trails(const std::vector<Particle>& particles) {
        if (!show_trails) return;
        
        for (size_t i = 0; i < particles.size() && i < trails.size(); i++) {
            if (particles[i].type <= 2 || !particles[i].name.empty()) {  // Named objects
                trails[i].push_back(particles[i].pos);
                if (trails[i].size() > MAX_TRAIL_LENGTH) {
                    trails[i].erase(trails[i].begin());
                }
            }
        }
    }
    
    void display(const std::vector<Particle>& particles, float time, float dt) {
        frame_count++;
        handle_input(particles);
        update_trails(particles);
        
        // Auto-follow tracked entity
        if (view_mode == 3) {
            size_t idx = tracker.get_current_index();
            if (idx < particles.size()) {
                camera_pos = particles[idx].pos;
            }
        }
        
        // Only update display every few frames
        if (frame_count % 5 != 0) return;
        
        std::cout << "\033[2J\033[H";  // Clear screen
        
        // Header
        std::cout << "=== Ultimate Solar System (Multiple Stars, Named Objects) ===\n";
        std::cout << "Time: " << std::fixed << std::setprecision(4) << time << " years | ";
        std::cout << "Particles: " << particles.size() << " | ";
        std::cout << (paused ? "PAUSED" : "RUNNING") << "\n";
        
        // Tracking info
        std::cout << tracker.get_info(particles) << "\n";
        
        // Controls
        std::cout << "Controls: WASD=pan +-=zoom 123=views NB=next/prev p=pause t=trails g=grid q=quit\n";
        std::cout << "─────────────────────────────────────────────────────────────────────────\n";
        
        // Visualization area
        const int WIDTH = 120;
        const int HEIGHT = 35;
        std::vector<std::vector<char>> screen(HEIGHT, std::vector<char>(WIDTH, ' '));
        std::vector<std::vector<int>> z_buffer(HEIGHT, std::vector<int>(WIDTH, -1));
        
        // Draw grid if enabled
        if (show_grid) {
            for (int y = 0; y < HEIGHT; y += 5) {
                for (int x = 0; x < WIDTH; x++) {
                    screen[y][x] = '-';
                }
            }
            for (int x = 0; x < WIDTH; x += 10) {
                for (int y = 0; y < HEIGHT; y++) {
                    if (screen[y][x] == '-') screen[y][x] = '+';
                    else screen[y][x] = '|';
                }
            }
        }
        
        // Draw trails
        if (show_trails) {
            for (size_t i = 0; i < trails.size(); i++) {
                for (size_t j = 0; j < trails[i].size(); j++) {
                    float2 rel_pos = trails[i][j] - camera_pos;
                    int sx = WIDTH/2 + (int)(rel_pos.x * zoom);
                    int sy = HEIGHT/2 - (int)(rel_pos.y * zoom * 0.5f);
                    
                    if (sx >= 0 && sx < WIDTH && sy >= 0 && sy < HEIGHT) {
                        if (screen[sy][sx] == ' ' || screen[sy][sx] == '-' || screen[sy][sx] == '|') {
                            // Fade trail based on age
                            if (j > trails[i].size() - 20) {
                                screen[sy][sx] = ':';
                            } else {
                                screen[sy][sx] = '.';
                            }
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
                
                // Different symbols based on star system
                if (p.star_system == 0) {  // Sol system
                    switch(p.type) {
                        case 0: symbol = '@'; priority = 10; break;  // Sun
                        case 1: symbol = 'O'; priority = 8; break;   // Planet
                        case 2: symbol = 'o'; priority = 6; break;   // Moon
                        case 3: symbol = '.'; priority = 2; break;   // Asteroid
                        case 4: symbol = ','; priority = 1; break;   // KBO
                        case 5: symbol = '*'; priority = 3; break;   // Ring
                    }
                } else {  // Other star systems
                    switch(p.type) {
                        case 0: symbol = '#'; priority = 10; break;  // Other stars
                        case 1: symbol = 'Q'; priority = 8; break;   // Exoplanets
                        case 2: symbol = 'q'; priority = 6; break;   // Exomoons
                        default: symbol = '`'; priority = 1; break;
                    }
                }
                
                // Highlight tracked entity
                if (i == tracker.get_current_index()) {
                    symbol = (symbol == '@' || symbol == '#') ? '*' : 
                            (symbol >= 'A' && symbol <= 'Z') ? symbol : 
                            (symbol >= 'a' && symbol <= 'z') ? symbol - 32 : '>';
                    priority = 15;  // Highest priority
                }
                
                if (priority > z_buffer[sy][sx]) {
                    screen[sy][sx] = symbol;
                    z_buffer[sy][sx] = priority;
                    
                    // Add labels for named bodies
                    if (show_labels && !p.name.empty() && p.type <= 1) {
                        int label_y = sy - 1;
                        if (label_y >= 0 && label_y < HEIGHT) {
                            size_t max_len = std::min(p.name.length(), (size_t)(WIDTH - sx));
                            for (size_t j = 0; j < max_len; j++) {
                                if (screen[label_y][sx + j] == ' ' || 
                                    screen[label_y][sx + j] == '-' ||
                                    screen[label_y][sx + j] == '|') {
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
        
        // Count particles by type and system
        std::map<std::pair<int, int>, int> counts;  // (system, type) -> count
        for (const auto& p : particles) {
            counts[{p.star_system, p.type}]++;
        }
        
        std::cout << "Sol: ";
        for (int t = 0; t < 6; t++) {
            if (counts[{0, t}] > 0) {
                const char* types[] = {"Stars", "Planets", "Moons", "Asteroids", "KBOs", "Ring"};
                std::cout << types[t] << ":" << counts[{0, t}] << " ";
            }
        }
        std::cout << "\n";
        
        if (counts[{1, 0}] > 0) {
            std::cout << "Alpha Centauri: ";
            for (int t = 0; t < 6; t++) {
                if (counts[{1, t}] > 0) {
                    const char* types[] = {"Stars", "Planets", "Moons", "Asteroids", "KBOs", "Ring"};
                    std::cout << types[t] << ":" << counts[{1, t}] << " ";
                }
            }
            std::cout << "\n";
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
        
        // Direct N-body for major bodies
        for (size_t i = 0; i < particles.size(); i++) {
            if (particles[i].type > 2 && particles[i].name.empty()) continue;
            
            for (size_t j = 0; j < particles.size(); j++) {
                if (i == j) continue;
                if (particles[j].type > 2 && particles[j].mass < 1e-9) continue;
                
                // Don't compute forces between different star systems
                // unless they're stars (for binary system dynamics)
                if (particles[i].star_system != particles[j].star_system &&
                    particles[i].type != 0 && particles[j].type != 0) continue;
                
                float2 delta = particles[j].pos - particles[i].pos;
                float dist_sq = delta.x * delta.x + delta.y * delta.y;
                dist_sq = std::max(dist_sq, 1e-6f);
                
                float dist = std::sqrt(dist_sq);
                float force_mag = SimUnits::G * particles[i].mass * particles[j].mass / dist_sq;
                
                particles[i].force += delta.normalized() * force_mag;
            }
        }
        
        // Simplified forces for small unnamed bodies
        #pragma omp parallel for
        for (size_t i = 0; i < particles.size(); i++) {
            if (particles[i].type < 3 || !particles[i].name.empty()) continue;
            
            // Find the star(s) in this system
            for (size_t j = 0; j < particles.size(); j++) {
                if (particles[j].type == 0 && particles[j].star_system == particles[i].star_system) {
                    float2 to_star = particles[j].pos - particles[i].pos;
                    float dist = to_star.length();
                    if (dist > 0.001f) {
                        float force_mag = SimUnits::G * particles[i].mass * particles[j].mass / (dist * dist);
                        particles[i].force += to_star.normalized() * force_mag;
                    }
                }
            }
        }
    }
};

// Build the ultimate system
void build_ultimate_system(std::vector<Particle>& particles) {
    particles.clear();
    particles.reserve(100000);  // Prepare for LOTS of particles
    
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> uniform(0, 1);
    
    // === SOL SYSTEM ===
    particles.emplace_back(float2(0, 0), float2(0, 0), 1.0f, 0, "Sol", 0);
    
    // Main planets
    struct Planet {
        const char* name;
        float dist_au;
        float mass_solar;
        int id;
    } planets[] = {
        {"Mercury", 0.387f, 1.66e-7f, 0},
        {"Venus", 0.723f, 2.45e-6f, 1},
        {"Earth", 1.000f, 3.00e-6f, 2},
        {"Mars", 1.524f, 3.23e-7f, 3},
        {"Jupiter", 5.203f, 9.55e-4f, 4},
        {"Saturn", 9.537f, 2.86e-4f, 5},
        {"Uranus", 19.191f, 4.37e-5f, 6},
        {"Neptune", 30.069f, 5.15e-5f, 7}
    };
    
    for (const auto& p : planets) {
        float v = std::sqrt(SimUnits::G * 1.0f / p.dist_au);
        particles.emplace_back(
            float2(p.dist_au, 0),
            float2(0, v),
            p.mass_solar,
            1,
            p.name,
            0
        );
        
        // Add moons for each planet
        if (p.id == 2) {  // Earth - add Moon
            float moon_dist = 0.00257f;  // AU
            float moon_v = std::sqrt(SimUnits::G * p.mass_solar / moon_dist);
            particles.emplace_back(
                float2(p.dist_au + moon_dist, 0),
                float2(0, v + moon_v),
                7.34e-8f,
                2,
                "Moon",
                0
            );
        }
        else if (p.id == 3) {  // Mars - Phobos and Deimos
            float phobos_dist = 9377e3 / 1.496e11;  // km to AU
            float deimos_dist = 23460e3 / 1.496e11;
            particles.emplace_back(
                float2(p.dist_au + phobos_dist, 0),
                float2(0, v + std::sqrt(SimUnits::G * p.mass_solar / phobos_dist)),
                5.3e-16f,
                2,
                "Phobos",
                0
            );
            particles.emplace_back(
                float2(p.dist_au, deimos_dist),
                float2(v + std::sqrt(SimUnits::G * p.mass_solar / deimos_dist), 0),
                7.4e-16f,
                2,
                "Deimos",
                0
            );
        }
        else if (p.id == 4) {  // Jupiter - Galilean moons + others
            const struct { const char* name; float dist_km; float mass_kg; } jup_moons[] = {
                {"Io", 421800, 8.93e22},
                {"Europa", 671100, 4.80e22},
                {"Ganymede", 1070400, 1.48e23},
                {"Callisto", 1882700, 1.08e23},
                {"Amalthea", 181400, 2.08e18},
                {"Himalia", 11460000, 6.70e18}
            };
            for (const auto& moon : jup_moons) {
                float moon_dist = moon.dist_km / 1.496e11;  // km to AU
                float moon_mass = moon.mass_kg / 1.989e30;  // kg to solar masses
                float moon_v = std::sqrt(SimUnits::G * p.mass_solar / moon_dist);
                float angle = uniform(rng) * 2 * M_PI;
                particles.emplace_back(
                    float2(p.dist_au + moon_dist * cos(angle), moon_dist * sin(angle)),
                    float2(-moon_v * sin(angle), v + moon_v * cos(angle)),
                    moon_mass,
                    2,
                    moon.name,
                    0
                );
            }
        }
        else if (p.id == 5) {  // Saturn - moons AND rings!
            const struct { const char* name; float dist_km; float mass_kg; } sat_moons[] = {
                {"Titan", 1221865, 1.35e23},
                {"Rhea", 527068, 2.31e21},
                {"Iapetus", 3560854, 1.81e21},
                {"Dione", 377415, 1.10e21},
                {"Tethys", 294672, 6.18e20},
                {"Enceladus", 238040, 1.08e20},
                {"Mimas", 185540, 3.75e19}
            };
            for (const auto& moon : sat_moons) {
                float moon_dist = moon.dist_km / 1.496e11;
                float moon_mass = moon.mass_kg / 1.989e30;
                float moon_v = std::sqrt(SimUnits::G * p.mass_solar / moon_dist);
                float angle = uniform(rng) * 2 * M_PI;
                particles.emplace_back(
                    float2(p.dist_au + moon_dist * cos(angle), moon_dist * sin(angle)),
                    float2(-moon_v * sin(angle), v + moon_v * cos(angle)),
                    moon_mass,
                    2,
                    moon.name,
                    0
                );
            }
            
            // SATURN'S RINGS!
            std::cout << "Adding Saturn's magnificent rings...\n";
            for (int j = 0; j < 5000; j++) {  // 5000 ring particles
                float ring_zone = uniform(rng);
                float r_km;
                if (ring_zone < 0.4f) {  // A ring
                    r_km = 122000 + (136780 - 122000) * uniform(rng);
                } else if (ring_zone < 0.8f) {  // B ring
                    r_km = 91980 + (117580 - 91980) * uniform(rng);
                } else {  // C ring
                    r_km = 74500 + (91980 - 74500) * uniform(rng);
                }
                
                float r_au = r_km / 1.496e11;
                float theta = uniform(rng) * 2 * M_PI;
                float v_ring = std::sqrt(SimUnits::G * p.mass_solar / r_au);
                
                particles.emplace_back(
                    float2(p.dist_au + r_au * cos(theta), r_au * sin(theta)),
                    float2(-v_ring * sin(theta), v + v_ring * cos(theta)),
                    1e-20f,
                    5,
                    "",
                    0
                );
            }
        }
        else if (p.id == 6) {  // Uranus - major moons
            const struct { const char* name; float dist_km; float mass_kg; } ur_moons[] = {
                {"Miranda", 129900, 6.59e19},
                {"Ariel", 190900, 1.35e21},
                {"Umbriel", 266000, 1.17e21},
                {"Titania", 436300, 3.53e21},
                {"Oberon", 583500, 3.01e21}
            };
            for (const auto& moon : ur_moons) {
                float moon_dist = moon.dist_km / 1.496e11;
                float moon_mass = moon.mass_kg / 1.989e30;
                float moon_v = std::sqrt(SimUnits::G * p.mass_solar / moon_dist);
                float angle = uniform(rng) * 2 * M_PI;
                particles.emplace_back(
                    float2(p.dist_au + moon_dist * cos(angle), moon_dist * sin(angle)),
                    float2(-moon_v * sin(angle), v + moon_v * cos(angle)),
                    moon_mass,
                    2,
                    moon.name,
                    0
                );
            }
        }
        else if (p.id == 7) {  // Neptune - Triton and others
            const struct { const char* name; float dist_km; float mass_kg; } nep_moons[] = {
                {"Triton", 354800, 2.14e22},
                {"Nereid", 5513818, 3.10e19},
                {"Proteus", 117647, 4.40e19}
            };
            for (const auto& moon : nep_moons) {
                float moon_dist = moon.dist_km / 1.496e11;
                float moon_mass = moon.mass_kg / 1.989e30;
                float moon_v = std::sqrt(SimUnits::G * p.mass_solar / moon_dist);
                float angle = uniform(rng) * 2 * M_PI;
                particles.emplace_back(
                    float2(p.dist_au + moon_dist * cos(angle), moon_dist * sin(angle)),
                    float2(-moon_v * sin(angle), v + moon_v * cos(angle)),
                    moon_mass,
                    2,
                    moon.name,
                    0
                );
            }
        }
    }
    
    // Named asteroids
    std::cout << "Adding named asteroids...\n";
    for (const auto& ast : named_asteroids) {
        float v = std::sqrt(SimUnits::G / ast.semi_major_au);
        float theta = uniform(rng) * 2 * M_PI;
        particles.emplace_back(
            float2(ast.semi_major_au * cos(theta), ast.semi_major_au * sin(theta)),
            float2(-v * sin(theta), v * cos(theta)),
            ast.mass_kg / 1.989e30f,
            3,
            ast.name,
            0
        );
    }
    
    // Named KBOs
    std::cout << "Adding named Kuiper Belt Objects...\n";
    for (const auto& kbo : named_kbos) {
        float v = std::sqrt(SimUnits::G / kbo.semi_major_au);
        float theta = uniform(rng) * 2 * M_PI;
        particles.emplace_back(
            float2(kbo.semi_major_au * cos(theta), kbo.semi_major_au * sin(theta)),
            float2(-v * sin(theta), v * cos(theta)),
            kbo.mass_kg / 1.989e30f,
            4,
            kbo.name,
            0
        );
    }
    
    // Random asteroids - LOTS of them!
    std::cout << "Adding thousands of asteroids...\n";
    for (int i = 0; i < 10000; i++) {
        float r = 2.2f + 1.1f * uniform(rng);
        float theta = uniform(rng) * 2 * M_PI;
        float v = std::sqrt(SimUnits::G / r);
        particles.emplace_back(
            float2(r * cos(theta), r * sin(theta)),
            float2(-v * sin(theta), v * cos(theta)),
            1e-12f,
            3,
            "",
            0
        );
    }
    
    // Random KBOs too!
    std::cout << "Adding thousands of KBOs...\n";
    for (int i = 0; i < 5000; i++) {
        float r = 30.0f + 20.0f * uniform(rng);
        float theta = uniform(rng) * 2 * M_PI;
        float v = std::sqrt(SimUnits::G / r);
        particles.emplace_back(
            float2(r * cos(theta), r * sin(theta)),
            float2(-v * sin(theta), v * cos(theta)),
            1e-14f,
            4,
            "",
            0
        );
    }
    
    // Add famous comets with highly elliptical orbits!
    std::cout << "Adding famous comets...\n";
    struct CometData {
        const char* name;
        float perihelion_au;  // Closest approach to sun
        float aphelion_au;    // Furthest distance
        float mass_kg;
    } comets[] = {
        {"Halley", 0.586f, 35.08f, 2.2e14f},
        {"Hale-Bopp", 0.914f, 370.8f, 1e16f},
        {"Hyakutake", 0.230f, 3410.0f, 1e13f},  // Long period!
        {"ISON", 0.012f, 1000.0f, 1e12f},       // Sungrazer
        {"Encke", 0.33f, 4.1f, 1e13f},          // Short period
        {"Swift-Tuttle", 0.96f, 51.23f, 1e16f}, // Perseid meteor shower
        {"Tempel-Tuttle", 0.98f, 19.69f, 1e15f}, // Leonid meteor shower
        {"NEOWISE", 0.306f, 715.0f, 1e14f}
    };
    
    for (const auto& comet : comets) {
        // Start at perihelion for visibility
        float r = comet.perihelion_au;
        float angle = uniform(rng) * 2 * M_PI;
        
        // Calculate velocity at perihelion using vis-viva equation
        // v² = GM(2/r - 1/a) where a = (perihelion + aphelion)/2
        float semi_major = (comet.perihelion_au + comet.aphelion_au) / 2.0f;
        float v_sq = SimUnits::G * (2.0f/r - 1.0f/semi_major);
        float v = std::sqrt(std::max(0.0f, v_sq));
        
        // Velocity is perpendicular at perihelion
        particles.emplace_back(
            float2(r * cos(angle), r * sin(angle)),
            float2(-v * sin(angle), v * cos(angle)),
            comet.mass_kg / 1.989e30f,
            4,  // Use KBO type for comets
            comet.name,
            0
        );
    }
    
    // === ALPHA CENTAURI SYSTEM (300 AU away - 3x further) ===
    std::cout << "Creating Alpha Centauri binary system...\n";
    
    // Binary star system - two stars orbiting common center of mass
    float2 ac_center(300, 0);  // 300 AU from Sol - much more realistic
    float ac_separation = 23.0f;  // Average separation of Alpha Cen A & B
    
    // Alpha Centauri A (1.1 solar masses)
    float ac_a_mass = 1.1f;
    float ac_b_mass = 0.91f;
    float total_mass = ac_a_mass + ac_b_mass;
    
    // Positions relative to center of mass
    float r_a = ac_separation * ac_b_mass / total_mass;  // A's distance from COM
    float r_b = ac_separation * ac_a_mass / total_mass;  // B's distance from COM
    
    // Orbital velocity for binary orbit
    float v_binary = std::sqrt(SimUnits::G * total_mass / ac_separation) * 0.5f;
    
    particles.emplace_back(
        ac_center + float2(r_a, 0),
        float2(0, v_binary * ac_b_mass / total_mass),
        ac_a_mass,
        0,
        "Alpha Cen A",
        1
    );
    
    particles.emplace_back(
        ac_center + float2(-r_b, 0),
        float2(0, -v_binary * ac_a_mass / total_mass),
        ac_b_mass,
        0,
        "Alpha Cen B",
        1
    );
    
    // Proxima Centauri (small red dwarf, 13000 AU from A&B)
    particles.emplace_back(
        ac_center + float2(0, -30),  // Simplified: 30 AU below in our view
        float2(0.1f, 0),  // Small velocity
        0.12f,
        0,
        "Proxima Cen",
        1
    );
    
    // Proxima b (Earth-like exoplanet!)
    float proxima_b_dist = 0.05f;  // Very close to red dwarf
    float proxima_b_v = std::sqrt(SimUnits::G * 0.12f / proxima_b_dist);
    particles.emplace_back(
        ac_center + float2(proxima_b_dist, -30),
        float2(0, proxima_b_v),
        3.0e-6f,  // Earth mass
        1,
        "Proxima b",
        1
    );
    
    // Add some planets around Alpha Cen A
    std::vector<std::pair<std::string, float>> ac_planets = {
        {"Aurora", 0.5f},      // Hot planet
        {"Pandora", 1.2f},     // Avatar reference!
        {"Polyphemus", 2.0f},  // Gas giant (Pandora's parent in Avatar)
        {"Minerva", 3.5f},
        {"Chiron", 5.0f}
    };
    
    for (const auto& [name, dist] : ac_planets) {
        float v = std::sqrt(SimUnits::G * ac_a_mass / dist);
        float angle = uniform(rng) * 2 * M_PI;
        
        float mass = (dist > 2.0f) ? 1e-3f : 5e-6f;  // Gas giants vs terrestrial
        
        particles.emplace_back(
            ac_center + float2(r_a + dist * cos(angle), dist * sin(angle)),
            float2(-v * sin(angle), v * cos(angle) + v_binary * ac_b_mass / total_mass),
            mass,
            1,
            name,
            1
        );
        
        // Add a moon to Polyphemus (Pandora!)
        if (name == "Polyphemus") {
            float moon_dist = 0.01f;
            float moon_v = std::sqrt(SimUnits::G * mass / moon_dist);
            particles.emplace_back(
                ac_center + float2(r_a + dist * cos(angle) + moon_dist, dist * sin(angle)),
                float2(-v * sin(angle), v * cos(angle) + moon_v + v_binary * ac_b_mass / total_mass),
                1e-6f,
                2,
                "Pandora(moon)",
                1
            );
        }
    }
    
    // Add asteroid belt around Alpha Centauri
    for (int i = 0; i < 3000; i++) {  // More asteroids!
        float r = 7.0f + 3.0f * uniform(rng);  // Belt from 7-10 AU
        float theta = uniform(rng) * 2 * M_PI;
        float v = std::sqrt(SimUnits::G * total_mass / r);
        
        particles.emplace_back(
            ac_center + float2(r * cos(theta), r * sin(theta)),
            float2(-v * sin(theta), v * cos(theta)),
            1e-15f,
            3,
            "",
            1
        );
    }
    
    std::cout << "Ultimate system built: " << particles.size() << " particles\n";
    std::cout << "Sol system + Alpha Centauri triple star system\n";
    std::cout << "Named objects include asteroids, KBOs, and exoplanets\n\n";
}

int main() {
    omp_set_num_threads(4);
    
    std::vector<Particle> particles;
    build_ultimate_system(particles);
    
    EntityTracker tracker;
    tracker.build_index(particles);
    
    PhysicsSolver solver;
    Visualizer viz(particles.size(), tracker);
    
    float time = 0;
    float dt = SimUnits::TIME_STEP;
    
    std::cout << "\033[2J\033[H";
    std::cout << "=== Ultimate Solar System Simulation ===\n";
    std::cout << "Features:\n";
    std::cout << "  - Sol system with named asteroids and KBOs\n";
    std::cout << "  - Alpha Centauri triple star system 100 AU away\n";
    std::cout << "  - Exoplanets including Pandora from Avatar!\n";
    std::cout << "  - Entity tracking: press N/B to cycle through named objects\n";
    std::cout << "\nPress any key to start...\n";
    std::cin.get();
    
    // Main loop
    while (true) {
        auto frame_start = std::chrono::high_resolution_clock::now();
        
        if (!viz.is_paused()) {
            // Velocity Verlet integration
            #pragma omp parallel for
            for (size_t i = 0; i < particles.size(); i++) {
                auto& p = particles[i];
                if (p.mass > 0) {
                    float2 acc = p.force / p.mass;
                    p.pos += p.vel * dt + acc * (0.5f * dt * dt);
                }
            }
            
            std::vector<float2> old_forces(particles.size());
            #pragma omp parallel for
            for (size_t i = 0; i < particles.size(); i++) {
                old_forces[i] = particles[i].force;
            }
            
            solver.compute_forces(particles);
            
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
        
        viz.display(particles, time, dt);
        
        auto frame_end = std::chrono::high_resolution_clock::now();
        auto frame_time = std::chrono::duration_cast<std::chrono::milliseconds>(frame_end - frame_start);
        if (frame_time.count() < 50) {
            std::this_thread::sleep_for(std::chrono::milliseconds(50 - frame_time.count()));
        }
    }
    
    return 0;
}
// Interactive Orbital & Collision Simulation
// Combines PM gravity with soft contact forces
// ASCII visualization with pan/zoom controls

#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>
#include <chrono>
#include <thread>
#include <termios.h>
#include <unistd.h>
#include <fcntl.h>
#include <cstring>
#include <algorithm>
#include <complex>

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
    float radius;
    char symbol;  // ASCII representation
    
    Particle(float2 p, float2 v, float m, float r, char s) 
        : pos(p), vel(v), mass(m), radius(r), symbol(s) {}
};

// Simple PM Gravity (simplified for 2 particles)
class PMGravity {
    static constexpr float G = 100.0f;  // Gravitational constant (scaled for visibility)
    
public:
    void compute_forces(std::vector<Particle>& particles) {
        // For two particles, just direct calculation
        // Full PM would use FFT for many particles
        
        for (auto& p : particles) {
            p.force = float2(0, 0);
        }
        
        if (particles.size() != 2) return;
        
        float2 diff = particles[1].pos - particles[0].pos;
        float dist_sq = diff.x * diff.x + diff.y * diff.y;
        
        // Softening to prevent singularity
        float softening = 1.0f;
        dist_sq = std::max(dist_sq, softening * softening);
        
        float dist = std::sqrt(dist_sq);
        float force_mag = G * particles[0].mass * particles[1].mass / dist_sq;
        float2 force_dir = diff / dist;
        
        particles[0].force += force_dir * force_mag;
        particles[1].force -= force_dir * force_mag;
    }
};

// Soft Contact Forces
class SoftContact {
    static constexpr float STIFFNESS = 5000.0f;
    static constexpr float DAMPING = 50.0f;
    
public:
    void compute_forces(std::vector<Particle>& particles) {
        if (particles.size() != 2) return;
        
        float2 diff = particles[1].pos - particles[0].pos;
        float dist = diff.length();
        float overlap = (particles[0].radius + particles[1].radius) - dist;
        
        if (overlap > 0 && dist > 0) {
            // Soft repulsion force
            float2 normal = diff / dist;
            
            // Spring-like force
            float spring_force = STIFFNESS * overlap;
            
            // Damping based on relative velocity
            float2 rel_vel = particles[1].vel - particles[0].vel;
            float vel_along_normal = rel_vel.x * normal.x + rel_vel.y * normal.y;
            float damping_force = DAMPING * vel_along_normal;
            
            float total_force = spring_force + damping_force;
            
            particles[0].force -= normal * total_force;
            particles[1].force += normal * total_force;
        }
    }
};

// ASCII Visualization
class ASCIIRenderer {
    int width = 80;
    int height = 30;
    float2 camera_pos;
    float zoom = 1.0f;
    std::vector<std::vector<char>> buffer;
    
public:
    ASCIIRenderer() : camera_pos(0, 0) {
        buffer.resize(height, std::vector<char>(width, ' '));
    }
    
    void pan(float dx, float dy) {
        camera_pos.x += dx / zoom;
        camera_pos.y += dy / zoom;
    }
    
    void zoom_in() { zoom *= 1.2f; }
    void zoom_out() { zoom /= 1.2f; }
    void reset_view() { camera_pos = float2(0, 0); zoom = 1.0f; }
    
    void render(const std::vector<Particle>& particles, float sim_time) {
        // Clear buffer
        for (auto& row : buffer) {
            std::fill(row.begin(), row.end(), ' ');
        }
        
        // Draw grid
        draw_grid();
        
        // Draw particles
        for (const auto& p : particles) {
            draw_particle(p);
        }
        
        // Draw trails (optional)
        draw_orbit_trail(particles);
        
        // Draw UI
        draw_ui(particles, sim_time);
        
        // Output to terminal
        display();
    }
    
private:
    void draw_grid() {
        // Draw axis lines
        int cx = world_to_screen_x(0);
        int cy = world_to_screen_y(0);
        
        // Vertical axis
        if (cx >= 0 && cx < width) {
            for (int y = 0; y < height; y++) {
                buffer[y][cx] = '|';
            }
        }
        
        // Horizontal axis
        if (cy >= 0 && cy < height) {
            for (int x = 0; x < width; x++) {
                buffer[cy][x] = '-';
            }
        }
        
        // Origin
        if (cx >= 0 && cx < width && cy >= 0 && cy < height) {
            buffer[cy][cx] = '+';
        }
    }
    
    void draw_particle(const Particle& p) {
        int sx = world_to_screen_x(p.pos.x);
        int sy = world_to_screen_y(p.pos.y);
        
        // Draw radius (approximate circle)
        int screen_radius = std::max(1, (int)(p.radius * zoom));
        
        for (int dy = -screen_radius; dy <= screen_radius; dy++) {
            for (int dx = -screen_radius; dx <= screen_radius; dx++) {
                if (dx*dx + dy*dy <= screen_radius*screen_radius) {
                    int px = sx + dx;
                    int py = sy + dy;
                    
                    if (px >= 0 && px < width && py >= 0 && py < height) {
                        // Center gets special symbol, edges get dots
                        if (dx == 0 && dy == 0) {
                            buffer[py][px] = p.symbol;
                        } else if (dx*dx + dy*dy == screen_radius*screen_radius) {
                            if (buffer[py][px] == ' ') buffer[py][px] = '.';
                        } else {
                            if (buffer[py][px] == ' ') buffer[py][px] = ':';
                        }
                    }
                }
            }
        }
    }
    
    void draw_orbit_trail(const std::vector<Particle>& particles) {
        // Simple orbit prediction for circular motion
        if (particles.size() != 2) return;
        
        // Calculate orbital parameters
        float2 com = (particles[0].pos * particles[0].mass + 
                     particles[1].pos * particles[1].mass) / 
                    (particles[0].mass + particles[1].mass);
        
        // Draw predicted orbit for smaller particle
        int smaller = (particles[0].mass < particles[1].mass) ? 0 : 1;
        float2 r = particles[smaller].pos - com;
        float orbit_radius = r.length();
        
        // Draw orbit circle
        for (int i = 0; i < 360; i += 10) {
            float angle = i * M_PI / 180.0f;
            float2 orbit_pos = com + float2(cos(angle), sin(angle)) * orbit_radius;
            
            int sx = world_to_screen_x(orbit_pos.x);
            int sy = world_to_screen_y(orbit_pos.y);
            
            if (sx >= 0 && sx < width && sy >= 0 && sy < height) {
                if (buffer[sy][sx] == ' ') buffer[sy][sx] = '\'';
            }
        }
    }
    
    void draw_ui(const std::vector<Particle>& particles, float sim_time) {
        // Top info bar
        std::string info = "Time: " + std::to_string((int)sim_time) + 
                          "s | Zoom: " + std::to_string((int)(zoom * 100)) + "% | " +
                          "Cam: (" + std::to_string((int)camera_pos.x) + "," + 
                          std::to_string((int)camera_pos.y) + ")";
        
        for (size_t i = 0; i < info.length() && i < width; i++) {
            buffer[0][i] = info[i];
        }
        
        // Energy info
        if (particles.size() == 2) {
            float ke = 0.5f * particles[0].mass * particles[0].vel.length() * particles[0].vel.length() +
                      0.5f * particles[1].mass * particles[1].vel.length() * particles[1].vel.length();
            
            float2 diff = particles[1].pos - particles[0].pos;
            float dist = diff.length();
            float pe = -100.0f * particles[0].mass * particles[1].mass / dist;  // G=100
            
            std::string energy = "KE: " + std::to_string((int)ke) + 
                               " PE: " + std::to_string((int)pe) +
                               " Total: " + std::to_string((int)(ke + pe));
            
            for (size_t i = 0; i < energy.length() && i < width; i++) {
                buffer[1][i] = energy[i];
            }
        }
        
        // Controls help (bottom)
        std::string controls = "WASD:pan +-:zoom R:reset Q:quit SPACE:pause";
        for (size_t i = 0; i < controls.length() && i < width; i++) {
            buffer[height-1][i] = controls[i];
        }
    }
    
    int world_to_screen_x(float wx) {
        return width/2 + (int)((wx - camera_pos.x) * zoom);
    }
    
    int world_to_screen_y(float wy) {
        return height/2 - (int)((wy - camera_pos.y) * zoom);  // Flip Y
    }
    
    void display() {
        // Clear screen
        std::cout << "\033[2J\033[H";
        
        // Draw buffer
        for (const auto& row : buffer) {
            for (char c : row) {
                // Color coding
                switch(c) {
                    case '@': std::cout << "\033[1;33m" << c << "\033[0m"; break;  // Yellow for star
                    case 'o': std::cout << "\033[1;36m" << c << "\033[0m"; break;  // Cyan for planet
                    case ':': std::cout << "\033[0;33m" << c << "\033[0m"; break;  // Dim yellow
                    case '.': std::cout << "\033[0;36m" << c << "\033[0m"; break;  // Dim cyan
                    case '\'': std::cout << "\033[0;32m" << c << "\033[0m"; break; // Green for orbit
                    case '|':
                    case '-':
                    case '+': std::cout << "\033[0;90m" << c << "\033[0m"; break;  // Gray for grid
                    default: std::cout << c;
                }
            }
            std::cout << '\n';
        }
        std::cout << std::flush;
    }
};

// Terminal input handling
class InputHandler {
    struct termios old_tio, new_tio;
    
public:
    InputHandler() {
        // Setup non-blocking input
        tcgetattr(STDIN_FILENO, &old_tio);
        new_tio = old_tio;
        new_tio.c_lflag &= (~ICANON & ~ECHO);
        tcsetattr(STDIN_FILENO, TCSANOW, &new_tio);
        
        // Make stdin non-blocking
        fcntl(STDIN_FILENO, F_SETFL, O_NONBLOCK);
    }
    
    ~InputHandler() {
        // Restore terminal settings
        tcsetattr(STDIN_FILENO, TCSANOW, &old_tio);
    }
    
    char get_input() {
        char c = 0;
        if (read(STDIN_FILENO, &c, 1) < 0) {
            return 0;  // No input
        }
        return c;
    }
};

// Main simulation
class OrbitalSimulation {
    std::vector<Particle> particles;
    PMGravity gravity;
    SoftContact contact;
    ASCIIRenderer renderer;
    InputHandler input;
    
    float sim_time = 0;
    float dt = 0.001f;  // Small timestep for stability
    bool paused = false;
    bool running = true;
    
public:
    void initialize() {
        // Create two particles: a massive "star" and smaller "planet"
        
        // Star (massive, stationary initially)
        particles.emplace_back(
            float2(0, 0),      // position
            float2(0, 0),      // velocity  
            100.0f,            // mass
            5.0f,              // radius
            '@'                // symbol
        );
        
        // Planet (smaller, orbiting)
        float orbital_radius = 30.0f;
        float orbital_speed = std::sqrt(100.0f * 100.0f / orbital_radius);  // v = sqrt(GM/r)
        
        particles.emplace_back(
            float2(orbital_radius, 0),     // position
            float2(0, orbital_speed),      // velocity for circular orbit
            10.0f,                          // mass
            2.0f,                           // radius
            'o'                             // symbol
        );
        
        // Adjust for center of mass motion
        float2 com_vel = (particles[0].vel * particles[0].mass + 
                         particles[1].vel * particles[1].mass) / 
                        (particles[0].mass + particles[1].mass);
        
        particles[0].vel -= com_vel;
        particles[1].vel -= com_vel;
    }
    
    void update() {
        if (paused) return;
        
        // Use Velocity Verlet for better energy conservation
        
        // Update positions
        for (auto& p : particles) {
            p.pos += p.vel * dt + p.force * (0.5f * dt * dt / p.mass);
        }
        
        // Save old forces
        std::vector<float2> old_forces;
        for (const auto& p : particles) {
            old_forces.push_back(p.force);
        }
        
        // Compute new forces
        gravity.compute_forces(particles);
        contact.compute_forces(particles);
        
        // Update velocities with average of old and new forces
        for (size_t i = 0; i < particles.size(); i++) {
            float2 avg_force = (old_forces[i] + particles[i].force) * 0.5f;
            particles[i].vel += avg_force * (dt / particles[i].mass);
        }
        
        sim_time += dt;
    }
    
    void handle_input() {
        char c = input.get_input();
        
        switch(c) {
            case 'q': case 'Q': running = false; break;
            case ' ': paused = !paused; break;
            case 'r': case 'R': renderer.reset_view(); break;
            
            // Camera controls
            case 'w': case 'W': renderer.pan(0, 5); break;
            case 's': case 'S': renderer.pan(0, -5); break;
            case 'a': case 'A': renderer.pan(-5, 0); break;
            case 'd': case 'D': renderer.pan(5, 0); break;
            
            // Zoom
            case '+': case '=': renderer.zoom_in(); break;
            case '-': case '_': renderer.zoom_out(); break;
            
            // Speed control
            case ',': dt = std::max(0.0001f, dt * 0.5f); break;
            case '.': dt = std::min(0.01f, dt * 2.0f); break;
        }
    }
    
    void run() {
        initialize();
        
        auto last_time = std::chrono::steady_clock::now();
        const auto frame_duration = std::chrono::milliseconds(50);  // 20 FPS
        
        while (running) {
            auto current_time = std::chrono::steady_clock::now();
            
            // Fixed timestep physics (subcycle for stability)
            int substeps = 10;
            for (int i = 0; i < substeps; i++) {
                update();
            }
            
            // Handle input
            handle_input();
            
            // Render at fixed framerate
            if (current_time - last_time >= frame_duration) {
                renderer.render(particles, sim_time);
                last_time = current_time;
            }
            
            // Small sleep to prevent CPU spinning
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    }
};

int main() {
    std::cout << "\033[2J\033[H";  // Clear screen
    std::cout << "=== Orbital & Collision Simulation ===\n";
    std::cout << "Combining PM Gravity with Soft Contact Forces\n\n";
    std::cout << "Controls:\n";
    std::cout << "  WASD - Pan camera\n";
    std::cout << "  +/-  - Zoom in/out\n";
    std::cout << "  R    - Reset view\n";
    std::cout << "  ,/.  - Slow down/speed up time\n";
    std::cout << "  Space - Pause/resume\n";
    std::cout << "  Q    - Quit\n\n";
    std::cout << "Press any key to start...";
    std::cin.get();
    
    try {
        OrbitalSimulation sim;
        sim.run();
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    std::cout << "\033[2J\033[H";  // Clear screen
    std::cout << "Simulation ended.\n";
    
    return 0;
}
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
        // Save current terminal settings
        tcgetattr(STDIN_FILENO, &old_tio);
        new_tio = old_tio;
        
        // Disable canonical mode and echo
        new_tio.c_lflag &= ~(ICANON | ECHO);
        tcsetattr(STDIN_FILENO, TCSANOW, &new_tio);
        
        // Make stdin non-blocking
        fcntl(STDIN_FILENO, F_SETFL, O_NONBLOCK);
    }
    
    ~KeyboardInput() {
        // Restore terminal settings
        tcsetattr(STDIN_FILENO, TCSANOW, &old_tio);
    }
    
    char getKey() {
        char c = 0;
        read(STDIN_FILENO, &c, 1);
        return c;
    }
};

// Interactive ASCII viewer with controls
class InteractiveViewer {
public:
    float center_x, center_y;
    float zoom;
    int width, height;
    std::vector<std::vector<int>> density;
    std::atomic<bool> running{true};
    
    InteractiveViewer(int w = 100, int h = 40, float initial_zoom = 1.0f) 
        : width(w), height(h), zoom(initial_zoom) {
        density.resize(height, std::vector<int>(width, 0));
    }
    
    void clear() {
        for (auto& row : density) {
            std::fill(row.begin(), row.end(), 0);
        }
    }
    
    void plotParticles(const std::vector<Particle>& particles, float box_size) {
        clear();
        
        // Count particles in each screen cell
        for (const auto& p : particles) {
            // Transform to view coordinates
            float vx = (p.pos.x - center_x) * zoom + width/2;
            float vy = (p.pos.y - center_y) * zoom + height/2;
            
            int sx = (int)vx;
            int sy = (int)vy;
            
            if (sx >= 0 && sx < width && sy >= 0 && sy < height) {
                density[sy][sx]++;
            }
        }
    }
    
    void render(float sim_time, float fps, size_t particle_count) {
        // Clear screen
        std::cout << "\033[2J\033[H" << std::flush;
        
        // Title bar
        std::cout << "╔" << std::string(width-2, '═') << "╗\n";
        std::cout << "║ 2M Particle Interactive Simulation";
        std::cout << std::string(width - 37, ' ') << "║\n";
        std::cout << "╠" << std::string(width-2, '═') << "╣\n";
        
        // Find max density for coloring
        int max_density = 0;
        for (const auto& row : density) {
            for (int d : row) {
                max_density = std::max(max_density, d);
            }
        }
        
        // Render density field
        for (const auto& row : density) {
            std::cout << "║";
            for (int d : row) {
                if (d == 0) {
                    std::cout << ' ';
                } else {
                    // Color based on density
                    float ratio = (float)d / (max_density + 1);
                    if (ratio < 0.1) {
                        std::cout << "\033[32m·\033[0m";  // Green dot
                    } else if (ratio < 0.3) {
                        std::cout << "\033[33m:\033[0m";  // Yellow colon
                    } else if (ratio < 0.5) {
                        std::cout << "\033[33mo\033[0m";  // Yellow o
                    } else if (ratio < 0.7) {
                        std::cout << "\033[31mO\033[0m";  // Red O
                    } else if (ratio < 0.9) {
                        std::cout << "\033[35m#\033[0m";  // Magenta #
                    } else {
                        std::cout << "\033[37;1m@\033[0m";  // Bright white @
                    }
                }
            }
            std::cout << "║\n";
        }
        
        // Status bar
        std::cout << "╠" << std::string(width-2, '═') << "╣\n";
        std::cout << "║ Time: " << std::fixed << std::setprecision(1) << sim_time;
        std::cout << " | FPS: " << std::setprecision(0) << fps;
        std::cout << " | Particles: " << particle_count/1000 << "K";
        std::cout << " | Zoom: " << std::setprecision(2) << zoom;
        std::cout << " | Center: (" << (int)center_x << "," << (int)center_y << ")";
        int used = 60;
        std::cout << std::string(width - used, ' ') << "║\n";
        
        // Controls
        std::cout << "║ Controls: WASD=pan  +/-=zoom  R=reset  Q=quit  SPACE=pause";
        std::cout << std::string(width - 61, ' ') << "║\n";
        std::cout << "╚" << std::string(width-2, '═') << "╝\n";
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
            case 'r': case 'R': 
                center_x = box_size/2; 
                center_y = box_size/2; 
                zoom = 1.0f;
                break;
            case 'q': case 'Q':
                running = false;
                break;
        }
        
        // Clamp zoom
        zoom = std::max(0.01f, std::min(100.0f, zoom));
    }
};

// Create initial particle distribution
std::vector<Particle> createGalaxy(size_t n, float box_size) {
    std::vector<Particle> particles(n);
    std::random_device rd;
    std::mt19937 gen(rd());
    
    // Create multiple galactic clusters
    std::normal_distribution<float> cluster_dist(0, box_size/20);
    std::uniform_real_distribution<float> angle_dist(0, 2*M_PI);
    std::uniform_real_distribution<float> radius_dist(box_size*0.1, box_size*0.4);
    std::uniform_real_distribution<float> mass_dist(0.5f, 2.0f);
    
    // Create 3-5 galaxy centers
    int num_centers = 3 + rand() % 3;
    std::vector<float2> centers;
    for (int i = 0; i < num_centers; i++) {
        float angle = angle_dist(gen);
        float r = radius_dist(gen);
        centers.push_back({
            box_size/2 + r * cos(angle),
            box_size/2 + r * sin(angle)
        });
    }
    
    // Distribute particles among galaxies
    for (size_t i = 0; i < n; i++) {
        int galaxy = i % num_centers;
        float2 center = centers[galaxy];
        
        // Spiral galaxy structure
        float angle = angle_dist(gen);
        float r = std::abs(cluster_dist(gen)) * (1 + 0.1f * angle);  // Spiral
        
        particles[i].pos.x = center.x + r * cos(angle);
        particles[i].pos.y = center.y + r * sin(angle);
        
        // Orbital velocity with some randomness
        float v_orbital = sqrt(100.0f / (r + 10.0f)) * (0.8f + 0.4f * (rand()/(float)RAND_MAX));
        float v_random = 2.0f * (rand()/(float)RAND_MAX - 0.5f);
        
        particles[i].vel.x = -v_orbital * sin(angle) + v_random;
        particles[i].vel.y = v_orbital * cos(angle) + v_random;
        
        particles[i].mass = mass_dist(gen);
        particles[i].radius = 1.0f;
        
        // Wrap around periodic boundaries
        while (particles[i].pos.x < 0) particles[i].pos.x += box_size;
        while (particles[i].pos.x >= box_size) particles[i].pos.x -= box_size;
        while (particles[i].pos.y < 0) particles[i].pos.y += box_size;
        while (particles[i].pos.y >= box_size) particles[i].pos.y -= box_size;
    }
    
    return particles;
}

int main() {
    std::cout << "=== 2 Million Particle Interactive Simulation ===\n\n";
    std::cout << "Initializing massive galaxy collision...\n";
    
    // Simulation parameters for 2M particles
    SimulationParams params;
    params.box_size = 10000.0f;
    params.gravity_constant = 0.5f;
    params.softening = 10.0f;
    params.dt = 0.05f;  // Smaller timestep for stability
    params.grid_size = 512;  // High resolution grid
    
    // Create 2 million particles!
    size_t n = 2000000;
    std::cout << "Creating " << n/1000000.0f << " million particles...\n";
    auto particles = createGalaxy(n, params.box_size);
    
    // Initialize backend with PM algorithm
    std::cout << "Initializing Particle Mesh backend (512x512 grid)...\n";
    auto backend = std::make_unique<SimpleBackend>();
    backend->setAlgorithm(ForceAlgorithm::PARTICLE_MESH);
    backend->initialize(n, params);
    backend->setParticles(particles);
    
    // Interactive viewer
    InteractiveViewer viewer(100, 40, 0.01f);  // Start zoomed out
    viewer.center_x = params.box_size / 2;
    viewer.center_y = params.box_size / 2;
    
    // Keyboard input handler
    KeyboardInput keyboard;
    
    // Timing
    auto last_frame = std::chrono::high_resolution_clock::now();
    auto last_fps_update = last_frame;
    float sim_time = 0;
    int frame_count = 0;
    float current_fps = 0;
    bool paused = false;
    
    std::cout << "\nStarting interactive simulation...\n";
    std::cout << "Press any key to begin.\n";
    std::this_thread::sleep_for(std::chrono::seconds(2));
    
    // Main simulation loop
    while (viewer.running) {
        auto frame_start = std::chrono::high_resolution_clock::now();
        
        // Handle input
        char key = keyboard.getKey();
        if (key) {
            if (key == ' ') {
                paused = !paused;
            } else {
                viewer.handleInput(key, params.box_size);
            }
        }
        
        // Run physics (if not paused)
        if (!paused) {
            backend->step(params.dt);
            sim_time += params.dt;
            
            // Get updated particles every few frames for visualization
            if (frame_count % 5 == 0) {
                backend->getParticles(particles);
            }
        }
        
        // Render
        viewer.plotParticles(particles, params.box_size);
        viewer.render(sim_time, current_fps, n);
        
        // Calculate FPS
        frame_count++;
        auto now = std::chrono::high_resolution_clock::now();
        auto fps_duration = std::chrono::duration<float>(now - last_fps_update).count();
        if (fps_duration > 0.5f) {  // Update FPS every 0.5 seconds
            current_fps = frame_count / fps_duration;
            frame_count = 0;
            last_fps_update = now;
        }
        
        // Frame rate limiting (target 30 FPS for smooth visualization)
        auto frame_duration = std::chrono::duration<float>(now - frame_start).count();
        float target_frame_time = 1.0f / 30.0f;
        if (frame_duration < target_frame_time) {
            std::this_thread::sleep_for(
                std::chrono::milliseconds((int)((target_frame_time - frame_duration) * 1000))
            );
        }
    }
    
    // Cleanup
    std::cout << "\n\nSimulation ended.\n";
    std::cout << "Final time: " << sim_time << " time units\n";
    std::cout << "Thank you for using DigiStar!\n";
    
    return 0;
}
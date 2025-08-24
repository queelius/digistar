#include "src/backend/SimpleBackend_v3.cpp"
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
#include <vector>
#include <string>

// Physical constants
const double G_REAL = 6.67430e-11;
const double AU = 1.496e11;
const double DAY = 86400;
const double SOLAR_MASS = 1.989e30;
const double EARTH_MASS = 5.972e24;
const double G_SIM = G_REAL * SOLAR_MASS * DAY * DAY / (AU * AU * AU);

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

// Enhanced viewer with comparison mode
class ComparisonViewer {
public:
    float center_x, center_y;
    float zoom;
    int width, height;
    std::vector<std::vector<char>> buffer_left;
    std::vector<std::vector<char>> buffer_right;
    std::atomic<bool> running{true};
    bool split_view;
    
    ComparisonViewer(int w = 120, int h = 40, float initial_zoom = 10.0f) 
        : width(w), height(h), zoom(initial_zoom), split_view(true) {
        buffer_left.resize(height, std::vector<char>(width/2, ' '));
        buffer_right.resize(height, std::vector<char>(width/2, ' '));
    }
    
    void clear() {
        for (auto& row : buffer_left) {
            std::fill(row.begin(), row.end(), ' ');
        }
        for (auto& row : buffer_right) {
            std::fill(row.begin(), row.end(), ' ');
        }
    }
    
    void plotPoint(float x, float y, char symbol, bool left_panel) {
        auto& buffer = left_panel ? buffer_left : buffer_right;
        int panel_width = width/2;
        
        float vx = (x - center_x) * zoom + panel_width/2;
        float vy = (y - center_y) * zoom + height/2;
        
        int sx = (int)vx;
        int sy = (int)vy;
        
        if (sx >= 0 && sx < panel_width && sy >= 0 && sy < height) {
            if (buffer[sy][sx] == ' ' || symbol == '*' || symbol == 'O') {
                buffer[sy][sx] = symbol;
            }
        }
    }
    
    void render(float sim_time, const std::string& left_algo, const std::string& right_algo,
                float left_fps, float right_fps, size_t n_particles) {
        std::cout << "\033[2J\033[H" << std::flush;
        
        // Title
        std::cout << "╔" << std::string(width-2, '═') << "╗\n";
        std::cout << "║ Backend Algorithm Comparison - " << n_particles << " particles";
        std::cout << std::string(width - 45, ' ') << "║\n";
        std::cout << "╠" << std::string(width/2-1, '═') << "╬" << std::string(width/2-1, '═') << "╣\n";
        
        // Algorithm headers
        std::cout << "║ " << std::left << std::setw(width/2-2) << left_algo 
                  << "║ " << std::setw(width/2-2) << right_algo << "║\n";
        std::cout << "╠" << std::string(width/2-1, '─') << "╬" << std::string(width/2-1, '─') << "╣\n";
        
        // Render both panels
        for (int y = 0; y < height; y++) {
            std::cout << "║";
            // Left panel
            for (char c : buffer_left[y]) {
                switch(c) {
                    case '*': std::cout << "\033[33;1m" << c << "\033[0m"; break;
                    case 'O': std::cout << "\033[36;1m" << c << "\033[0m"; break;
                    case 'o': std::cout << "\033[34m" << c << "\033[0m"; break;
                    case '.': std::cout << "\033[32m" << c << "\033[0m"; break;
                    default: std::cout << c;
                }
            }
            std::cout << "║";
            // Right panel
            for (char c : buffer_right[y]) {
                switch(c) {
                    case '*': std::cout << "\033[33;1m" << c << "\033[0m"; break;
                    case 'O': std::cout << "\033[36;1m" << c << "\033[0m"; break;
                    case 'o': std::cout << "\033[34m" << c << "\033[0m"; break;
                    case '.': std::cout << "\033[32m" << c << "\033[0m"; break;
                    default: std::cout << c;
                }
            }
            std::cout << "║\n";
        }
        
        // Status bars
        std::cout << "╠" << std::string(width/2-1, '─') << "╬" << std::string(width/2-1, '─') << "╣\n";
        
        std::cout << "║ FPS: " << std::fixed << std::setprecision(1) << std::setw(5) << left_fps 
                  << " | Time: " << std::setw(7) << sim_time << " days";
        std::cout << std::string(width/2 - 30, ' ');
        std::cout << "║ FPS: " << std::setw(5) << right_fps 
                  << " | Time: " << std::setw(7) << sim_time << " days";
        std::cout << std::string(width/2 - 30, ' ') << "║\n";
        
        // Controls
        std::cout << "╠" << std::string(width-2, '═') << "╣\n";
        std::cout << "║ Controls: WASD=pan  +/-=zoom  R=reset  B=switch backends  Q=quit  SPACE=pause";
        std::cout << std::string(width - 81, ' ') << "║\n";
        std::cout << "╚" << std::string(width-2, '═') << "╝\n";
    }
    
    void handleInput(char key) {
        float pan_speed = 0.5f / zoom;
        float zoom_speed = 1.2f;
        
        switch(key) {
            case 'w': case 'W': center_y -= pan_speed; break;
            case 's': case 'S': center_y += pan_speed; break;
            case 'a': case 'A': center_x -= pan_speed; break;
            case 'd': case 'D': center_x += pan_speed; break;
            case '+': case '=': zoom *= zoom_speed; break;
            case '-': case '_': zoom /= zoom_speed; break;
            case 'r': case 'R': 
                center_x = 0;
                center_y = 0; 
                zoom = 10.0f;
                break;
            case 'q': case 'Q':
                running = false;
                break;
        }
        
        zoom = std::max(0.1f, std::min(1000.0f, zoom));
    }
};

// Create simple test system
std::vector<Particle> createTestSystem(size_t n, float box_size, int system_type) {
    std::vector<Particle> particles;
    std::random_device rd;
    std::mt19937 gen(rd());
    
    float center = box_size / 2;
    
    if (system_type == 0) {
        // Simple binary system for accuracy test
        // Star 1
        Particle star1;
        star1.pos = {center - 0.5f, center};
        star1.vel = {0, 0.1f};
        star1.mass = 1.0f;
        star1.radius = 0.01f;
        particles.push_back(star1);
        
        // Star 2
        Particle star2;
        star2.pos = {center + 0.5f, center};
        star2.vel = {0, -0.1f};
        star2.mass = 1.0f;
        star2.radius = 0.01f;
        particles.push_back(star2);
        
    } else if (system_type == 1) {
        // Sun + planets system
        // Sun
        Particle sun;
        sun.pos = {center, center};
        sun.vel = {0, 0};
        sun.mass = 1.0f;
        sun.radius = 0.005f;
        particles.push_back(sun);
        
        // Add planets at different radii
        float radii[] = {0.39f, 0.72f, 1.0f, 1.52f, 5.2f};
        for (int i = 0; i < 5; i++) {
            Particle planet;
            float r = radii[i];
            float angle = 0;  // Start all at same angle to see differences
            
            planet.pos.x = center + r * cos(angle);
            planet.pos.y = center + r * sin(angle);
            
            float v_orbit = sqrt(G_SIM * sun.mass / r);
            planet.vel.x = -v_orbit * sin(angle);
            planet.vel.y = v_orbit * cos(angle);
            planet.mass = 0.001f;
            planet.radius = 0.001f;
            particles.push_back(planet);
        }
        
    } else {
        // Random cloud for performance test
        std::uniform_real_distribution<float> pos_dist(0, box_size);
        std::uniform_real_distribution<float> vel_dist(-0.1f, 0.1f);
        
        for (size_t i = 0; i < n; i++) {
            Particle p;
            p.pos.x = pos_dist(gen);
            p.pos.y = pos_dist(gen);
            p.vel.x = vel_dist(gen);
            p.vel.y = vel_dist(gen);
            p.mass = 1.0f / n;  // Equal mass distribution
            p.radius = 0.001f;
            particles.push_back(p);
        }
    }
    
    return particles;
}

// Calculate system energy for accuracy comparison
double calculateTotalEnergy(const std::vector<Particle>& particles, float G) {
    double kinetic = 0;
    double potential = 0;
    
    // Kinetic energy
    for (const auto& p : particles) {
        double v2 = p.vel.x * p.vel.x + p.vel.y * p.vel.y;
        kinetic += 0.5 * p.mass * v2;
    }
    
    // Potential energy
    for (size_t i = 0; i < particles.size(); i++) {
        for (size_t j = i + 1; j < particles.size(); j++) {
            float dx = particles[j].pos.x - particles[i].pos.x;
            float dy = particles[j].pos.y - particles[i].pos.y;
            float r = sqrt(dx*dx + dy*dy + 1e-6f);
            potential -= G * particles[i].mass * particles[j].mass / r;
        }
    }
    
    return kinetic + potential;
}

int main() {
    std::cout << "=== Backend Algorithm Comparison ===\n\n";
    std::cout << "Compare different force calculation algorithms:\n";
    std::cout << "  1. Brute Force (O(n²)) - Most accurate, slowest\n";
    std::cout << "  2. Barnes-Hut (O(n log n)) - Good accuracy, faster\n";
    std::cout << "  3. Particle Mesh (O(n)) - Approximate, fastest\n\n";
    
    std::cout << "Select test system:\n";
    std::cout << "  0. Binary system (2 bodies) - Best for accuracy test\n";
    std::cout << "  1. Solar system (6 bodies) - Planetary orbits\n";
    std::cout << "  2. Random cloud (1000 bodies) - Performance test\n";
    std::cout << "  3. Large cloud (10000 bodies) - Stress test\n";
    std::cout << "Choice [0-3]: ";
    
    int system_choice;
    std::cin >> system_choice;
    std::cin.ignore();  // Clear newline
    
    // Setup based on choice
    size_t n_particles = 2;
    int system_type = 0;
    
    switch(system_choice) {
        case 0: n_particles = 2; system_type = 0; break;
        case 1: n_particles = 6; system_type = 1; break;
        case 2: n_particles = 1000; system_type = 2; break;
        case 3: n_particles = 10000; system_type = 2; break;
        default: n_particles = 2; system_type = 0; break;
    }
    
    // Simulation parameters
    SimulationParams params;
    params.box_size = 20.0f;
    params.gravity_constant = G_SIM;
    params.softening = 0.001f;
    params.dt = 0.01f;
    params.grid_size = 256;
    params.theta = 0.5f;  // Barnes-Hut accuracy
    
    // Create initial particles
    std::cout << "\nCreating " << n_particles << " particle system...\n";
    auto initial_particles = createTestSystem(n_particles, params.box_size, system_type);
    
    // Create two backends to compare
    std::cout << "Initializing backends...\n";
    
    // Left backend - Brute Force (reference)
    auto backend_left = std::make_unique<SimpleBackend>();
    backend_left->setAlgorithm(ForceAlgorithm::BRUTE_FORCE);
    backend_left->initialize(initial_particles.size(), params);
    
    // Right backend - configurable
    auto backend_right = std::make_unique<SimpleBackend>();
    ForceAlgorithm right_algo = ForceAlgorithm::BARNES_HUT;  // Default
    backend_right->setAlgorithm(right_algo);
    backend_right->initialize(initial_particles.size(), params);
    
    // Set initial particles for both
    auto particles_left = initial_particles;
    auto particles_right = initial_particles;
    backend_left->setParticles(particles_left);
    backend_right->setParticles(particles_right);
    
    // Calculate initial energy
    double initial_energy = calculateTotalEnergy(initial_particles, params.gravity_constant);
    std::cout << "Initial total energy: " << std::scientific << initial_energy << "\n";
    
    // Viewer
    ComparisonViewer viewer(120, 30, 10.0f);
    viewer.center_x = 0;
    viewer.center_y = 0;
    
    // Keyboard input
    KeyboardInput keyboard;
    
    // Timing
    float sim_time = 0;
    bool paused = false;
    float left_fps = 0, right_fps = 0;
    
    // Algorithm names
    std::vector<std::string> algo_names = {
        "Brute Force (O(n²))",
        "Barnes-Hut (O(n log n))",
        "Particle Mesh (O(n))",
        "Hybrid"
    };
    
    std::cout << "\nStarting comparison...\n";
    std::cout << "Press 'b' to cycle through algorithms for right panel.\n";
    std::cout << "Press any key to begin.\n";
    std::this_thread::sleep_for(std::chrono::seconds(3));
    
    // Main loop
    auto last_frame = std::chrono::high_resolution_clock::now();
    
    while (viewer.running) {
        auto frame_start = std::chrono::high_resolution_clock::now();
        
        // Handle input
        char key = keyboard.getKey();
        if (key) {
            if (key == ' ') {
                paused = !paused;
            } else if (key == 'b' || key == 'B') {
                // Cycle through algorithms for right panel
                if (right_algo == ForceAlgorithm::BRUTE_FORCE) {
                    right_algo = ForceAlgorithm::BARNES_HUT;
                } else if (right_algo == ForceAlgorithm::BARNES_HUT) {
                    right_algo = ForceAlgorithm::PARTICLE_MESH;
                } else {
                    right_algo = ForceAlgorithm::BRUTE_FORCE;
                }
                backend_right->setAlgorithm(right_algo);
                
                // Reset particles to ensure same starting point
                particles_right = particles_left;
                backend_right->setParticles(particles_right);
            } else {
                viewer.handleInput(key);
            }
        }
        
        // Physics update
        if (!paused) {
            // Step left backend (Brute Force)
            auto left_start = std::chrono::high_resolution_clock::now();
            backend_left->step(params.dt);
            auto left_end = std::chrono::high_resolution_clock::now();
            float left_ms = std::chrono::duration<float, std::milli>(left_end - left_start).count();
            left_fps = 1000.0f / left_ms;
            
            // Step right backend
            auto right_start = std::chrono::high_resolution_clock::now();
            backend_right->step(params.dt);
            auto right_end = std::chrono::high_resolution_clock::now();
            float right_ms = std::chrono::duration<float, std::milli>(right_end - right_start).count();
            right_fps = 1000.0f / right_ms;
            
            sim_time += params.dt;
            
            // Get particles
            backend_left->getParticles(particles_left);
            backend_right->getParticles(particles_right);
        }
        
        // Calculate energy conservation (every 100 steps)
        static int energy_counter = 0;
        if (++energy_counter % 100 == 0) {
            double left_energy = calculateTotalEnergy(particles_left, params.gravity_constant);
            double right_energy = calculateTotalEnergy(particles_right, params.gravity_constant);
            
            std::cout << "\rEnergy drift - Left: " << std::scientific << std::setprecision(3) 
                      << (left_energy - initial_energy) / initial_energy * 100 << "% | Right: "
                      << (right_energy - initial_energy) / initial_energy * 100 << "%  " << std::flush;
        }
        
        // Render
        viewer.clear();
        
        // Plot particles in both panels
        for (size_t i = 0; i < particles_left.size(); i++) {
            float x_left = particles_left[i].pos.x - params.box_size/2;
            float y_left = particles_left[i].pos.y - params.box_size/2;
            float x_right = particles_right[i].pos.x - params.box_size/2;
            float y_right = particles_right[i].pos.y - params.box_size/2;
            
            char symbol = '.';
            if (i == 0 && system_type <= 1) symbol = '*';  // First body is star
            else if (system_type == 1 && i <= 5) symbol = 'o';  // Planets
            
            viewer.plotPoint(x_left, y_left, symbol, true);   // Left panel
            viewer.plotPoint(x_right, y_right, symbol, false); // Right panel
        }
        
        viewer.render(sim_time, algo_names[0], algo_names[(int)right_algo], 
                     left_fps, right_fps, particles_left.size());
        
        // Frame limiting
        auto frame_duration = std::chrono::duration<float>(
            std::chrono::high_resolution_clock::now() - frame_start).count();
        float target_frame_time = 1.0f / 30.0f;  // 30 FPS
        if (frame_duration < target_frame_time) {
            std::this_thread::sleep_for(
                std::chrono::milliseconds((int)((target_frame_time - frame_duration) * 1000))
            );
        }
    }
    
    // Final comparison
    std::cout << "\n\n=== Final Comparison ===\n";
    
    // Calculate position differences
    double max_diff = 0;
    double avg_diff = 0;
    for (size_t i = 0; i < particles_left.size(); i++) {
        float dx = particles_right[i].pos.x - particles_left[i].pos.x;
        float dy = particles_right[i].pos.y - particles_left[i].pos.y;
        double diff = sqrt(dx*dx + dy*dy);
        max_diff = std::max(max_diff, diff);
        avg_diff += diff;
    }
    avg_diff /= particles_left.size();
    
    std::cout << "Position difference (Right vs Brute Force):\n";
    std::cout << "  Average: " << std::scientific << avg_diff << " AU\n";
    std::cout << "  Maximum: " << max_diff << " AU\n";
    
    // Energy conservation
    double left_energy = calculateTotalEnergy(particles_left, params.gravity_constant);
    double right_energy = calculateTotalEnergy(particles_right, params.gravity_constant);
    
    std::cout << "\nEnergy conservation:\n";
    std::cout << "  Brute Force: " << (left_energy - initial_energy) / initial_energy * 100 << "%\n";
    std::cout << "  " << algo_names[(int)right_algo] << ": " 
              << (right_energy - initial_energy) / initial_energy * 100 << "%\n";
    
    std::cout << "\nSimulation time: " << std::fixed << sim_time << " days\n";
    
    return 0;
}
/**
 * Enhanced SDL2 Physics Viewer Demo
 * 
 * Demonstrates advanced SDL2 rendering with realistic physics:
 * - Gravitational attraction between particles
 * - Spring connections forming composite bodies  
 * - Color-coded rendering based on properties
 * - Interactive camera controls (pan/zoom)
 * - Performance metrics overlay
 */

#include <iostream>
#include <memory>
#include <vector>
#include <random>
#include <cmath>
#include <chrono>
#include <algorithm>
#include <SDL2/SDL.h>

struct Particle {
    float x, y;     // Position
    float vx, vy;   // Velocity  
    float fx, fy;   // Force accumulator
    float mass;     // Mass for physics
    float radius;   // Rendering size
    uint32_t color; // RGBA color
    uint32_t id;    // Unique identifier
};

struct Spring {
    uint32_t p1, p2;  // Particle indices
    float rest_length; // Equilibrium length
    float stiffness;   // Spring constant
    float damping;     // Damping factor
    uint32_t color;    // Rendering color
    bool active;
};

struct Camera {
    float x = 0, y = 0;    // Center position
    float zoom = 1.0f;     // Zoom level
};

class PhysicsViewer {
private:
    SDL_Window* window = nullptr;
    SDL_Renderer* renderer = nullptr;
    int width = 1920;
    int height = 1080;
    Camera camera;
    
    // Performance tracking
    struct {
        float fps = 0;
        float physics_time = 0;
        float render_time = 0;
        size_t particle_count = 0;
        size_t spring_count = 0;
    } stats;
    
    // Input state
    struct {
        bool mouse_left = false;
        bool mouse_right = false;
        int mouse_x = 0, mouse_y = 0;
        int mouse_drag_start_x = 0, mouse_drag_start_y = 0;
        float camera_start_x = 0, camera_start_y = 0;
    } input;
    
public:
    bool initialize(const std::string& title, int w, int h) {
        width = w;
        height = h;
        
        if (SDL_Init(SDL_INIT_VIDEO) < 0) {
            std::cerr << "SDL initialization failed: " << SDL_GetError() << std::endl;
            return false;
        }
        
        window = SDL_CreateWindow(title.c_str(),
                                 SDL_WINDOWPOS_CENTERED,
                                 SDL_WINDOWPOS_CENTERED,
                                 width, height,
                                 SDL_WINDOW_SHOWN | SDL_WINDOW_RESIZABLE);
        
        if (!window) {
            std::cerr << "Window creation failed: " << SDL_GetError() << std::endl;
            SDL_Quit();
            return false;
        }
        
        renderer = SDL_CreateRenderer(window, -1, 
                                    SDL_RENDERER_ACCELERATED | SDL_RENDERER_PRESENTVSYNC);
        
        if (!renderer) {
            std::cerr << "Renderer creation failed: " << SDL_GetError() << std::endl;
            SDL_DestroyWindow(window);
            SDL_Quit();
            return false;
        }
        
        return true;
    }
    
    void shutdown() {
        if (renderer) {
            SDL_DestroyRenderer(renderer);
            renderer = nullptr;
        }
        if (window) {
            SDL_DestroyWindow(window);
            window = nullptr;
        }
        SDL_Quit();
    }
    
    bool processEvents() {
        SDL_Event event;
        while (SDL_PollEvent(&event)) {
            switch (event.type) {
                case SDL_QUIT:
                    return false;
                    
                case SDL_MOUSEBUTTONDOWN:
                    if (event.button.button == SDL_BUTTON_LEFT) {
                        input.mouse_left = true;
                        input.mouse_drag_start_x = event.button.x;
                        input.mouse_drag_start_y = event.button.y;
                        input.camera_start_x = camera.x;
                        input.camera_start_y = camera.y;
                    } else if (event.button.button == SDL_BUTTON_RIGHT) {
                        input.mouse_right = true;
                    }
                    break;
                    
                case SDL_MOUSEBUTTONUP:
                    if (event.button.button == SDL_BUTTON_LEFT) {
                        input.mouse_left = false;
                    } else if (event.button.button == SDL_BUTTON_RIGHT) {
                        input.mouse_right = false;
                    }
                    break;
                    
                case SDL_MOUSEMOTION:
                    input.mouse_x = event.motion.x;
                    input.mouse_y = event.motion.y;
                    
                    // Camera panning
                    if (input.mouse_left) {
                        float dx = (event.motion.x - input.mouse_drag_start_x) / camera.zoom;
                        float dy = (event.motion.y - input.mouse_drag_start_y) / camera.zoom;
                        camera.x = input.camera_start_x - dx;
                        camera.y = input.camera_start_y - dy;
                    }
                    break;
                    
                case SDL_MOUSEWHEEL:
                    if (event.wheel.y > 0) {
                        camera.zoom *= 1.1f;
                    } else if (event.wheel.y < 0) {
                        camera.zoom /= 1.1f;
                    }
                    camera.zoom = std::clamp(camera.zoom, 0.01f, 100.0f);
                    break;
                    
                case SDL_WINDOWEVENT:
                    if (event.window.event == SDL_WINDOWEVENT_RESIZED) {
                        width = event.window.data1;
                        height = event.window.data2;
                    }
                    break;
                    
                case SDL_KEYDOWN:
                    switch (event.key.keysym.sym) {
                        case SDLK_ESCAPE:
                            return false;
                        case SDLK_r:  // Reset camera
                            camera.x = camera.y = 0;
                            camera.zoom = 1.0f;
                            break;
                    }
                    break;
            }
        }
        return true;
    }
    
    void worldToScreen(float world_x, float world_y, int& screen_x, int& screen_y) {
        screen_x = static_cast<int>((world_x - camera.x) * camera.zoom + width / 2);
        screen_y = static_cast<int>((world_y - camera.y) * camera.zoom + height / 2);
    }
    
    void render(const std::vector<Particle>& particles, const std::vector<Spring>& springs) {
        auto render_start = std::chrono::high_resolution_clock::now();
        
        // Clear screen
        SDL_SetRenderDrawColor(renderer, 5, 5, 15, 255);  // Dark blue background
        SDL_RenderClear(renderer);
        
        // Render springs first (so they appear behind particles)
        for (const auto& spring : springs) {
            if (!spring.active) continue;
            if (spring.p1 >= particles.size() || spring.p2 >= particles.size()) continue;
            
            const auto& p1 = particles[spring.p1];
            const auto& p2 = particles[spring.p2];
            
            int x1, y1, x2, y2;
            worldToScreen(p1.x, p1.y, x1, y1);
            worldToScreen(p2.x, p2.y, x2, y2);
            
            // Only render if at least one endpoint is on screen
            if ((x1 >= -50 && x1 < width + 50 && y1 >= -50 && y1 < height + 50) ||
                (x2 >= -50 && x2 < width + 50 && y2 >= -50 && y2 < height + 50)) {
                
                uint8_t r = (spring.color >> 16) & 0xFF;
                uint8_t g = (spring.color >> 8) & 0xFF;
                uint8_t b = spring.color & 0xFF;
                
                SDL_SetRenderDrawColor(renderer, r, g, b, 128);
                SDL_RenderDrawLine(renderer, x1, y1, x2, y2);
            }
        }
        
        // Render particles
        for (const auto& p : particles) {
            int screen_x, screen_y;
            worldToScreen(p.x, p.y, screen_x, screen_y);
            
            int radius = std::max(1, static_cast<int>(p.radius * camera.zoom));
            
            // Cull particles outside screen with margin
            if (screen_x < -radius || screen_x >= width + radius || 
                screen_y < -radius || screen_y >= height + radius) {
                continue;
            }
            
            uint8_t r = (p.color >> 16) & 0xFF;
            uint8_t g = (p.color >> 8) & 0xFF;
            uint8_t b = p.color & 0xFF;
            
            SDL_SetRenderDrawColor(renderer, r, g, b, 255);
            
            // Draw filled circle
            for (int dy = -radius; dy <= radius; dy++) {
                for (int dx = -radius; dx <= radius; dx++) {
                    if (dx*dx + dy*dy <= radius*radius) {
                        int px = screen_x + dx;
                        int py = screen_y + dy;
                        if (px >= 0 && px < width && py >= 0 && py < height) {
                            SDL_RenderDrawPoint(renderer, px, py);
                        }
                    }
                }
            }
        }
        
        // Render UI overlay
        renderUI();
        
        SDL_RenderPresent(renderer);
        
        auto render_end = std::chrono::high_resolution_clock::now();
        stats.render_time = std::chrono::duration<float, std::milli>(render_end - render_start).count();
        stats.particle_count = particles.size();
        stats.spring_count = springs.size();
    }
    
private:
    void renderUI() {
        // Simple text rendering using pixels (basic but functional)
        const int text_start_y = 10;
        const int line_height = 16;
        int line = 0;
        
        SDL_SetRenderDrawColor(renderer, 0, 255, 0, 255);  // Green text
        
        // Draw basic "text" using simple pixel patterns for FPS
        char fps_text[64];
        snprintf(fps_text, sizeof(fps_text), "FPS: %.1f", stats.fps);
        
        // Very basic "bitmap font" - just draw some rectangles to show info is available
        SDL_Rect info_bg = {5, 5, 200, 100};
        SDL_SetRenderDrawColor(renderer, 0, 0, 0, 128);
        SDL_RenderFillRect(renderer, &info_bg);
        
        SDL_SetRenderDrawColor(renderer, 0, 255, 0, 255);
        SDL_RenderDrawRect(renderer, &info_bg);
        
        // Draw simple indicators
        for (int i = 0; i < 5; i++) {
            SDL_Rect dot = {10, 10 + i * 15, 4, 4};
            SDL_RenderFillRect(renderer, &dot);
        }
    }
    
public:
    void updateStats(float dt) {
        if (dt > 0) {
            stats.fps = 1.0f / dt;
        }
    }
    
    const auto& getStats() const { return stats; }
    void setPhysicsTime(float time_ms) { stats.physics_time = time_ms; }
    
    ~PhysicsViewer() {
        shutdown();
    }
};

// Physics simulation functions
void computeGravity(std::vector<Particle>& particles, float G = 6.67e-11f) {
    // Clear forces
    for (auto& p : particles) {
        p.fx = p.fy = 0;
    }
    
    // O(N^2) gravity - simplified for demo
    for (size_t i = 0; i < particles.size(); i++) {
        for (size_t j = i + 1; j < particles.size(); j++) {
            auto& p1 = particles[i];
            auto& p2 = particles[j];
            
            float dx = p2.x - p1.x;
            float dy = p2.y - p1.y;
            float r2 = dx*dx + dy*dy;
            float r = std::sqrt(r2);
            
            if (r > 1e-6f) {  // Avoid division by zero
                float force = G * p1.mass * p2.mass / r2;
                float fx = force * dx / r;
                float fy = force * dy / r;
                
                p1.fx += fx;
                p1.fy += fy;
                p2.fx -= fx;
                p2.fy -= fy;
            }
        }
    }
}

void computeSprings(std::vector<Particle>& particles, std::vector<Spring>& springs) {
    for (auto& spring : springs) {
        if (!spring.active) continue;
        if (spring.p1 >= particles.size() || spring.p2 >= particles.size()) continue;
        
        auto& p1 = particles[spring.p1];
        auto& p2 = particles[spring.p2];
        
        float dx = p2.x - p1.x;
        float dy = p2.y - p1.y;
        float length = std::sqrt(dx*dx + dy*dy);
        
        if (length > 1e-6f) {
            float extension = length - spring.rest_length;
            float force_mag = spring.stiffness * extension;
            
            // Damping force
            float vrel_x = p2.vx - p1.vx;
            float vrel_y = p2.vy - p1.vy;
            float damping_force = spring.damping * (vrel_x * dx + vrel_y * dy) / length;
            
            float total_force = force_mag + damping_force;
            float fx = total_force * dx / length;
            float fy = total_force * dy / length;
            
            p1.fx += fx;
            p1.fy += fy;
            p2.fx -= fx;
            p2.fy -= fy;
            
            // Update spring color based on tension
            float tension = std::abs(extension / spring.rest_length);
            uint8_t red = std::min(255, static_cast<int>(tension * 255));
            spring.color = (red << 16) | (0 << 8) | (255 - red);
        }
    }
}

void integrate(std::vector<Particle>& particles, float dt) {
    for (auto& p : particles) {
        // Simple Euler integration  
        float ax = p.fx / p.mass;
        float ay = p.fy / p.mass;
        
        p.vx += ax * dt;
        p.vy += ay * dt;
        
        p.x += p.vx * dt;
        p.y += p.vy * dt;
        
        // Update particle color based on velocity magnitude
        float vel_mag = std::sqrt(p.vx * p.vx + p.vy * p.vy);
        uint8_t intensity = std::min(255, static_cast<int>(vel_mag * 10));
        p.color = (intensity << 16) | (128 << 8) | (255 - intensity);
    }
}

int main() {
    std::cout << "DigiStar Enhanced SDL2 Physics Viewer Demo\n";
    std::cout << "Controls:\n";
    std::cout << "  Left mouse: Pan camera\n";
    std::cout << "  Mouse wheel: Zoom\n";
    std::cout << "  R: Reset camera\n";
    std::cout << "  ESC: Exit\n\n";
    
    PhysicsViewer viewer;
    if (!viewer.initialize("DigiStar Enhanced Physics Demo", 1920, 1080)) {
        std::cerr << "Failed to initialize viewer\n";
        return -1;
    }
    
    // Create particle system
    std::vector<Particle> particles;
    std::vector<Spring> springs;
    
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> pos_dist(-400.0f, 400.0f);
    std::uniform_real_distribution<float> vel_dist(-10.0f, 10.0f);
    std::uniform_real_distribution<float> mass_dist(1.0f, 5.0f);
    
    // Create particles
    const size_t num_particles = 50;
    particles.reserve(num_particles);
    
    for (size_t i = 0; i < num_particles; i++) {
        Particle p;
        p.id = i;
        p.x = pos_dist(rng);
        p.y = pos_dist(rng);
        p.vx = vel_dist(rng);
        p.vy = vel_dist(rng);
        p.fx = p.fy = 0;
        p.mass = mass_dist(rng);
        p.radius = 3 + p.mass;
        p.color = 0xFF8080FF;  // Light blue
        
        particles.push_back(p);
    }
    
    // Create some springs to connect nearby particles
    for (size_t i = 0; i < particles.size(); i++) {
        for (size_t j = i + 1; j < particles.size(); j++) {
            float dx = particles[j].x - particles[i].x;
            float dy = particles[j].y - particles[i].y;
            float dist = std::sqrt(dx*dx + dy*dy);
            
            // Connect particles that are close together
            if (dist < 100.0f && springs.size() < num_particles) {
                Spring s;
                s.p1 = i;
                s.p2 = j;
                s.rest_length = dist;
                s.stiffness = 0.1f;
                s.damping = 0.01f;
                s.color = 0xFF4040FF;  // Cyan
                s.active = true;
                
                springs.push_back(s);
            }
        }
    }
    
    std::cout << "Created " << particles.size() << " particles and " << springs.size() << " springs\n";
    
    // Main simulation loop
    bool running = true;
    auto last_time = std::chrono::steady_clock::now();
    
    while (running) {
        auto current_time = std::chrono::steady_clock::now();
        float dt = std::chrono::duration<float>(current_time - last_time).count();
        last_time = current_time;
        
        // Limit timestep for stability
        dt = std::min(dt, 0.016f);
        
        if (!viewer.processEvents()) {
            running = false;
        }
        
        // Physics simulation
        auto physics_start = std::chrono::high_resolution_clock::now();
        
        computeGravity(particles, 100.0f);  // Scaled gravity for visibility
        computeSprings(particles, springs);
        integrate(particles, dt);
        
        auto physics_end = std::chrono::high_resolution_clock::now();
        float physics_time = std::chrono::duration<float, std::milli>(physics_end - physics_start).count();
        
        viewer.setPhysicsTime(physics_time);
        viewer.updateStats(dt);
        viewer.render(particles, springs);
    }
    
    std::cout << "Simulation completed.\n";
    std::cout << "Final stats:\n";
    const auto& stats = viewer.getStats();
    std::cout << "  Average FPS: " << stats.fps << "\n";
    std::cout << "  Physics time: " << stats.physics_time << " ms\n";
    std::cout << "  Render time: " << stats.render_time << " ms\n";
    
    return 0;
}
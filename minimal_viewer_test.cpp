/**
 * Minimal SDL2 Graphics Viewer Test
 * 
 * A simple test demonstrating that SDL2 integration works
 * with a basic particle rendering.
 */

#include <iostream>
#include <memory>
#include <vector>
#include <random>
#include <cmath>
#include <chrono>
#include <SDL2/SDL.h>

struct Particle {
    float x, y;
    float vx, vy;
    float mass;
    float radius;
    uint32_t color;
};

class MinimalViewer {
private:
    SDL_Window* window = nullptr;
    SDL_Renderer* renderer = nullptr;
    int width = 1280;
    int height = 720;
    
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
                                 SDL_WINDOW_SHOWN);
        
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
            if (event.type == SDL_QUIT) {
                return false;
            }
        }
        return true;
    }
    
    void render(const std::vector<Particle>& particles) {
        // Clear screen
        SDL_SetRenderDrawColor(renderer, 0, 0, 0, 255);
        SDL_RenderClear(renderer);
        
        // Render particles
        for (const auto& p : particles) {
            // Convert world coordinates to screen coordinates (simple mapping)
            int screen_x = static_cast<int>(p.x + width / 2);
            int screen_y = static_cast<int>(p.y + height / 2);
            
            if (screen_x >= 0 && screen_x < width && screen_y >= 0 && screen_y < height) {
                // Extract color components
                uint8_t r = (p.color >> 16) & 0xFF;
                uint8_t g = (p.color >> 8) & 0xFF;
                uint8_t b = p.color & 0xFF;
                
                SDL_SetRenderDrawColor(renderer, r, g, b, 255);
                
                // Draw particle as a small filled circle
                int radius = std::max(1, static_cast<int>(p.radius));
                for (int dy = -radius; dy <= radius; dy++) {
                    for (int dx = -radius; dx <= radius; dx++) {
                        if (dx*dx + dy*dy <= radius*radius) {
                            SDL_RenderDrawPoint(renderer, screen_x + dx, screen_y + dy);
                        }
                    }
                }
            }
        }
        
        SDL_RenderPresent(renderer);
    }
    
    ~MinimalViewer() {
        shutdown();
    }
};

int main() {
    std::cout << "Minimal SDL2 Graphics Viewer Test\n";
    
    // Create viewer
    MinimalViewer viewer;
    if (!viewer.initialize("DigiStar SDL2 Test", 1280, 720)) {
        std::cerr << "Failed to initialize viewer\n";
        return -1;
    }
    
    // Create some test particles
    std::vector<Particle> particles;
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> pos_dist(-300.0f, 300.0f);
    std::uniform_real_distribution<float> vel_dist(-20.0f, 20.0f);
    std::uniform_real_distribution<float> radius_dist(2.0f, 8.0f);
    
    const size_t num_particles = 100;
    particles.reserve(num_particles);
    
    for (size_t i = 0; i < num_particles; i++) {
        Particle p;
        p.x = pos_dist(rng);
        p.y = pos_dist(rng);
        p.vx = vel_dist(rng);
        p.vy = vel_dist(rng);
        p.mass = 1.0f + rng() % 10;
        p.radius = radius_dist(rng);
        
        // Color based on velocity magnitude
        float vel_mag = std::sqrt(p.vx * p.vx + p.vy * p.vy);
        uint8_t intensity = std::min(255, static_cast<int>(vel_mag * 6));
        p.color = (intensity << 16) | (128 << 8) | (255 - intensity);
        
        particles.push_back(p);
    }
    
    std::cout << "Created " << particles.size() << " test particles\n";
    std::cout << "Press ESC or close window to exit\n";
    
    // Main loop
    bool running = true;
    auto last_time = std::chrono::steady_clock::now();
    
    while (running) {
        if (!viewer.processEvents()) {
            running = false;
        }
        
        auto current_time = std::chrono::steady_clock::now();
        float dt = std::chrono::duration<float>(current_time - last_time).count();
        last_time = current_time;
        
        // Simple physics update
        for (auto& p : particles) {
            p.x += p.vx * dt;
            p.y += p.vy * dt;
            
            // Bounce off edges
            if (p.x < -400 || p.x > 400) p.vx = -p.vx;
            if (p.y < -300 || p.y > 300) p.vy = -p.vy;
        }
        
        viewer.render(particles);
        
        // Limit FPS
        SDL_Delay(16); // ~60 FPS
    }
    
    std::cout << "Shutting down viewer\n";
    return 0;
}
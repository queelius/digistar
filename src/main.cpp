/**
 * Scaled Composite Viewer - PM Solver + Spatial Grid Integration
 *
 * Scales up the simulation from ~450 to 450,000+ particles using:
 * - Particle Mesh (PM) solver for O(N log N) gravity
 * - Multi-resolution spatial grids for local interactions
 * - Optimized rendering with LOD
 */

#include <iostream>
#include <memory>
#include <vector>
#include <random>
#include <cmath>
#include <chrono>
#include <algorithm>
#include <cstring>
#include <cstdlib>
#include <string>
#include <atomic>
#include <SDL2/SDL.h>
// composite_system.h removed - using backend's composite system
#include "physics/pm_solver.h"
#include "physics/sparse_spatial_grid.h"
#include "physics/modular_physics_backend.h"
#include "physics/pm_gravity_backend.h"
#include "physics/cpu_collision_backend.h"
#include "physics/cpu_virtual_spring_backend.h"

using namespace digistar;

struct Particle {
    float x, y;
    float vx, vy;
    float ax, ay;  // Accelerations (used by PM solver)
    float fx, fy;  // Local forces (collisions, springs)
    float mass;
    float radius;
    uint32_t base_color;
    uint32_t display_color;
    uint32_t id;
    float temperature;
    bool pinned;
    bool selected;
};

// Spring struct now comes from virtual_spring_network_backend.h
using Spring = digistar::Spring;

struct Camera {
    float x = 0, y = 0;
    float zoom = 0.3f;  // Good zoom for star system view
};

class ScaledCompositeViewer {
private:
    SDL_Window* window = nullptr;
    SDL_Renderer* renderer = nullptr;
    int width = 1920;
    int height = 1080;
    Camera camera;

    // Physics systems
    std::unique_ptr<PMSolver> pm_solver;
    std::unique_ptr<SparseMultiResolutionGrid<Particle>> spatial_grid;
    std::unique_ptr<ModularPhysicsBackend<Particle>> physics_backend;
    // CompositeManager now handled by spring backend

    // Simulation parameters
    const float WORLD_SIZE = 10000.0f;
    const float dt = 0.01f;
    const int SUBSTEPS = 1;  // Reduced substeps for performance

    struct {
        float fps = 0;
        float physics_time = 0;
        float render_time = 0;
        float pm_solver_time = 0;
        float spatial_grid_time = 0;
        float collision_time = 0;
        int particles_rendered = 0;
        size_t spring_count = 0;
        size_t composite_count = 0;
        PMSolver::GridStats pm_stats;
        SparseMultiResolutionGrid<Particle>::MultiGridStats grid_stats;
    } stats;

    struct {
        bool mouse_left = false;
        bool mouse_middle = false;
        bool mouse_right = false;
        int mouse_x = 0, mouse_y = 0;
        int mouse_drag_start_x = 0, mouse_drag_start_y = 0;
        float camera_start_x = 0, camera_start_y = 0;
        int selected_particle = -1;
        float drag_offset_x = 0, drag_offset_y = 0;
        bool panning = false;
        bool show_stats = true;
        bool show_grid_overlay = false;
        bool ctrl_held = false;
        bool shift_held = false;
        bool pause_physics = false;
        bool use_lod = true;  // Level of detail rendering
    } input;

public:
    bool initialize(const std::string& title, int w, int h) {
        width = w;
        height = h;

        if (SDL_Init(SDL_INIT_VIDEO) < 0) {
            std::cerr << "SDL initialization failed: " << SDL_GetError() << std::endl;
            return false;
        }

        SDL_SetHint(SDL_HINT_RENDER_SCALE_QUALITY, "1");

        window = SDL_CreateWindow(
            title.c_str(),
            SDL_WINDOWPOS_CENTERED,
            SDL_WINDOWPOS_CENTERED,
            width, height,
            SDL_WINDOW_SHOWN | SDL_WINDOW_RESIZABLE
        );

        if (!window) {
            std::cerr << "Window creation failed: " << SDL_GetError() << std::endl;
            return false;
        }

        renderer = SDL_CreateRenderer(window, -1,
            SDL_RENDERER_ACCELERATED | SDL_RENDERER_PRESENTVSYNC);

        if (!renderer) {
            std::cerr << "Renderer creation failed: " << SDL_GetError() << std::endl;
            return false;
        }

        SDL_SetRenderDrawBlendMode(renderer, SDL_BLENDMODE_BLEND);

        // Initialize PM solver
        PMSolver::Config pm_config;
        pm_config.grid_size = 256;  // Higher resolution for more particles
        pm_config.box_size = WORLD_SIZE;
        pm_config.G = 50.0f;  // Reduced G for stability with many particles
        pm_config.softening = 5.0f;
        pm_config.use_toroidal = true;
        pm_solver = std::make_unique<PMSolver>(pm_config);
        pm_solver->initialize();

        // Initialize sparse spatial grid (optimized cell sizes for performance)
        SparseMultiResolutionGrid<Particle>::Config grid_config;
        grid_config.world_size = WORLD_SIZE;
        grid_config.toroidal = true;
        grid_config.contact_cell_size = 16.0f;  // Optimized from 4 units (8x fewer cells!)
        grid_config.spring_cell_size = 32.0f;   // Adjusted proportionally
        grid_config.thermal_cell_size = 128.0f;
        grid_config.radiation_cell_size = 512.0f;
        spatial_grid = std::make_unique<SparseMultiResolutionGrid<Particle>>(grid_config);

        // Initialize modular physics backend with all components
        // Gravity backend
        typename IGravityBackend<Particle>::Config gravity_config;
        gravity_config.G = 50.0f;
        gravity_config.grid_size = 256;
        gravity_config.box_size = WORLD_SIZE;
        gravity_config.softening = 5.0f;
        auto gravity_backend = std::make_unique<PMGravityBackend<Particle>>(gravity_config);

        // Collision backend
        typename ICollisionBackend<Particle>::Config collision_config;
        collision_config.contact_radius = 4.0f;
        collision_config.spring_stiffness = 500.0f;
        collision_config.damping_coefficient = 0.2f;
        collision_config.num_threads = 0;  // Auto-detect
        auto collision_backend = std::make_unique<CpuCollisionBackend<Particle>>(collision_config);

        // Virtual spring backend
        typename IVirtualSpringNetworkBackend<Particle>::Config spring_config;
        spring_config.formation_distance = 3.0f;
        spring_config.formation_velocity = 2.0f;
        spring_config.spring_stiffness = 100.0f;
        spring_config.spring_damping = 0.5f;
        spring_config.max_stretch = 2.0f;
        spring_config.max_force = 500.0f;
        spring_config.track_composites = true;
        spring_config.min_composite_size = 5;
        spring_config.max_springs_per_particle = 12;
        spring_config.max_total_springs = 100000;
        auto spring_backend = std::make_unique<CpuVirtualSpringBackend<Particle>>(spring_config);

        // Create modular backend
        physics_backend = std::make_unique<ModularPhysicsBackend<Particle>>(
            std::move(gravity_backend),
            std::move(collision_backend),
            std::move(spring_backend)
        );

        return true;
    }

    void cleanup() {
        // CompositeManager cleanup handled by spring backend

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

    // Create star system with planets
    void createStarSystem(std::vector<Particle>& particles, int num_particles) {
        std::mt19937 rng(42);
        std::normal_distribution<float> normal(0.0f, 1.0f);

        particles.clear();
        particles.reserve(num_particles);

        // Create central star (massive, made of many particles)
        int star_particles = 200;
        float star_mass_total = 5000.0f;
        float star_mass_per = star_mass_total / star_particles;

        for (int i = 0; i < star_particles; i++) {
            Particle p = {};

            // Distribute star particles in a sphere
            float r = std::pow(float(rng()) / RAND_MAX, 1.0f/3.0f) * 30.0f;
            float theta = float(rng()) / RAND_MAX * 2.0f * M_PI;
            float phi = acos(1 - 2 * float(rng()) / RAND_MAX);

            p.x = r * sin(phi) * cos(theta);
            p.y = r * sin(phi) * sin(theta);

            // Small internal velocities
            p.vx = normal(rng) * 0.5f;
            p.vy = normal(rng) * 0.5f;

            p.mass = star_mass_per;
            p.radius = 3.0f;
            p.base_color = 0xFFFFFF00;  // Yellow star
            p.display_color = p.base_color;
            p.id = particles.size();
            p.temperature = 100.0f;
            p.pinned = false;
            p.selected = false;

            particles.push_back(p);
        }

        // Create planets (5 major planets with surface detail)
        struct PlanetDef {
            float orbital_radius;
            float mass;
            int particle_count;
            uint32_t color;
            const char* name;
        };

        std::vector<PlanetDef> planets = {
            {200.0f,  50.0f,  500,  0xFF808080, "Mercury"},  // Gray
            {400.0f,  100.0f, 800,  0xFFFFAA00, "Venus"},    // Orange
            {600.0f,  150.0f, 1200, 0xFF0080FF, "Earth"},    // Blue
            {900.0f,  120.0f, 1000, 0xFFFF4040, "Mars"},     // Red
            {1400.0f, 300.0f, 2000, 0xFFAA8844, "Jupiter"},  // Brown
        };

        for (const auto& planet_def : planets) {
            // Calculate orbital velocity for circular orbit
            float v_orbital = std::sqrt(star_mass_total * 50.0f / planet_def.orbital_radius);

            // Random position on orbit
            float orbit_angle = float(rng()) / RAND_MAX * 2.0f * M_PI;
            float planet_cx = planet_def.orbital_radius * cos(orbit_angle);
            float planet_cy = planet_def.orbital_radius * sin(orbit_angle);

            // Planet velocity (perpendicular to radius)
            float planet_vx = -v_orbital * sin(orbit_angle);
            float planet_vy = v_orbital * cos(orbit_angle);

            float mass_per_particle = planet_def.mass / planet_def.particle_count;

            // Create planet with surface structure
            for (int i = 0; i < planet_def.particle_count; i++) {
                Particle p = {};

                // Sphere distribution for planet
                float r = std::pow(float(rng()) / RAND_MAX, 1.0f/3.0f) * 20.0f;

                // Add surface density variation (crust is denser)
                if (r > 15.0f) {
                    r = 15.0f + (r - 15.0f) * 2.0f;  // Surface layer
                }

                float theta = float(rng()) / RAND_MAX * 2.0f * M_PI;
                float phi = acos(1 - 2 * float(rng()) / RAND_MAX);

                p.x = planet_cx + r * sin(phi) * cos(theta);
                p.y = planet_cy + r * sin(phi) * sin(theta);

                // Planet orbital velocity plus small internal motion
                p.vx = planet_vx + normal(rng) * 0.1f;
                p.vy = planet_vy + normal(rng) * 0.1f;

                p.mass = mass_per_particle;
                p.radius = 1.5f;

                // Color variation for surface features
                uint32_t base = planet_def.color;
                int variation = (rng() % 40) - 20;
                uint8_t r_color = ((base >> 16) & 0xFF) + variation;
                uint8_t g_color = ((base >> 8) & 0xFF) + variation;
                uint8_t b_color = (base & 0xFF) + variation;
                p.base_color = 0xFF000000 | (r_color << 16) | (g_color << 8) | b_color;

                p.display_color = p.base_color;
                p.id = particles.size();
                p.temperature = 10.0f;
                p.pinned = false;
                p.selected = false;

                particles.push_back(p);
            }
        }

        // Add asteroid belt between Mars and Jupiter
        int asteroid_count = num_particles - particles.size();
        for (int i = 0; i < asteroid_count; i++) {
            Particle p = {};

            // Asteroid belt radius
            float r = 1100.0f + (float(rng()) / RAND_MAX - 0.5f) * 200.0f;
            float theta = float(rng()) / RAND_MAX * 2.0f * M_PI;

            p.x = r * cos(theta);
            p.y = r * sin(theta);

            // Orbital velocity with some scatter
            float v_orbital = std::sqrt(star_mass_total * 50.0f / r);
            p.vx = -v_orbital * sin(theta) + normal(rng) * 2.0f;
            p.vy = v_orbital * cos(theta) + normal(rng) * 2.0f;

            p.mass = 0.01f + float(rng()) / RAND_MAX * 0.05f;
            p.radius = 0.5f + p.mass * 10.0f;
            p.base_color = 0xFF606060;  // Dark gray
            p.display_color = p.base_color;
            p.id = particles.size();
            p.temperature = 1.0f;
            p.pinned = false;
            p.selected = false;

            particles.push_back(p);
        }

        // CompositeManager now handled by spring backend

        std::cout << "Created star system with " << particles.size() << " particles:\n";
        std::cout << "  - Central star: " << star_particles << " particles\n";
        std::cout << "  - 5 planets with surface structure\n";
        std::cout << "  - Asteroid belt: " << asteroid_count << " particles\n";
    }

    // Physics update using modular backend
    void updatePhysics(std::vector<Particle>& particles, std::vector<Spring>& springs) {
        if (input.pause_physics) return;

        auto t_start = std::chrono::high_resolution_clock::now();

        // Use modular backend if available
        if (physics_backend) {
            for (int substep = 0; substep < SUBSTEPS; substep++) {
                float sub_dt = dt / SUBSTEPS;
                physics_backend->step(particles, sub_dt);
            }

            // Get stats from backend
            auto backend_stats = physics_backend->getStats();
            stats.physics_time = backend_stats.total_ms;
            stats.pm_solver_time = backend_stats.gravity_ms;
            stats.spatial_grid_time = backend_stats.grid_update_ms;
            stats.collision_time = backend_stats.collision_ms;

            // Get spring stats if available
            if (auto* spring_backend = physics_backend->getSpringBackend()) {
                auto spring_stats = spring_backend->getStats();
                stats.spring_count = spring_stats.active_springs;
                stats.composite_count = spring_stats.num_composites;

                // Update springs vector for visualization
                const auto& backend_springs = spring_backend->getSprings();
                springs.clear();
                for (const auto& s : backend_springs) {
                    if (s.active) {
                        Spring spring;
                        spring.p1 = s.p1;
                        spring.p2 = s.p2;
                        spring.active = true;
                        springs.push_back(spring);
                    }
                }
            }
        } else {
            // Fallback to old physics (for compatibility)
            for (int substep = 0; substep < SUBSTEPS; substep++) {
                // Clear accelerations
                for (auto& p : particles) {
                    p.ax = 0;
                    p.ay = 0;
                    p.fx = 0;
                    p.fy = 0;
                }

                // Step 1: Compute gravity using PM solver
                auto t_pm_start = std::chrono::high_resolution_clock::now();
                pm_solver->computeForces(particles);
                auto t_pm_end = std::chrono::high_resolution_clock::now();
                stats.pm_solver_time = std::chrono::duration<float, std::milli>(t_pm_end - t_pm_start).count();

                // Step 2: Update spatial grids
                auto t_grid_start = std::chrono::high_resolution_clock::now();
                spatial_grid->update(particles, true);
                auto t_grid_end = std::chrono::high_resolution_clock::now();
                stats.spatial_grid_time = std::chrono::duration<float, std::milli>(t_grid_end - t_grid_start).count();

                // Step 3: Compute collisions
                auto t_collision_start = std::chrono::high_resolution_clock::now();
                computeCollisions(particles);
                auto t_collision_end = std::chrono::high_resolution_clock::now();
                stats.collision_time = std::chrono::duration<float, std::milli>(t_collision_end - t_collision_start).count();

                // Step 4: Spring forces
                if (!springs.empty()) {
                    computeSpringForces(particles, springs);
                }

                // Step 5: Integrate positions
                float sub_dt = dt / SUBSTEPS;
                for (auto& p : particles) {
                    if (p.pinned) continue;

                    float total_ax = p.ax + p.fx / p.mass;
                    float total_ay = p.ay + p.fy / p.mass;

                    p.vx += total_ax * sub_dt;
                    p.vy += total_ay * sub_dt;
                    p.x += p.vx * sub_dt;
                    p.y += p.vy * sub_dt;

                    wrapPosition(p);
                }
            }

            stats.pm_stats = pm_solver->getStats();
            stats.grid_stats = spatial_grid->getStats();

            auto t_end = std::chrono::high_resolution_clock::now();
            stats.physics_time = std::chrono::duration<float, std::milli>(t_end - t_start).count();
        }
    }

    // Compute collisions using spatial grid
    void computeCollisions(std::vector<Particle>& particles) {
        const float k_contact = 500.0f;
        const float damping = 0.2f;

        // Process collision pairs using sparse contact grid
        spatial_grid->processPairs(
            SparseMultiResolutionGrid<Particle>::CONTACT,
            particles,
            [&](int i, int j, float dist) {
                auto& p1 = particles[i];
                auto& p2 = particles[j];

                float min_dist = p1.radius + p2.radius;
                if (dist < min_dist && dist > 0.001f) {
                    float overlap = min_dist - dist;

                    // Direction from p2 to p1
                    float dx = p1.x - p2.x;
                    float dy = p1.y - p2.y;

                    // Handle toroidal wrapping
                    if (dx > WORLD_SIZE * 0.5f) dx -= WORLD_SIZE;
                    if (dx < -WORLD_SIZE * 0.5f) dx += WORLD_SIZE;
                    if (dy > WORLD_SIZE * 0.5f) dy -= WORLD_SIZE;
                    if (dy < -WORLD_SIZE * 0.5f) dy += WORLD_SIZE;

                    float dist_actual = std::sqrt(dx*dx + dy*dy);
                    if (dist_actual < 0.001f) return;

                    dx /= dist_actual;
                    dy /= dist_actual;

                    // Hertzian contact force
                    float force = k_contact * std::pow(overlap, 1.5f);

                    // Relative velocity for damping
                    float vrel_x = p1.vx - p2.vx;
                    float vrel_y = p1.vy - p2.vy;
                    float vrel_normal = vrel_x * dx + vrel_y * dy;

                    // Contact damping
                    float damping_force = -damping * vrel_normal * std::sqrt(overlap);
                    force += damping_force;

                    // Apply equal and opposite forces
                    p1.fx += force * dx;
                    p1.fy += force * dy;
                    p2.fx -= force * dx;
                    p2.fy -= force * dy;
                }
            }
        );
    }

    // Compute spring forces
    void computeSpringForces(std::vector<Particle>& particles, std::vector<Spring>& springs) {
        for (auto& spring : springs) {
            if (!spring.active) continue;
            if (spring.p1 >= particles.size() || spring.p2 >= particles.size()) continue;

            auto& p1 = particles[spring.p1];
            auto& p2 = particles[spring.p2];

            float dx = p2.x - p1.x;
            float dy = p2.y - p1.y;

            // Handle toroidal wrapping
            if (dx > WORLD_SIZE * 0.5f) dx -= WORLD_SIZE;
            if (dx < -WORLD_SIZE * 0.5f) dx += WORLD_SIZE;
            if (dy > WORLD_SIZE * 0.5f) dy -= WORLD_SIZE;
            if (dy < -WORLD_SIZE * 0.5f) dy += WORLD_SIZE;

            float dist = std::sqrt(dx * dx + dy * dy);
            if (dist < 0.001f) continue;

            float extension = dist - spring.rest_length;

            // Spring force
            float force = spring.stiffness * extension;

            // Check breaking condition using max_force
            if (std::abs(extension) > spring.rest_length * 2.0f ||
                std::abs(force) > spring.max_force) {
                spring.active = false;
                continue;
            }

            // Damping
            float vrel_x = p2.vx - p1.vx;
            float vrel_y = p2.vy - p1.vy;
            float vrel_along = (vrel_x * dx + vrel_y * dy) / dist;
            force += spring.damping * vrel_along;

            // Apply forces
            float fx = force * dx / dist;
            float fy = force * dy / dist;

            p1.fx += fx;
            p1.fy += fy;
            p2.fx -= fx;
            p2.fy -= fy;
        }
    }

    // Wrap position for toroidal topology
    void wrapPosition(Particle& p) {
        float half = WORLD_SIZE * 0.5f;

        if (p.x > half) p.x -= WORLD_SIZE;
        if (p.x < -half) p.x += WORLD_SIZE;
        if (p.y > half) p.y -= WORLD_SIZE;
        if (p.y < -half) p.y += WORLD_SIZE;
    }

    // Render with level of detail
    void render(const std::vector<Particle>& particles, const std::vector<Spring>& springs) {
        auto t_start = std::chrono::high_resolution_clock::now();

        SDL_SetRenderDrawColor(renderer, 10, 10, 20, 255);
        SDL_RenderClear(renderer);

        // Calculate LOD threshold based on zoom
        float lod_threshold = 2.0f / camera.zoom;  // Pixels
        stats.particles_rendered = 0;

        // Draw particles with LOD
        for (const auto& p : particles) {
            float screen_x = (p.x - camera.x) * camera.zoom + width / 2;
            float screen_y = (p.y - camera.y) * camera.zoom + height / 2;

            // Frustum culling
            if (screen_x < -50 || screen_x > width + 50 ||
                screen_y < -50 || screen_y > height + 50) {
                continue;
            }

            float screen_radius = p.radius * camera.zoom;

            // LOD: Skip very small particles
            if (input.use_lod && screen_radius < lod_threshold) {
                // Aggregate into pixel brightness instead
                int px = (int)screen_x;
                int py = (int)screen_y;
                if (px >= 0 && px < width && py >= 0 && py < height) {
                    // Just render as a point
                    uint8_t r = (p.display_color >> 16) & 0xFF;
                    uint8_t g = (p.display_color >> 8) & 0xFF;
                    uint8_t b = p.display_color & 0xFF;
                    SDL_SetRenderDrawColor(renderer, r, g, b, 100);
                    SDL_RenderDrawPoint(renderer, px, py);
                }
                continue;
            }

            stats.particles_rendered++;

            // Draw particle
            if (screen_radius > 1) {
                drawFilledCircle(screen_x, screen_y, screen_radius, p.display_color);
            } else {
                // Draw as point
                uint8_t r = (p.display_color >> 16) & 0xFF;
                uint8_t g = (p.display_color >> 8) & 0xFF;
                uint8_t b = p.display_color & 0xFF;
                SDL_SetRenderDrawColor(renderer, r, g, b, 255);
                SDL_RenderDrawPoint(renderer, (int)screen_x, (int)screen_y);
            }
        }

        // Draw grid overlay if enabled
        if (input.show_grid_overlay) {
            drawGridOverlay();
        }

        // Draw stats
        if (input.show_stats) {
            drawStats();
        }

        SDL_RenderPresent(renderer);

        auto t_end = std::chrono::high_resolution_clock::now();
        stats.render_time = std::chrono::duration<float, std::milli>(t_end - t_start).count();
    }

    void drawFilledCircle(int cx, int cy, int radius, uint32_t color) {
        uint8_t r = (color >> 16) & 0xFF;
        uint8_t g = (color >> 8) & 0xFF;
        uint8_t b = color & 0xFF;
        uint8_t a = (color >> 24) & 0xFF;

        SDL_SetRenderDrawColor(renderer, r, g, b, a);

        if (radius <= 1) {
            SDL_RenderDrawPoint(renderer, cx, cy);
            return;
        }

        for (int y = -radius; y <= radius; y++) {
            int x = (int)std::sqrt(radius * radius - y * y);
            SDL_RenderDrawLine(renderer, cx - x, cy + y, cx + x, cy + y);
        }
    }

    void drawGridOverlay() {
        SDL_SetRenderDrawColor(renderer, 50, 50, 100, 50);

        // Draw PM solver grid
        float cell_size = WORLD_SIZE / 256.0f;  // PM grid size

        for (int i = 0; i <= 256; i++) {
            float world_x = -WORLD_SIZE * 0.5f + i * cell_size;
            float screen_x = (world_x - camera.x) * camera.zoom + width / 2;

            if (screen_x >= 0 && screen_x < width) {
                SDL_RenderDrawLine(renderer, screen_x, 0, screen_x, height);
            }
        }

        for (int i = 0; i <= 256; i++) {
            float world_y = -WORLD_SIZE * 0.5f + i * cell_size;
            float screen_y = (world_y - camera.y) * camera.zoom + height / 2;

            if (screen_y >= 0 && screen_y < height) {
                SDL_RenderDrawLine(renderer, 0, screen_y, width, screen_y);
            }
        }
    }

    void drawStats() {
        int y = 10;
        char buffer[256];

        SDL_SetRenderDrawColor(renderer, 255, 255, 255, 200);

        snprintf(buffer, sizeof(buffer), "FPS: %.1f", stats.fps);
        drawText(10, y, buffer);
        y += 20;

        snprintf(buffer, sizeof(buffer), "Physics: %.1fms (PM: %.1fms, Grid: %.1fms, Coll: %.1fms)",
                 stats.physics_time, stats.pm_solver_time, stats.spatial_grid_time, stats.collision_time);
        drawText(10, y, buffer);
        y += 20;

        snprintf(buffer, sizeof(buffer), "Render: %.1fms (LOD: %d/%d particles)",
                 stats.render_time, stats.particles_rendered, (int)stats.grid_stats.contact_stats.total_particles);
        drawText(10, y, buffer);
        y += 20;

        snprintf(buffer, sizeof(buffer), "PM Grid: density [%.3f, %.3f] avg: %.3f",
                 stats.pm_stats.min_density, stats.pm_stats.max_density, stats.pm_stats.avg_density);
        drawText(10, y, buffer);
        y += 20;

        snprintf(buffer, sizeof(buffer), "Contact Grid: %zu cells, max %zu/cell",
                 stats.grid_stats.contact_stats.occupied_cells,
                 stats.grid_stats.contact_stats.max_particles_per_cell);
        drawText(10, y, buffer);
        y += 20;

        snprintf(buffer, sizeof(buffer), "Camera: (%.1f, %.1f) zoom: %.3f",
                 camera.x, camera.y, camera.zoom);
        drawText(10, y, buffer);
        y += 20;

        if (input.pause_physics) {
            SDL_SetRenderDrawColor(renderer, 255, 100, 100, 255);
            drawText(10, y, "PHYSICS PAUSED");
            y += 20;
        }

        // Controls
        y = height - 140;
        drawText(10, y, "Controls:");
        y += 20;
        drawText(10, y, "  Mouse: Pan (middle), Zoom (wheel)");
        y += 20;
        drawText(10, y, "  Space: Pause physics");
        y += 20;
        drawText(10, y, "  G: Toggle grid overlay");
        y += 20;
        drawText(10, y, "  S: Toggle stats");
        y += 20;
        drawText(10, y, "  L: Toggle LOD rendering");
    }

    void drawText(int x, int y, const char* text) {
        // Simple text rendering placeholder
        // In real implementation, use SDL_ttf or bitmap fonts
        SDL_Rect rect = {x, y, (int)(strlen(text) * 8), 16};
        SDL_SetRenderDrawColor(renderer, 0, 0, 0, 100);
        SDL_RenderFillRect(renderer, &rect);
    }

    bool handleEvents() {
        SDL_Event event;
        while (SDL_PollEvent(&event)) {
            switch (event.type) {
                case SDL_QUIT:
                    return false;

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
                        case SDLK_SPACE:
                            input.pause_physics = !input.pause_physics;
                            break;
                        case SDLK_s:
                            input.show_stats = !input.show_stats;
                            break;
                        case SDLK_g:
                            input.show_grid_overlay = !input.show_grid_overlay;
                            break;
                        case SDLK_l:
                            input.use_lod = !input.use_lod;
                            break;
                        case SDLK_LCTRL:
                        case SDLK_RCTRL:
                            input.ctrl_held = true;
                            break;
                        case SDLK_LSHIFT:
                        case SDLK_RSHIFT:
                            input.shift_held = true;
                            break;
                    }
                    break;

                case SDL_KEYUP:
                    switch (event.key.keysym.sym) {
                        case SDLK_LCTRL:
                        case SDLK_RCTRL:
                            input.ctrl_held = false;
                            break;
                        case SDLK_LSHIFT:
                        case SDLK_RSHIFT:
                            input.shift_held = false;
                            break;
                    }
                    break;

                case SDL_MOUSEBUTTONDOWN:
                    input.mouse_x = event.button.x;
                    input.mouse_y = event.button.y;

                    if (event.button.button == SDL_BUTTON_MIDDLE) {
                        input.mouse_middle = true;
                        input.panning = true;
                        input.camera_start_x = camera.x;
                        input.camera_start_y = camera.y;
                        input.mouse_drag_start_x = input.mouse_x;
                        input.mouse_drag_start_y = input.mouse_y;
                    }
                    break;

                case SDL_MOUSEBUTTONUP:
                    if (event.button.button == SDL_BUTTON_MIDDLE) {
                        input.mouse_middle = false;
                        input.panning = false;
                    }
                    break;

                case SDL_MOUSEMOTION:
                    input.mouse_x = event.motion.x;
                    input.mouse_y = event.motion.y;

                    if (input.panning) {
                        float dx = (input.mouse_x - input.mouse_drag_start_x) / camera.zoom;
                        float dy = (input.mouse_y - input.mouse_drag_start_y) / camera.zoom;
                        camera.x = input.camera_start_x - dx;
                        camera.y = input.camera_start_y - dy;
                    }
                    break;

                case SDL_MOUSEWHEEL:
                    {
                        float zoom_factor = 1.1f;
                        if (event.wheel.y > 0) {
                            camera.zoom *= zoom_factor;
                        } else if (event.wheel.y < 0) {
                            camera.zoom /= zoom_factor;
                        }
                        camera.zoom = std::max(0.001f, std::min(10.0f, camera.zoom));
                    }
                    break;
            }
        }
        return true;
    }

    void run(int argc, char* argv[]) {
        std::vector<Particle> particles;
        std::vector<Spring> springs;

        // Default to 10k particles for star system
        int num_particles = 10000;

        // Parse command line args for particle count
        if (argc > 1) {
            num_particles = std::atoi(argv[1]);
        }

        std::cout << "Starting with " << num_particles << " particles\n";
        createStarSystem(particles, num_particles);

        bool running = true;
        auto last_frame_time = std::chrono::high_resolution_clock::now();
        auto last_fps_time = last_frame_time;
        int frame_count = 0;

        while (running) {
            auto current_time = std::chrono::high_resolution_clock::now();

            // Handle events
            running = handleEvents();

            // Update physics
            updatePhysics(particles, springs);

            // Render
            render(particles, springs);

            // Update FPS counter
            frame_count++;
            auto fps_duration = std::chrono::duration<float>(current_time - last_fps_time).count();
            if (fps_duration > 1.0f) {
                stats.fps = frame_count / fps_duration;
                frame_count = 0;
                last_fps_time = current_time;
            }

            last_frame_time = current_time;
        }
    }
};

// Benchmark function
void runBenchmark() {
    std::cout << "DigiStar PM Solver Scalability Benchmark\n";
    std::cout << "=========================================\n\n";

    // Test configurations
    std::vector<int> particle_counts = {100, 500, 1000, 5000, 10000, 50000, 100000, 500000};
    const int warmup_steps = 5;
    const int benchmark_steps = 20;

    // CSV header
    std::cout << "# CSV Output for plotting:\n";
    std::cout << "particles,pm_solver_ms,spatial_grid_ms,collision_ms,integration_ms,total_ms,fps,memory_mb\n";

    // Initialize physics systems once
    PMSolver::Config pm_config;
    pm_config.grid_size = 256;
    pm_config.box_size = 10000.0f;
    pm_config.G = 50.0f;
    pm_config.softening = 5.0f;
    auto pm_solver = std::make_unique<PMSolver>(pm_config);
    pm_solver->initialize();

    SparseMultiResolutionGrid<Particle>::Config grid_config;
    grid_config.world_size = 10000.0f;
    auto spatial_grid = std::make_unique<SparseMultiResolutionGrid<Particle>>(grid_config);

    // Test each particle count
    for (int num_particles : particle_counts) {
        std::vector<Particle> particles;
        particles.reserve(num_particles);

        // Create particles with realistic distribution
        std::mt19937 rng(42);
        std::normal_distribution<float> normal(0.0f, 1.0f);

        for (int i = 0; i < num_particles; i++) {
            Particle p = {};
            float r = 2000.0f * std::pow(float(rng()) / RAND_MAX, 0.5f);
            float theta = float(rng()) / RAND_MAX * 2.0f * M_PI;
            p.x = r * cos(theta);
            p.y = r * sin(theta);

            // Circular velocity for stability
            float v_circular = std::sqrt(50.0f * 5000.0f / (r + 10.0f));
            p.vx = -v_circular * sin(theta) + normal(rng) * 2.0f;
            p.vy = v_circular * cos(theta) + normal(rng) * 2.0f;

            p.mass = 0.1f + float(rng()) / RAND_MAX * 0.2f;
            p.radius = 1.0f;
            p.ax = p.ay = 0;
            p.fx = p.fy = 0;
            particles.push_back(p);
        }

        // Timing accumulators
        double total_pm_time = 0;
        double total_grid_time = 0;
        double total_collision_time = 0;
        double total_integration_time = 0;
        double total_frame_time = 0;

        // Warmup runs
        for (int step = 0; step < warmup_steps; step++) {
            for (auto& p : particles) {
                p.ax = p.ay = 0;
                p.fx = p.fy = 0;
            }
            pm_solver->computeForces(particles);
            spatial_grid->update(particles, false);  // full rebuild for warmup
        }

        // Benchmark runs
        for (int step = 0; step < benchmark_steps; step++) {
            auto frame_start = std::chrono::high_resolution_clock::now();

            // Clear accelerations
            for (auto& p : particles) {
                p.ax = p.ay = 0;
                p.fx = p.fy = 0;
            }

            // PM solver
            auto t1 = std::chrono::high_resolution_clock::now();
            pm_solver->computeForces(particles);
            auto t2 = std::chrono::high_resolution_clock::now();
            total_pm_time += std::chrono::duration<double, std::milli>(t2 - t1).count();

            // Spatial grid update (incremental - only ~1% change cells)
            t1 = std::chrono::high_resolution_clock::now();
            spatial_grid->update(particles, true);  // incremental update
            t2 = std::chrono::high_resolution_clock::now();
            total_grid_time += std::chrono::duration<double, std::milli>(t2 - t1).count();

            // Collision detection
            t1 = std::chrono::high_resolution_clock::now();
            std::atomic<int> collision_pairs(0);
            spatial_grid->processPairs(
                SparseMultiResolutionGrid<Particle>::CONTACT,
                particles,
                [&](int i, int j, float dist) {
                    collision_pairs.fetch_add(1, std::memory_order_relaxed);
                    auto& p1 = particles[i];
                    auto& p2 = particles[j];

                    float min_dist = p1.radius + p2.radius;
                    if (dist < min_dist && dist > 0.001f) {
                        float overlap = min_dist - dist;
                        float dx = (p1.x - p2.x) / dist;
                        float dy = (p1.y - p2.y) / dist;
                        float force = 500.0f * std::pow(overlap, 1.5f);
                        float fx = force * dx;
                        float fy = force * dy;
                        #pragma omp atomic
                        p1.fx += fx;
                        #pragma omp atomic
                        p1.fy += fy;
                        #pragma omp atomic
                        p2.fx -= fx;
                        #pragma omp atomic
                        p2.fy -= fy;
                    }
                }
            );
            t2 = std::chrono::high_resolution_clock::now();
            total_collision_time += std::chrono::duration<double, std::milli>(t2 - t1).count();

            // Integration
            t1 = std::chrono::high_resolution_clock::now();
            float dt = 0.01f;
            for (auto& p : particles) {
                float total_ax = p.ax + p.fx / p.mass;
                float total_ay = p.ay + p.fy / p.mass;
                p.vx += total_ax * dt;
                p.vy += total_ay * dt;
                p.x += p.vx * dt;
                p.y += p.vy * dt;

                // Wrap for toroidal topology
                if (p.x > 5000.0f) p.x -= 10000.0f;
                if (p.x < -5000.0f) p.x += 10000.0f;
                if (p.y > 5000.0f) p.y -= 10000.0f;
                if (p.y < -5000.0f) p.y += 10000.0f;
            }
            t2 = std::chrono::high_resolution_clock::now();
            total_integration_time += std::chrono::duration<double, std::milli>(t2 - t1).count();

            auto frame_end = std::chrono::high_resolution_clock::now();
            total_frame_time += std::chrono::duration<double, std::milli>(frame_end - frame_start).count();
        }

        // Calculate averages
        double avg_pm_time = total_pm_time / benchmark_steps;
        double avg_grid_time = total_grid_time / benchmark_steps;
        double avg_collision_time = total_collision_time / benchmark_steps;
        double avg_integration_time = total_integration_time / benchmark_steps;
        double avg_frame_time = total_frame_time / benchmark_steps;
        double fps = 1000.0 / avg_frame_time;

        // Estimate memory usage (rough)
        double memory_mb = (num_particles * sizeof(Particle)) / (1024.0 * 1024.0);
        memory_mb += (256 * 256 * 4 * sizeof(float)) / (1024.0 * 1024.0); // PM grids

        // Output CSV row
        std::cout << num_particles << ","
                  << avg_pm_time << ","
                  << avg_grid_time << ","
                  << avg_collision_time << ","
                  << avg_integration_time << ","
                  << avg_frame_time << ","
                  << fps << ","
                  << memory_mb << "\n";

        // Also print human-readable summary
        if (num_particles <= 10000) {  // Only print details for smaller counts
            std::cerr << "\n" << num_particles << " particles:\n";
            std::cerr << "  PM Solver:    " << avg_pm_time << " ms\n";
            std::cerr << "  Spatial Grid: " << avg_grid_time << " ms\n";
            std::cerr << "  Collisions:   " << avg_collision_time << " ms\n";
            std::cerr << "  Integration:  " << avg_integration_time << " ms\n";
            std::cerr << "  Total:        " << avg_frame_time << " ms (" << fps << " FPS)\n";
        }
    }

    std::cout << "\n# Theoretical complexity analysis:\n";
    std::cout << "# PM Solver: O(N log N) - FFT-based\n";
    std::cout << "# Spatial Grid: O(N) - linear insertion\n";
    std::cout << "# Collisions: O(N) average, O(N²) worst case locally\n";
    std::cout << "# Integration: O(N) - linear update\n";
}

int main(int argc, char* argv[]) {
    // Check for benchmark mode
    bool benchmark = false;
    bool headless = false;

    for (int i = 1; i < argc; i++) {
        if (std::string(argv[i]) == "--benchmark") {
            benchmark = true;
        } else if (std::string(argv[i]) == "--headless") {
            headless = true;
        }
    }

    if (benchmark) {
        runBenchmark();
        return 0;
    }

    if (headless) {
        // Run headless performance test
        std::cout << "Running in headless mode (no graphics)\n";

        std::vector<Particle> particles;
        std::vector<Spring> springs;

        // Create star system with 10k particles
        std::mt19937 rng(42);
        std::normal_distribution<float> normal(0.0f, 1.0f);

        int num_particles = 10000;
        particles.reserve(num_particles);

        // Simple particle creation for testing
        for (int i = 0; i < num_particles; i++) {
            Particle p = {};
            float r = 1500.0f * std::pow(float(rng()) / RAND_MAX, 0.5f);
            float theta = float(rng()) / RAND_MAX * 2.0f * M_PI;
            p.x = r * cos(theta);
            p.y = r * sin(theta);
            p.vx = normal(rng) * 5.0f;
            p.vy = normal(rng) * 5.0f;
            p.mass = 0.1f + float(rng()) / RAND_MAX * 0.2f;
            p.radius = 1.0f;
            particles.push_back(p);
        }

        // Initialize physics systems
        PMSolver::Config pm_config;
        pm_config.grid_size = 256;
        pm_config.box_size = 10000.0f;
        pm_config.G = 50.0f;
        pm_config.softening = 5.0f;
        auto pm_solver = std::make_unique<PMSolver>(pm_config);
        pm_solver->initialize();

        SparseMultiResolutionGrid<Particle>::Config grid_config;
        grid_config.world_size = 10000.0f;
        auto spatial_grid = std::make_unique<SparseMultiResolutionGrid<Particle>>(grid_config);

        std::cout << "Testing star system with " << num_particles << " particles\n";

        // Run 10 physics steps
        for (int step = 0; step < 10; step++) {
            auto t1 = std::chrono::high_resolution_clock::now();

            // Clear accelerations
            for (auto& p : particles) {
                p.ax = p.ay = 0;
                p.fx = p.fy = 0;
            }

            // PM solver
            pm_solver->computeForces(particles);

            // Spatial grid
            spatial_grid->update(particles, true);  // incremental update

            // Integration
            float dt = 0.01f;
            for (auto& p : particles) {
                p.vx += p.ax * dt;
                p.vy += p.ay * dt;
                p.x += p.vx * dt;
                p.y += p.vy * dt;
            }

            auto t2 = std::chrono::high_resolution_clock::now();
            float ms = std::chrono::duration<float, std::milli>(t2 - t1).count();

            std::cout << "Step " << step << ": " << ms << " ms\n";
        }

        std::cout << "Headless test complete!\n";
        return 0;
    }

    // Normal GUI mode
    ScaledCompositeViewer viewer;

    if (!viewer.initialize("DigiStar - Scaled PM Solver Demo", 1920, 1080)) {
        std::cerr << "Failed to initialize viewer\n";
        return 1;
    }

    viewer.run(argc, argv);
    viewer.cleanup();

    return 0;
}
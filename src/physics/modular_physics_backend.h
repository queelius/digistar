/**
 * Modular Physics Backend
 *
 * Composes individual physics backends (gravity, collision, springs, etc.)
 * into a complete simulation backend. This allows mixing and matching
 * different implementations for each force type.
 *
 * Example:
 *   auto gravity = std::make_unique<PMGravityBackend<Particle>>(gravity_config);
 *   auto collision = std::make_unique<CpuCollisionBackend<Particle>>(collision_config);
 *   ModularPhysicsBackend<Particle> physics(std::move(gravity), std::move(collision));
 */

#pragma once

#include <memory>
#include <vector>
#include <chrono>
#include "gravity_backend.h"
#include "collision_backend.h"
#include "sparse_spatial_grid.h"

namespace digistar {

template<typename Particle>
class ModularPhysicsBackend {
public:
    struct Stats {
        double gravity_ms = 0;
        double collision_ms = 0;
        double spring_ms = 0;
        double grid_update_ms = 0;
        double integration_ms = 0;
        double total_ms = 0;
        size_t collision_pairs = 0;
        size_t active_springs = 0;
    };

    /**
     * Constructor with individual backend components
     */
    ModularPhysicsBackend(
        std::unique_ptr<IGravityBackend<Particle>> gravity_backend = nullptr,
        std::unique_ptr<ICollisionBackend<Particle>> collision_backend = nullptr)
        : gravity_backend_(std::move(gravity_backend)),
          collision_backend_(std::move(collision_backend)) {

        // Initialize spatial grid for collisions
        typename SparseSpatialGrid<Particle>::Config grid_config;
        grid_config.world_size = 10000.0f;
        grid_config.cell_size = 16.0f;  // Optimized cell size
        grid_config.toroidal = true;
        spatial_grid_ = std::make_unique<SparseSpatialGrid<Particle>>(grid_config);
    }

    /**
     * Perform one physics step
     * 1. Clear forces
     * 2. Compute gravity (if backend provided)
     * 3. Update spatial grid
     * 4. Compute collisions (if backend provided)
     * 5. Compute springs (if backend provided)
     * 6. Integrate positions and velocities
     */
    void step(std::vector<Particle>& particles, float dt) {
        auto frame_start = std::chrono::high_resolution_clock::now();

        // Clear forces
        for (auto& p : particles) {
            p.ax = 0;
            p.ay = 0;
            p.fx = 0;
            p.fy = 0;
        }

        // Gravity (long-range forces via PM solver)
        if (gravity_backend_) {
            auto t1 = std::chrono::high_resolution_clock::now();
            gravity_backend_->computeGravity(particles, dt);
            auto t2 = std::chrono::high_resolution_clock::now();
            stats_.gravity_ms = std::chrono::duration<double, std::milli>(t2 - t1).count();
        }

        // Update spatial grid (incremental update for efficiency)
        auto t1 = std::chrono::high_resolution_clock::now();
        if (grid_initialized_) {
            spatial_grid_->incrementalUpdate(particles);
        } else {
            spatial_grid_->rebuild(particles);
            grid_initialized_ = true;
        }
        auto t2 = std::chrono::high_resolution_clock::now();
        stats_.grid_update_ms = std::chrono::duration<double, std::milli>(t2 - t1).count();

        // Collisions (short-range contact forces)
        if (collision_backend_) {
            t1 = std::chrono::high_resolution_clock::now();
            collision_backend_->computeCollisions(particles, *spatial_grid_, dt);
            t2 = std::chrono::high_resolution_clock::now();
            stats_.collision_ms = std::chrono::duration<double, std::milli>(t2 - t1).count();

            auto collision_stats = collision_backend_->getStats();
            stats_.collision_pairs = collision_stats.pairs_checked;
        }

        // Springs (TODO: add spring backend)
        stats_.spring_ms = 0;
        stats_.active_springs = 0;

        // Integration (Leapfrog)
        t1 = std::chrono::high_resolution_clock::now();
        integrateParticles(particles, dt);
        t2 = std::chrono::high_resolution_clock::now();
        stats_.integration_ms = std::chrono::duration<double, std::milli>(t2 - t1).count();

        // Total time
        auto frame_end = std::chrono::high_resolution_clock::now();
        stats_.total_ms = std::chrono::duration<double, std::milli>(frame_end - frame_start).count();
    }

    /**
     * Get statistics from last step
     */
    Stats getStats() const { return stats_; }

    /**
     * Get estimated FPS based on last frame time
     */
    double getFPS() const {
        return stats_.total_ms > 0 ? 1000.0 / stats_.total_ms : 0;
    }

    /**
     * Set gravity backend
     */
    void setGravityBackend(std::unique_ptr<IGravityBackend<Particle>> backend) {
        gravity_backend_ = std::move(backend);
    }

    /**
     * Set collision backend
     */
    void setCollisionBackend(std::unique_ptr<ICollisionBackend<Particle>> backend) {
        collision_backend_ = std::move(backend);
    }

    /**
     * Get spatial grid for direct access
     */
    SparseSpatialGrid<Particle>& getSpatialGrid() { return *spatial_grid_; }

private:
    // Backend components
    std::unique_ptr<IGravityBackend<Particle>> gravity_backend_;
    std::unique_ptr<ICollisionBackend<Particle>> collision_backend_;
    // TODO: Add spring_backend_, thermal_backend_, etc.

    // Spatial indexing
    std::unique_ptr<SparseSpatialGrid<Particle>> spatial_grid_;
    bool grid_initialized_ = false;

    // Statistics
    mutable Stats stats_;

    /**
     * Integrate particle positions and velocities using Leapfrog
     */
    void integrateParticles(std::vector<Particle>& particles, float dt) {
        const float world_size = spatial_grid_->config.world_size;
        const float half_world = world_size * 0.5f;

        #pragma omp parallel for
        for (size_t i = 0; i < particles.size(); i++) {
            auto& p = particles[i];

            // Total acceleration = gravity + contact_forces/mass
            float total_ax = p.ax + p.fx / p.mass;
            float total_ay = p.ay + p.fy / p.mass;

            // Leapfrog integration
            p.vx += total_ax * dt;
            p.vy += total_ay * dt;
            p.x += p.vx * dt;
            p.y += p.vy * dt;

            // Toroidal wrapping
            if (p.x > half_world) p.x -= world_size;
            if (p.x < -half_world) p.x += world_size;
            if (p.y > half_world) p.y -= world_size;
            if (p.y < -half_world) p.y += world_size;
        }
    }
};

} // namespace digistar
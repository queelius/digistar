/**
 * Collision Detection Backend Interface
 *
 * Allows swapping between different collision detection strategies
 * to find the optimal approach for different particle counts and distributions.
 */

#pragma once

#include <vector>
#include <string>
#include <memory>
#include <chrono>
#include "sparse_spatial_grid.h"

namespace digistar {

// Forward declaration
template<typename Particle>
class SparseSpatialGrid;

/**
 * Statistics for backend performance analysis
 */
struct CollisionStats {
    size_t pairs_checked = 0;
    size_t collisions_found = 0;
    double computation_time_ms = 0;
    double efficiency = 0;  // collisions_found / pairs_checked
    size_t memory_bytes_used = 0;
};

/**
 * Abstract base class for collision detection backends
 */
template<typename Particle>
class ICollisionBackend {
public:
    struct Config {
        float contact_radius = 16.0f;
        float spring_stiffness = 500.0f;
        float damping_coefficient = 0.2f;
        int num_threads = 1;
        bool enable_simd = false;
    };

    virtual ~ICollisionBackend() = default;

    /**
     * Compute collision forces for all particles
     * Updates particle fx, fy fields with collision forces
     */
    virtual void computeCollisions(
        std::vector<Particle>& particles,
        SparseSpatialGrid<Particle>& grid,
        float dt) = 0;

    /**
     * Get backend name for identification
     */
    virtual std::string getName() const = 0;

    /**
     * Get description of the approach
     */
    virtual std::string getDescription() const = 0;

    /**
     * Get last computation statistics
     */
    virtual CollisionStats getStats() const { return stats_; }

    /**
     * Set configuration
     */
    virtual void setConfig(const Config& config) { config_ = config; }

    /**
     * Check if backend supports parallel execution
     */
    virtual bool supportsParallel() const { return false; }

    /**
     * Check if backend supports SIMD
     */
    virtual bool supportsSimd() const { return false; }

protected:
    Config config_;
    mutable CollisionStats stats_;

    /**
     * Helper: Calculate Hertzian contact force
     */
    inline float calculateContactForce(float overlap) const {
        return config_.spring_stiffness * std::pow(overlap, 1.5f);
    }

    /**
     * Helper: Start timing
     */
    inline auto startTimer() const {
        return std::chrono::high_resolution_clock::now();
    }

    /**
     * Helper: End timing and update stats
     */
    inline void endTimer(const std::chrono::high_resolution_clock::time_point& start_time) const {
        auto end_time = std::chrono::high_resolution_clock::now();
        stats_.computation_time_ms = std::chrono::duration<double, std::milli>(
            end_time - start_time).count();
    }
};

/**
 * Single-threaded reference implementation
 * This is our baseline for comparison
 */
template<typename Particle>
class SingleThreadedBackend : public ICollisionBackend<Particle> {
public:
    std::string getName() const override {
        return "SingleThreaded";
    }

    std::string getDescription() const override {
        return "Single-threaded reference implementation (baseline)";
    }

    void computeCollisions(
        std::vector<Particle>& particles,
        SparseSpatialGrid<Particle>& grid,
        float dt) override {

        auto start = this->startTimer();
        this->stats_.pairs_checked = 0;
        this->stats_.collisions_found = 0;

        int cell_radius = std::ceil(this->config_.contact_radius / grid.config.cell_size);
        float radius2 = this->config_.contact_radius * this->config_.contact_radius;

        // Convert to vector for OpenMP
        std::vector<std::pair<uint64_t, std::vector<uint32_t>>> cell_list;
        cell_list.reserve(grid.cells.size());
        for (const auto& [key, particles] : grid.cells) {
            cell_list.push_back({key, particles});
        }

        // Iterate through all occupied cells with OpenMP
        #pragma omp parallel for schedule(dynamic, 64)
        for (size_t cell_idx = 0; cell_idx < cell_list.size(); cell_idx++) {
            const auto& [center_key, center_particles] = cell_list[cell_idx];
            // Decode cell coordinates
            int cx = (center_key >> 32) & 0xFFFFFFFF;
            int cy = center_key & 0xFFFFFFFF;

            // Process pairs within the same cell
            for (size_t i = 0; i < center_particles.size(); i++) {
                for (size_t j = i + 1; j < center_particles.size(); j++) {
                    uint32_t idx1 = center_particles[i];
                    uint32_t idx2 = center_particles[j];
                    this->stats_.pairs_checked++;

                    float dx = particles[idx1].x - particles[idx2].x;
                    float dy = particles[idx1].y - particles[idx2].y;
                    float dist2 = dx * dx + dy * dy;

                    float min_dist = particles[idx1].radius + particles[idx2].radius;
                    if (dist2 < min_dist * min_dist && dist2 > 0.001f) {
                        this->stats_.collisions_found++;

                        float dist = std::sqrt(dist2);
                        float overlap = min_dist - dist;
                        dx /= dist;
                        dy /= dist;

                        // Hertzian contact force
                        float force = this->calculateContactForce(overlap);

                        // Relative velocity for damping
                        float vrel_x = particles[idx1].vx - particles[idx2].vx;
                        float vrel_y = particles[idx1].vy - particles[idx2].vy;
                        float vrel_normal = vrel_x * dx + vrel_y * dy;

                        // Add damping
                        force -= this->config_.damping_coefficient * vrel_normal * std::sqrt(overlap);

                        // Apply forces
                        float fx = force * dx;
                        float fy = force * dy;

                        #pragma omp atomic
                        particles[idx1].fx += fx;
                        #pragma omp atomic
                        particles[idx1].fy += fy;
                        #pragma omp atomic
                        particles[idx2].fx -= fx;
                        #pragma omp atomic
                        particles[idx2].fy -= fy;
                    }
                }
            }

            // Check neighboring cells
            for (int dy = -cell_radius; dy <= cell_radius; dy++) {
                for (int dx = -cell_radius; dx <= cell_radius; dx++) {
                    if (dx == 0 && dy == 0) continue;
                    // Only process half to avoid duplicates
                    if (dx < 0 || (dx == 0 && dy < 0)) continue;

                    uint64_t neighbor_key = grid.hashCell(cx + dx, cy + dy);
                    auto neighbor_it = grid.cells.find(neighbor_key);
                    if (neighbor_it == grid.cells.end()) continue;

                    const auto& neighbor_particles = neighbor_it->second;

                    // Check all pairs between cells
                    for (uint32_t idx1 : center_particles) {
                        for (uint32_t idx2 : neighbor_particles) {
                            float dx = particles[idx1].x - particles[idx2].x;
                            float dy = particles[idx1].y - particles[idx2].y;

                            // Handle toroidal wrapping
                            if (grid.config.toroidal) {
                                if (dx > grid.config.world_size * 0.5f) dx -= grid.config.world_size;
                                if (dx < -grid.config.world_size * 0.5f) dx += grid.config.world_size;
                                if (dy > grid.config.world_size * 0.5f) dy -= grid.config.world_size;
                                if (dy < -grid.config.world_size * 0.5f) dy += grid.config.world_size;
                            }

                            float dist2 = dx * dx + dy * dy;

                            // Skip if beyond contact radius
                            if (dist2 > radius2) continue;

                            this->stats_.pairs_checked++;

                            float min_dist = particles[idx1].radius + particles[idx2].radius;

                            if (dist2 < min_dist * min_dist && dist2 > 0.001f) {
                                this->stats_.collisions_found++;

                                float dist = std::sqrt(dist2);
                                float overlap = min_dist - dist;
                                dx /= dist;
                                dy /= dist;

                                float force = this->calculateContactForce(overlap);

                                float vrel_x = particles[idx1].vx - particles[idx2].vx;
                                float vrel_y = particles[idx1].vy - particles[idx2].vy;
                                float vrel_normal = vrel_x * dx + vrel_y * dy;

                                force -= this->config_.damping_coefficient * vrel_normal * std::sqrt(overlap);

                                float fx = force * dx;
                                float fy = force * dy;

                                particles[idx1].fx += fx;
                                particles[idx1].fy += fy;
                                particles[idx2].fx -= fx;
                                particles[idx2].fy -= fy;
                            }
                        }
                    }
                }
            }
        }

        this->endTimer(start);

        if (this->stats_.pairs_checked > 0) {
            this->stats_.efficiency = double(this->stats_.collisions_found) /
                                      double(this->stats_.pairs_checked);
        }
    }
};

/**
 * Thread-local accumulation backend
 * Each thread accumulates forces locally, then merges
 */
template<typename Particle>
class ThreadLocalBackend : public ICollisionBackend<Particle> {
public:
    std::string getName() const override {
        return "ThreadLocal";
    }

    std::string getDescription() const override {
        return "Thread-local force accumulation with post-merge";
    }

    bool supportsParallel() const override { return true; }

    void computeCollisions(
        std::vector<Particle>& particles,
        SparseSpatialGrid<Particle>& grid,
        float dt) override;  // Implementation in .cpp file
};

/**
 * Spatial tiling backend
 * Divides space into tiles, one thread per tile
 */
template<typename Particle>
class TiledBackend : public ICollisionBackend<Particle> {
public:
    std::string getName() const override {
        return "Tiled";
    }

    std::string getDescription() const override {
        return "Spatial tiling with one thread per tile region";
    }

    bool supportsParallel() const override { return true; }

    void computeCollisions(
        std::vector<Particle>& particles,
        SparseSpatialGrid<Particle>& grid,
        float dt) override;  // Implementation in .cpp file
};

/**
 * Atomic operations backend (for comparison)
 */
template<typename Particle>
class AtomicBackend : public ICollisionBackend<Particle> {
public:
    std::string getName() const override {
        return "Atomic";
    }

    std::string getDescription() const override {
        return "OpenMP parallel with atomic force updates";
    }

    bool supportsParallel() const override { return true; }

    void computeCollisions(
        std::vector<Particle>& particles,
        SparseSpatialGrid<Particle>& grid,
        float dt) override;  // Implementation in .cpp file
};

/**
 * Factory for creating backends
 */
template<typename Particle>
class CollisionBackendFactory {
public:
    enum BackendType {
        SINGLE_THREADED,
        THREAD_LOCAL,
        TILED,
        ATOMIC
    };

    static std::unique_ptr<ICollisionBackend<Particle>> create(
        BackendType type,
        const typename ICollisionBackend<Particle>::Config& config = {}) {

        std::unique_ptr<ICollisionBackend<Particle>> backend;

        switch (type) {
            case SINGLE_THREADED:
                backend = std::make_unique<SingleThreadedBackend<Particle>>();
                break;
            case THREAD_LOCAL:
                backend = std::make_unique<ThreadLocalBackend<Particle>>();
                break;
            case TILED:
                backend = std::make_unique<TiledBackend<Particle>>();
                break;
            case ATOMIC:
                backend = std::make_unique<AtomicBackend<Particle>>();
                break;
            default:
                backend = std::make_unique<SingleThreadedBackend<Particle>>();
        }

        backend->setConfig(config);
        return backend;
    }

    static std::vector<BackendType> getAllTypes() {
        return {SINGLE_THREADED, THREAD_LOCAL, TILED, ATOMIC};
    }

    static std::string getTypeName(BackendType type) {
        switch (type) {
            case SINGLE_THREADED: return "SingleThreaded";
            case THREAD_LOCAL: return "ThreadLocal";
            case TILED: return "Tiled";
            case ATOMIC: return "Atomic";
            default: return "Unknown";
        }
    }
};

} // namespace digistar
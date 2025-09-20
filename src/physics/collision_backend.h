/**
 * Collision Detection Backend Interface
 *
 * Abstract interface for collision detection implementations.
 * This allows for different strategies (CPU, GPU, etc.) to be swapped.
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
        float contact_radius = 4.0f;       // Detection radius (typically 2-4x particle radius)
        float spring_stiffness = 500.0f;   // Hertzian contact stiffness
        float damping_coefficient = 0.2f;  // Velocity damping
        int num_threads = 0;               // 0 = auto-detect
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

} // namespace digistar
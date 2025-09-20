/**
 * Gravity Backend Interface
 *
 * Abstract interface for gravity computation implementations.
 * This allows for different strategies (PM, direct N-body, tree codes, etc.) to be swapped.
 */

#pragma once

#include <vector>
#include <string>
#include <memory>

namespace digistar {

/**
 * Statistics for gravity computation performance
 */
struct GravityStats {
    double deposit_time_ms = 0;      // Time for mass deposition (PM)
    double fft_time_ms = 0;          // Time for FFT operations (PM)
    double interpolate_time_ms = 0;  // Time for force interpolation (PM)
    double tree_build_time_ms = 0;   // Time for tree construction (tree codes)
    double force_calc_time_ms = 0;   // Time for direct force calculations
    double total_time_ms = 0;        // Total computation time
    size_t interactions_computed = 0; // Number of particle interactions
    size_t num_particles = 0;        // Number of particles processed
};

/**
 * Abstract base class for gravity computation backends
 */
template<typename Particle>
class IGravityBackend {
public:
    struct Config {
        float G = 50.0f;             // Gravitational constant
        float softening = 5.0f;      // Force softening length
        float theta = 0.5f;          // Opening angle for tree codes
        int grid_size = 256;         // Grid resolution for PM
        float box_size = 10000.0f;   // World size
    };

    virtual ~IGravityBackend() = default;

    /**
     * Compute gravitational forces/accelerations for all particles
     * Updates particle ax, ay fields with gravitational accelerations
     */
    virtual void computeGravity(
        std::vector<Particle>& particles,
        float dt) = 0;

    /**
     * Get backend name for identification
     */
    virtual std::string getName() const = 0;

    /**
     * Get last computation statistics
     */
    virtual GravityStats getStats() const { return stats_; }

    /**
     * Set configuration
     */
    virtual void setConfig(const Config& config) { config_ = config; }

    /**
     * Get configuration
     */
    virtual const Config& getConfig() const { return config_; }

protected:
    Config config_;
    mutable GravityStats stats_;
};

} // namespace digistar
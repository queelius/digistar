/**
 * Particle Mesh (PM) Gravity Backend
 *
 * Computes long-range gravitational forces using FFT-based Particle Mesh method.
 * Achieves O(N log N) scaling for N-body gravity calculations.
 *
 * Key features:
 * - Cloud-in-Cell (CIC) mass deposition
 * - FFT-based Poisson solver
 * - Toroidal boundary conditions
 * - Configurable grid resolution and softening
 */

#pragma once

#include <vector>
#include <memory>
#include <chrono>
#include "pm_solver.h"

namespace digistar {

/**
 * PM gravity computation statistics
 */
struct GravityStats {
    double deposit_time_ms = 0;
    double fft_time_ms = 0;
    double interpolate_time_ms = 0;
    double total_time_ms = 0;
    size_t grid_cells = 0;
    size_t num_particles = 0;
};

/**
 * PM-based gravity backend for long-range forces
 */
template<typename Particle>
class PMGravityBackend {
public:
    struct Config {
        int grid_size = 256;        // Grid resolution (NxN)
        float box_size = 10000.0f;  // World size
        float G = 50.0f;            // Gravitational constant
        float softening = 5.0f;     // Force softening length
    };

    PMGravityBackend(const Config& config = Config()) : config_(config) {
        // Create PM solver configuration
        PMSolver::Config pm_config;
        pm_config.grid_size = config_.grid_size;
        pm_config.box_size = config_.box_size;
        pm_config.G = config_.G;
        pm_config.softening = config_.softening;

        // Initialize PM solver
        pm_solver_ = std::make_unique<PMSolver>(pm_config);
        pm_solver_->initialize();

        stats_.grid_cells = config_.grid_size * config_.grid_size;
    }

    /**
     * Compute gravitational forces for all particles
     * Updates particle ax, ay fields with gravitational accelerations
     */
    void computeGravity(std::vector<Particle>& particles, float dt) {
        auto start = startTimer();
        stats_.num_particles = particles.size();

        // Clear accelerations
        for (auto& p : particles) {
            p.ax = 0;
            p.ay = 0;
        }

        // Compute forces using PM solver
        auto t1 = startTimer();
        pm_solver_->computeForces(particles);
        auto t2 = startTimer();

        // For now, we don't have detailed timing from PM solver
        // TODO: Add timing to PM solver
        stats_.deposit_time_ms = 0;
        stats_.fft_time_ms = 0;
        stats_.interpolate_time_ms = 0;

        endTimer(start);
    }

    /**
     * Get last computation statistics
     */
    GravityStats getStats() const { return stats_; }

    /**
     * Get backend name
     */
    std::string getName() const { return "PMGravityBackend"; }

    /**
     * Get configuration
     */
    const Config& getConfig() const { return config_; }

    /**
     * Get PM solver grid statistics
     */
    PMSolver::GridStats getGridStats() const {
        return pm_solver_->getStats();
    }

private:
    Config config_;
    std::unique_ptr<PMSolver> pm_solver_;
    mutable GravityStats stats_;

    /**
     * Start timing
     */
    inline auto startTimer() const {
        return std::chrono::high_resolution_clock::now();
    }

    /**
     * End timing and update stats
     */
    inline void endTimer(const std::chrono::high_resolution_clock::time_point& start_time) const {
        auto end_time = std::chrono::high_resolution_clock::now();
        stats_.total_time_ms = std::chrono::duration<double, std::milli>(
            end_time - start_time).count();
    }
};

} // namespace digistar
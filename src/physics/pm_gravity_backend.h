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
#include "gravity_backend.h"
#include "pm_solver.h"

namespace digistar {


/**
 * PM-based gravity backend for long-range forces
 */
template<typename Particle>
class PMGravityBackend : public IGravityBackend<Particle> {
public:
    using typename IGravityBackend<Particle>::Config;

    PMGravityBackend(const Config& config = Config()) {
        this->config_ = config;
        // Create PM solver configuration
        PMSolver::Config pm_config;
        pm_config.grid_size = this->config_.grid_size;
        pm_config.box_size = this->config_.box_size;
        pm_config.G = this->config_.G;
        pm_config.softening = this->config_.softening;

        // Initialize PM solver
        pm_solver_ = std::make_unique<PMSolver>(pm_config);
        pm_solver_->initialize();

        this->stats_.num_particles = 0;
    }

    /**
     * Compute gravitational forces for all particles
     * Updates particle ax, ay fields with gravitational accelerations
     */
    void computeGravity(std::vector<Particle>& particles, float dt) override {
        auto start = startTimer();
        this->stats_.num_particles = particles.size();

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
        this->stats_.deposit_time_ms = 0;
        this->stats_.fft_time_ms = 0;
        this->stats_.interpolate_time_ms = 0;

        endTimer(start);
    }

    /**
     * Get backend name
     */
    std::string getName() const override { return "PMGravityBackend"; }

    /**
     * Get PM solver grid statistics
     */
    PMSolver::GridStats getGridStats() const {
        return pm_solver_->getStats();
    }

private:
    std::unique_ptr<PMSolver> pm_solver_;

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
        this->stats_.total_time_ms = std::chrono::duration<double, std::milli>(
            end_time - start_time).count();
    }
};

} // namespace digistar
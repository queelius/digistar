/**
 * Particle Mesh (PM) Solver for O(N log N) Gravity
 *
 * Uses FFT-based convolution with Green's function for efficient
 * long-range force calculations in toroidal topology.
 */

#pragma once

#include <vector>
#include <complex>
#include <fftw3.h>
#include <cmath>
#include <memory>

namespace digistar {

class PMSolver {
public:
    struct Config {
        int grid_size;           // Grid resolution (power of 2)
        float box_size;         // Physical size of simulation box
        float G;                // Gravitational constant
        float softening;        // Force softening parameter
        bool use_toroidal;      // Enable toroidal topology

        Config() : grid_size(128), box_size(10000.0f), G(100.0f),
                  softening(10.0f), use_toroidal(true) {}
    };

private:
    Config config;

    // Grid dimensions
    int nx, ny;                        // Grid size in x,y
    float dx, dy;                      // Cell size

    // FFTW arrays and plans
    float* density_grid;               // Real-space density
    fftwf_complex* density_fft;        // Fourier-space density
    fftwf_complex* potential_fft;      // Fourier-space potential
    float* potential_grid;             // Real-space potential
    float* force_x_grid;               // X-component of force
    float* force_y_grid;               // Y-component of force

    fftwf_plan plan_forward;           // FFT plan forward
    fftwf_plan plan_backward;          // FFT plan backward

    // Green's function kernel (precomputed)
    std::vector<float> green_function;

    // Memory management
    bool initialized = false;

public:
    PMSolver(const Config& cfg = Config());
    ~PMSolver();

    // Initialize solver (allocate memory, create plans)
    void initialize();
    void cleanup();

    // Main solver interface
    template<typename Particle>
    void computeForces(std::vector<Particle>& particles) {
        if (!initialized) initialize();

        // Step 1: Deposit mass onto grid (Cloud-in-Cell)
        depositMass(particles);

        // Step 2: FFT density to Fourier space
        fftwf_execute(plan_forward);

        // Step 3: Multiply by Green's function to get potential
        multiplyGreenFunction();

        // Step 4: Inverse FFT to get potential in real space
        fftwf_execute(plan_backward);

        // Step 5: Compute forces from potential gradient
        computeForceGradient();

        // Step 6: Interpolate forces back to particles
        interpolateForces(particles);
    }

    // Get grid statistics
    struct GridStats {
        float min_density, max_density;
        float avg_density;
        float min_potential, max_potential;
        int particles_per_cell;
    };
    GridStats getStats() const;

private:
    // Cloud-in-Cell (CIC) mass deposition
    template<typename Particle>
    void depositMass(const std::vector<Particle>& particles) {
        // Clear density grid
        std::fill(density_grid, density_grid + nx * ny, 0.0f);

        for (const auto& p : particles) {
            // Map particle position to grid (with periodic boundaries)
            float px = p.x + config.box_size * 0.5f;
            float py = p.y + config.box_size * 0.5f;

            // Ensure positive modulo
            px = px - std::floor(px / config.box_size) * config.box_size;
            py = py - std::floor(py / config.box_size) * config.box_size;

            float gx = px / dx;
            float gy = py / dy;

            // Get grid cell indices
            int ix = (int)gx;
            int iy = (int)gy;

            // Get fractional position within cell
            float fx = gx - ix;
            float fy = gy - iy;

            // Wrap indices for toroidal topology
            ix = ix % nx;
            iy = iy % ny;
            if (ix < 0) ix += nx;
            if (iy < 0) iy += ny;

            // CIC weights for 4 neighboring cells
            float w00 = (1.0f - fx) * (1.0f - fy);
            float w10 = fx * (1.0f - fy);
            float w01 = (1.0f - fx) * fy;
            float w11 = fx * fy;

            // Deposit mass to neighboring cells (with periodic wrap)
            int ix1 = (ix + 1) % nx;
            int iy1 = (iy + 1) % ny;

            density_grid[iy * nx + ix] += p.mass * w00;
            density_grid[iy * nx + ix1] += p.mass * w10;
            density_grid[iy1 * nx + ix] += p.mass * w01;
            density_grid[iy1 * nx + ix1] += p.mass * w11;
        }

        // Normalize by cell volume
        float cell_volume = dx * dy;
        for (int i = 0; i < nx * ny; i++) {
            density_grid[i] /= cell_volume;
        }
    }

    // Multiply by Green's function in Fourier space
    void multiplyGreenFunction();

    // Compute force from potential gradient
    void computeForceGradient();

    // CIC force interpolation back to particles
    template<typename Particle>
    void interpolateForces(std::vector<Particle>& particles) {
        for (auto& p : particles) {
            // Map particle position to grid
            float px = p.x + config.box_size * 0.5f;
            float py = p.y + config.box_size * 0.5f;

            // Ensure positive modulo
            px = px - std::floor(px / config.box_size) * config.box_size;
            py = py - std::floor(py / config.box_size) * config.box_size;

            float gx = px / dx;
            float gy = py / dy;

            int ix = (int)gx;
            int iy = (int)gy;

            float fx = gx - ix;
            float fy = gy - iy;

            // Wrap indices for toroidal topology
            ix = ix % nx;
            iy = iy % ny;
            if (ix < 0) ix += nx;
            if (iy < 0) iy += ny;

            // CIC weights
            float w00 = (1.0f - fx) * (1.0f - fy);
            float w10 = fx * (1.0f - fy);
            float w01 = (1.0f - fx) * fy;
            float w11 = fx * fy;

            int ix1 = (ix + 1) % nx;
            int iy1 = (iy + 1) % ny;

            // Interpolate forces from grid
            float force_x =
                force_x_grid[iy * nx + ix] * w00 +
                force_x_grid[iy * nx + ix1] * w10 +
                force_x_grid[iy1 * nx + ix] * w01 +
                force_x_grid[iy1 * nx + ix1] * w11;

            float force_y =
                force_y_grid[iy * nx + ix] * w00 +
                force_y_grid[iy * nx + ix1] * w10 +
                force_y_grid[iy1 * nx + ix] * w01 +
                force_y_grid[iy1 * nx + ix1] * w11;

            // Add to particle acceleration
            p.ax += force_x / p.mass;
            p.ay += force_y / p.mass;
        }
    }

    // Precompute Green's function for Poisson equation
    void computeGreenFunction();

    // Helper: wrap coordinate for toroidal topology
    float wrapCoordinate(float x, float box) const {
        x = fmodf(x, box);
        if (x < 0) x += box;
        return x;
    }
};

} // namespace digistar
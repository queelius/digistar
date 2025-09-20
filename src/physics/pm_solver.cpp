/**
 * Particle Mesh (PM) Solver Implementation
 */

#include "pm_solver.h"
#include <iostream>
#include <algorithm>
#include <numeric>

namespace digistar {

PMSolver::PMSolver(const Config& cfg) : config(cfg) {
    nx = ny = config.grid_size;
    dx = dy = config.box_size / config.grid_size;
}

PMSolver::~PMSolver() {
    cleanup();
}

void PMSolver::initialize() {
    if (initialized) return;

    // Allocate memory for grids
    int real_size = nx * ny;
    int complex_size = nx * (ny / 2 + 1);  // FFTW R2C format

    density_grid = (float*)fftwf_malloc(sizeof(float) * real_size);
    potential_grid = (float*)fftwf_malloc(sizeof(float) * real_size);
    force_x_grid = (float*)fftwf_malloc(sizeof(float) * real_size);
    force_y_grid = (float*)fftwf_malloc(sizeof(float) * real_size);

    density_fft = (fftwf_complex*)fftwf_malloc(sizeof(fftwf_complex) * complex_size);
    potential_fft = (fftwf_complex*)fftwf_malloc(sizeof(fftwf_complex) * complex_size);

    // Create FFTW plans
    plan_forward = fftwf_plan_dft_r2c_2d(ny, nx, density_grid, density_fft, FFTW_ESTIMATE);
    plan_backward = fftwf_plan_dft_c2r_2d(ny, nx, potential_fft, potential_grid, FFTW_ESTIMATE);

    // Precompute Green's function
    computeGreenFunction();

    initialized = true;
}

void PMSolver::cleanup() {
    if (!initialized) return;

    fftwf_destroy_plan(plan_forward);
    fftwf_destroy_plan(plan_backward);

    fftwf_free(density_grid);
    fftwf_free(potential_grid);
    fftwf_free(force_x_grid);
    fftwf_free(force_y_grid);
    fftwf_free(density_fft);
    fftwf_free(potential_fft);

    initialized = false;
}

void PMSolver::computeGreenFunction() {
    int complex_size = nx * (ny / 2 + 1);
    green_function.resize(complex_size);

    // Green's function for Poisson equation in Fourier space
    // G(k) = -4πG / |k|^2

    for (int ky = 0; ky < ny; ky++) {
        // Handle negative frequencies for y
        float ky_val = (ky <= ny/2) ? ky : ky - ny;
        float ky_phys = 2.0f * M_PI * ky_val / config.box_size;

        for (int kx = 0; kx < nx/2 + 1; kx++) {
            float kx_phys = 2.0f * M_PI * kx / config.box_size;

            int idx = ky * (nx/2 + 1) + kx;

            float k2 = kx_phys * kx_phys + ky_phys * ky_phys;

            if (k2 > 0) {
                // Add softening to prevent singularity
                float k_soft = std::sqrt(k2 + std::pow(2.0f * M_PI * config.softening / config.box_size, 2));
                green_function[idx] = -4.0f * M_PI * config.G / (k_soft * k_soft);
            } else {
                // DC component (k=0): set to zero (no net force)
                green_function[idx] = 0.0f;
            }
        }
    }
}

void PMSolver::multiplyGreenFunction() {
    int complex_size = nx * (ny / 2 + 1);

    for (int i = 0; i < complex_size; i++) {
        // Multiply density by Green's function to get potential
        float real = density_fft[i][0];
        float imag = density_fft[i][1];

        potential_fft[i][0] = real * green_function[i];
        potential_fft[i][1] = imag * green_function[i];
    }

    // Normalize for FFTW (it doesn't normalize inverse transform)
    float norm = 1.0f / (nx * ny);
    for (int i = 0; i < complex_size; i++) {
        potential_fft[i][0] *= norm;
        potential_fft[i][1] *= norm;
    }
}

void PMSolver::computeForceGradient() {
    // Compute force as negative gradient of potential
    // F = -∇Φ

    for (int iy = 0; iy < ny; iy++) {
        for (int ix = 0; ix < nx; ix++) {
            int idx = iy * nx + ix;

            // Compute gradient using centered differences with periodic boundaries
            int ix_plus = (ix + 1) % nx;
            int ix_minus = (ix - 1 + nx) % nx;
            int iy_plus = (iy + 1) % ny;
            int iy_minus = (iy - 1 + ny) % ny;

            // Force = -gradient of potential
            force_x_grid[idx] = -(potential_grid[iy * nx + ix_plus] -
                                  potential_grid[iy * nx + ix_minus]) / (2.0f * dx);

            force_y_grid[idx] = -(potential_grid[iy_plus * nx + ix] -
                                  potential_grid[iy_minus * nx + ix]) / (2.0f * dy);
        }
    }
}

PMSolver::GridStats PMSolver::getStats() const {
    GridStats stats = {};

    if (!initialized) return stats;

    int size = nx * ny;

    // Density statistics
    stats.min_density = *std::min_element(density_grid, density_grid + size);
    stats.max_density = *std::max_element(density_grid, density_grid + size);
    stats.avg_density = std::accumulate(density_grid, density_grid + size, 0.0f) / size;

    // Potential statistics
    stats.min_potential = *std::min_element(potential_grid, potential_grid + size);
    stats.max_potential = *std::max_element(potential_grid, potential_grid + size);

    return stats;
}

} // namespace digistar
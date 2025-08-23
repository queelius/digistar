#pragma once

#include <vector>
#include <complex>
#include <cmath>
#include <omp.h>
#include "../backend/ISimulationBackend.h"
#include "FastFFT2D.h"

// Particle Mesh (PM) algorithm using our custom FFT
// Achieves O(n) scaling without external dependencies
class ParticleMeshCustom {
private:
    size_t grid_size;
    float box_size;
    float cell_size;
    
    // Grid arrays
    std::vector<float> density;
    std::vector<float> potential;
    std::vector<float2> field;
    
    // FFT arrays
    std::vector<std::complex<float>> density_fft;
    std::vector<std::complex<float>> potential_fft;
    
    // Our custom FFT
    std::unique_ptr<FastFFT2D> fft;
    
    // Precomputed Green's function for Poisson solver
    std::vector<float> greens_function;
    
    // Cloud-In-Cell (CIC) weight function
    void getCICWeights(float x, float y, int& ix, int& iy, float weights[2][2]) {
        // Convert to grid coordinates
        float gx = x / cell_size;
        float gy = y / cell_size;
        
        // Get lower-left cell
        ix = (int)floor(gx);
        iy = (int)floor(gy);
        
        // Fractional part
        float fx = gx - ix;
        float fy = gy - iy;
        
        // CIC weights (bilinear interpolation)
        weights[0][0] = (1.0f - fx) * (1.0f - fy);
        weights[1][0] = fx * (1.0f - fy);
        weights[0][1] = (1.0f - fx) * fy;
        weights[1][1] = fx * fy;
        
        // Periodic boundary conditions
        ix = (ix + grid_size) % grid_size;
        iy = (iy + grid_size) % grid_size;
    }
    
    // Initialize Green's function for Poisson solver
    void initializeGreensFunction(float gravity_constant, float softening) {
        greens_function.resize(grid_size * grid_size);
        
        for (size_t ky = 0; ky < grid_size; ky++) {
            for (size_t kx = 0; kx < grid_size; kx++) {
                // Wave numbers (accounting for Nyquist)
                float kx_val = (kx <= grid_size/2) ? kx : kx - grid_size;
                float ky_val = (ky <= grid_size/2) ? ky : ky - grid_size;
                
                kx_val *= 2.0f * M_PI / box_size;
                ky_val *= 2.0f * M_PI / box_size;
                
                float k2 = kx_val * kx_val + ky_val * ky_val;
                
                if (k2 > 0) {
                    // Green's function: G(k) = -4πG / (k² + softening²)
                    greens_function[ky * grid_size + kx] = 
                        -4.0f * M_PI * gravity_constant / (k2 + softening * softening);
                } else {
                    greens_function[ky * grid_size + kx] = 0;  // DC component
                }
            }
        }
    }
    
public:
    ParticleMeshCustom(size_t grid_size_, float box_size_, float gravity_constant, float softening)
        : grid_size(grid_size_), box_size(box_size_), cell_size(box_size_ / grid_size_) {
        
        // Check power of 2
        if ((grid_size & (grid_size - 1)) != 0) {
            throw std::runtime_error("Grid size must be power of 2 for custom FFT");
        }
        
        // Allocate grids
        size_t total_cells = grid_size * grid_size;
        density.resize(total_cells);
        potential.resize(total_cells);
        field.resize(total_cells);
        
        // FFT arrays
        density_fft.resize(total_cells);
        potential_fft.resize(total_cells);
        
        // Initialize custom FFT
        fft = std::make_unique<FastFFT2D>(grid_size);
        
        // Initialize Green's function
        initializeGreensFunction(gravity_constant, softening);
    }
    
    // Main PM force calculation
    void calculateForces(const std::vector<Particle>& particles,
                        std::vector<float2>& forces,
                        float gravity_constant) {
        
        size_t n = particles.size();
        size_t gs = grid_size;
        
        // Step 1: Clear density grid
        std::fill(density.begin(), density.end(), 0.0f);
        
        // Step 2: Assign particles to mesh using CIC
        for (const auto& p : particles) {
            int ix, iy;
            float weights[2][2];
            getCICWeights(p.pos.x, p.pos.y, ix, iy, weights);
            
            // Distribute mass to 4 nearest grid points
            for (int dy = 0; dy < 2; dy++) {
                for (int dx = 0; dx < 2; dx++) {
                    int gx = (ix + dx) % gs;
                    int gy = (iy + dy) % gs;
                    density[gy * gs + gx] += p.mass * weights[dx][dy] / (cell_size * cell_size);
                }
            }
        }
        
        // Step 3: FFT the density field using our custom FFT
        fft->forward2D_r2c(density.data(), density_fft.data());
        
        // Step 4: Solve Poisson equation in Fourier space
        // Φ(k) = G(k) * ρ(k)
        #pragma omp parallel for
        for (size_t i = 0; i < density_fft.size(); i++) {
            potential_fft[i] = density_fft[i] * greens_function[i];
        }
        
        // Step 5: Inverse FFT to get potential
        fft->inverse2D_c2r(potential_fft.data(), potential.data());
        
        // Step 6: Calculate force field from potential gradient
        // F = -∇Φ using finite differences
        #pragma omp parallel for
        for (size_t y = 0; y < gs; y++) {
            for (size_t x = 0; x < gs; x++) {
                // Central differences with periodic boundaries
                size_t xp = (x + 1) % gs;
                size_t xm = (x + gs - 1) % gs;
                size_t yp = (y + 1) % gs;
                size_t ym = (y + gs - 1) % gs;
                
                float fx = -(potential[y * gs + xp] - potential[y * gs + xm]) / (2.0f * cell_size);
                float fy = -(potential[yp * gs + x] - potential[ym * gs + x]) / (2.0f * cell_size);
                
                field[y * gs + x] = {fx, fy};
            }
        }
        
        // Step 7: Interpolate forces back to particles
        #pragma omp parallel for
        for (size_t i = 0; i < n; i++) {
            int ix, iy;
            float weights[2][2];
            getCICWeights(particles[i].pos.x, particles[i].pos.y, ix, iy, weights);
            
            float fx = 0, fy = 0;
            
            // Gather force from 4 nearest grid points
            for (int dy = 0; dy < 2; dy++) {
                for (int dx = 0; dx < 2; dx++) {
                    int gx = (ix + dx) % gs;
                    int gy = (iy + dy) % gs;
                    float w = weights[dx][dy];
                    fx += field[gy * gs + gx].x * w;
                    fy += field[gy * gs + gx].y * w;
                }
            }
            
            // Apply force (F = ma, so a = F/m, but we store F*m for integration)
            forces[i].x = fx * particles[i].mass;
            forces[i].y = fy * particles[i].mass;
        }
    }
    
    // Get memory usage
    size_t getMemoryUsage() const {
        size_t gs2 = grid_size * grid_size;
        return gs2 * sizeof(float) * 3 +           // density, potential, greens
               gs2 * sizeof(float2) +               // field
               gs2 * sizeof(std::complex<float>) * 2;  // FFT arrays
    }
    
    // Get the density field (for visualization/debugging)
    const std::vector<float>& getDensityField() const { return density; }
    const std::vector<float>& getPotentialField() const { return potential; }
    const std::vector<float2>& getForceField() const { return field; }
};
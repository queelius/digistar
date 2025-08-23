#pragma once

#include <vector>
#include <complex>
#include <cmath>
#include <omp.h>
#include "../backend/ISimulationBackend.h"

// Particle Mesh (PM) algorithm for O(n) gravity calculation
// Uses FFT to solve Poisson's equation: ∇²Φ = 4πGρ
// Steps:
// 1. Assign particles to mesh (CIC - Cloud In Cell)
// 2. FFT the density field
// 3. Solve Poisson equation in Fourier space
// 4. Inverse FFT to get potential
// 5. Calculate forces from potential gradient
// 6. Interpolate forces back to particles

class ParticleMesh {
private:
    size_t grid_size;
    float box_size;
    float cell_size;
    
    // Grid arrays
    std::vector<float> density;     // Density field ρ
    std::vector<float> potential;   // Gravitational potential Φ
    std::vector<float2> field;      // Force field -∇Φ
    
    // FFT arrays (using real-to-complex transform)
    std::vector<std::complex<float>> density_fft;
    std::vector<std::complex<float>> potential_fft;
    
    // Precomputed Green's function for Poisson solver
    std::vector<float> greens_function;
    
    // Cloud-In-Cell (CIC) weight function
    // Returns weights for the 4 nearest grid points
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
    
    // Simple DFT implementation (for portability - replace with FFTW for production)
    void fft2d(std::vector<std::complex<float>>& data, bool inverse = false) {
        size_t n = grid_size;
        float norm = inverse ? 1.0f / (n * n) : 1.0f;
        std::vector<std::complex<float>> temp(n * n);
        
        // 1D FFT along rows
        for (size_t y = 0; y < n; y++) {
            for (size_t kx = 0; kx < n; kx++) {
                std::complex<float> sum(0, 0);
                for (size_t x = 0; x < n; x++) {
                    float angle = 2.0f * M_PI * kx * x / n * (inverse ? 1 : -1);
                    std::complex<float> w(cos(angle), sin(angle));
                    sum += data[y * n + x] * w;
                }
                temp[y * n + kx] = sum * norm;
            }
        }
        
        // 1D FFT along columns
        for (size_t x = 0; x < n; x++) {
            for (size_t ky = 0; ky < n; ky++) {
                std::complex<float> sum(0, 0);
                for (size_t y = 0; y < n; y++) {
                    float angle = 2.0f * M_PI * ky * y / n * (inverse ? 1 : -1);
                    std::complex<float> w(cos(angle), sin(angle));
                    sum += temp[y * n + x] * w;
                }
                data[ky * n + x] = sum;
            }
        }
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
    ParticleMesh(size_t grid_size_, float box_size_, float gravity_constant, float softening)
        : grid_size(grid_size_), box_size(box_size_), cell_size(box_size_ / grid_size_) {
        
        // Allocate grids
        size_t total_cells = grid_size * grid_size;
        density.resize(total_cells);
        potential.resize(total_cells);
        field.resize(total_cells);
        
        // FFT arrays
        density_fft.resize(total_cells);
        potential_fft.resize(total_cells);
        
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
        
        // Step 3: FFT the density field
        for (size_t i = 0; i < density.size(); i++) {
            density_fft[i] = std::complex<float>(density[i], 0);
        }
        fft2d(density_fft);
        
        // Step 4: Solve Poisson equation in Fourier space
        // Φ(k) = G(k) * ρ(k)
        for (size_t i = 0; i < density_fft.size(); i++) {
            potential_fft[i] = density_fft[i] * greens_function[i];
        }
        
        // Step 5: Inverse FFT to get potential
        fft2d(potential_fft, true);
        for (size_t i = 0; i < potential.size(); i++) {
            potential[i] = potential_fft[i].real();
        }
        
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
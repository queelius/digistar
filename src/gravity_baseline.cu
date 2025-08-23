// gravity_baseline.cu - Minimal gravity-only particle simulation
// Goal: Maximum particles with just gravity to establish performance baseline

#include <cuda_runtime.h>
#include <cufft.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <iostream>
#include <chrono>

// Simulation parameters
constexpr int GRID_SIZE = 512;  // PM grid resolution (512x512 for 2D)
constexpr float BOX_SIZE = 1000.0f;
constexpr float G = 1.0f;  // Gravitational constant
constexpr float SOFTENING = 0.1f;  // Prevent singularities
constexpr float DT = 0.016f;  // 60 FPS

// Particle structure (minimal for maximum count)
struct Particle {
    float2 pos;
    float2 vel;
    float mass;
    float radius;  // For visualization only
};

// PM Grid for O(n) gravity calculation
class PMSolver2D {
private:
    float2* d_density_grid;
    float2* d_potential_grid;
    cufftHandle plan_forward;
    cufftHandle plan_inverse;
    int grid_size;
    float box_size;
    
public:
    PMSolver2D(int grid_size, float box_size) 
        : grid_size(grid_size), box_size(box_size) {
        
        size_t grid_bytes = grid_size * grid_size * sizeof(float2);
        cudaMalloc(&d_density_grid, grid_bytes);
        cudaMalloc(&d_potential_grid, grid_bytes);
        
        // Create FFT plans for 2D
        cufftPlan2d(&plan_forward, grid_size, grid_size, CUFFT_R2C);
        cufftPlan2d(&plan_inverse, grid_size, grid_size, CUFFT_C2R);
    }
    
    ~PMSolver2D() {
        cudaFree(d_density_grid);
        cudaFree(d_potential_grid);
        cufftDestroy(plan_forward);
        cufftDestroy(plan_inverse);
    }
    
    void compute_forces(Particle* particles, int n, float2* forces);
};

// Kernels
__global__ void clear_grid(float* grid, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        grid[idx] = 0.0f;
    }
}

__global__ void project_to_grid(Particle* particles, int n, float* density_grid, 
                                int grid_size, float box_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    
    Particle p = particles[idx];
    
    // Map particle position to grid
    float2 grid_pos = p.pos * (grid_size / box_size);
    int gx = (int)grid_pos.x;
    int gy = (int)grid_pos.y;
    
    // Wrap around for periodic boundaries
    gx = (gx + grid_size) % grid_size;
    gy = (gy + grid_size) % grid_size;
    
    // Cloud-in-cell (CIC) interpolation for smoother density
    float fx = grid_pos.x - gx;
    float fy = grid_pos.y - gy;
    
    int gxp = (gx + 1) % grid_size;
    int gyp = (gy + 1) % grid_size;
    
    // Distribute mass to 4 neighboring cells
    atomicAdd(&density_grid[gy * grid_size + gx], p.mass * (1-fx) * (1-fy));
    atomicAdd(&density_grid[gy * grid_size + gxp], p.mass * fx * (1-fy));
    atomicAdd(&density_grid[gyp * grid_size + gx], p.mass * (1-fx) * fy);
    atomicAdd(&density_grid[gyp * grid_size + gxp], p.mass * fx * fy);
}

__global__ void solve_poisson_fourier(cufftComplex* density_fft, 
                                      cufftComplex* potential_fft,
                                      int grid_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = (grid_size/2 + 1) * grid_size;
    if (idx >= total) return;
    
    // Get k-space coordinates
    int kx = idx % (grid_size/2 + 1);
    int ky = idx / (grid_size/2 + 1);
    
    // Adjust for negative frequencies
    if (ky > grid_size/2) ky -= grid_size;
    
    // Compute k^2 (avoid division by zero)
    float k2 = kx * kx + ky * ky;
    if (k2 == 0) {
        potential_fft[idx].x = 0;
        potential_fft[idx].y = 0;
        return;
    }
    
    // Poisson equation in Fourier space: potential = -4πG * density / k^2
    float factor = -4.0f * M_PI * G / k2;
    potential_fft[idx].x = density_fft[idx].x * factor;
    potential_fft[idx].y = density_fft[idx].y * factor;
}

__global__ void compute_forces_from_potential(float* potential_grid, 
                                              float2* force_grid,
                                              int grid_size, float box_size) {
    int gx = blockIdx.x * blockDim.x + threadIdx.x;
    int gy = blockIdx.y * blockDim.y + threadIdx.y;
    if (gx >= grid_size || gy >= grid_size) return;
    
    int idx = gy * grid_size + gx;
    
    // Compute force as negative gradient of potential
    int gxm = (gx - 1 + grid_size) % grid_size;
    int gxp = (gx + 1) % grid_size;
    int gym = (gy - 1 + grid_size) % grid_size;
    int gyp = (gy + 1) % grid_size;
    
    float dx = box_size / grid_size;
    
    // Central differencing for gradient
    force_grid[idx].x = -(potential_grid[gy * grid_size + gxp] - 
                          potential_grid[gy * grid_size + gxm]) / (2.0f * dx);
    force_grid[idx].y = -(potential_grid[gyp * grid_size + gx] - 
                          potential_grid[gym * grid_size + gx]) / (2.0f * dx);
}

__global__ void interpolate_forces(Particle* particles, int n, 
                                   float2* force_grid, float2* forces,
                                   int grid_size, float box_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    
    Particle p = particles[idx];
    
    // Map particle position to grid
    float2 grid_pos = p.pos * (grid_size / box_size);
    int gx = (int)grid_pos.x;
    int gy = (int)grid_pos.y;
    
    gx = (gx + grid_size) % grid_size;
    gy = (gy + grid_size) % grid_size;
    
    // Bilinear interpolation
    float fx = grid_pos.x - gx;
    float fy = grid_pos.y - gy;
    
    int gxp = (gx + 1) % grid_size;
    int gyp = (gy + 1) % grid_size;
    
    float2 f00 = force_grid[gy * grid_size + gx];
    float2 f10 = force_grid[gy * grid_size + gxp];
    float2 f01 = force_grid[gyp * grid_size + gx];
    float2 f11 = force_grid[gyp * grid_size + gxp];
    
    forces[idx].x = f00.x * (1-fx) * (1-fy) + f10.x * fx * (1-fy) +
                    f01.x * (1-fx) * fy + f11.x * fx * fy;
    forces[idx].y = f00.y * (1-fx) * (1-fy) + f10.y * fx * (1-fy) +
                    f01.y * (1-fx) * fy + f11.y * fx * fy;
    
    // Apply force based on particle mass
    forces[idx].x *= p.mass;
    forces[idx].y *= p.mass;
}

__global__ void integrate_particles(Particle* particles, float2* forces, 
                                   int n, float dt) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    
    Particle& p = particles[idx];
    
    // Simple Euler integration (can upgrade to Verlet later)
    float2 acceleration = forces[idx];
    if (p.mass > 0) {
        acceleration.x /= p.mass;
        acceleration.y /= p.mass;
    }
    
    p.vel.x += acceleration.x * dt;
    p.vel.y += acceleration.y * dt;
    p.pos.x += p.vel.x * dt;
    p.pos.y += p.vel.y * dt;
    
    // Wrap around boundaries
    if (p.pos.x < 0) p.pos.x += BOX_SIZE;
    if (p.pos.x >= BOX_SIZE) p.pos.x -= BOX_SIZE;
    if (p.pos.y < 0) p.pos.y += BOX_SIZE;
    if (p.pos.y >= BOX_SIZE) p.pos.y -= BOX_SIZE;
}

// PM Solver implementation
void PMSolver2D::compute_forces(Particle* particles, int n, float2* forces) {
    int grid_total = grid_size * grid_size;
    
    // Clear density grid
    clear_grid<<<(grid_total + 255)/256, 256>>>((float*)d_density_grid, grid_total * 2);
    
    // Project particles to density grid
    project_to_grid<<<(n + 255)/256, 256>>>(particles, n, (float*)d_density_grid, 
                                            grid_size, box_size);
    
    // FFT density to frequency space
    cufftExecR2C(plan_forward, (float*)d_density_grid, (cufftComplex*)d_density_grid);
    
    // Solve Poisson equation in Fourier space
    int freq_size = (grid_size/2 + 1) * grid_size;
    solve_poisson_fourier<<<(freq_size + 255)/256, 256>>>((cufftComplex*)d_density_grid,
                                                          (cufftComplex*)d_potential_grid,
                                                          grid_size);
    
    // Inverse FFT to get potential
    cufftExecC2R(plan_inverse, (cufftComplex*)d_potential_grid, (float*)d_potential_grid);
    
    // Normalize (cuFFT doesn't normalize inverse transform)
    thrust::transform(thrust::device_ptr<float>((float*)d_potential_grid),
                     thrust::device_ptr<float>((float*)d_potential_grid + grid_total),
                     thrust::device_ptr<float>((float*)d_potential_grid),
                     thrust::placeholders::_1 / (float)(grid_total));
    
    // Compute forces from potential gradient
    dim3 block(16, 16);
    dim3 grid((grid_size + 15)/16, (grid_size + 15)/16);
    compute_forces_from_potential<<<grid, block>>>((float*)d_potential_grid,
                                                   d_density_grid,  // Reuse for forces
                                                   grid_size, box_size);
    
    // Interpolate forces back to particles
    interpolate_forces<<<(n + 255)/256, 256>>>(particles, n, d_density_grid, forces,
                                               grid_size, box_size);
}

// Simple benchmark
void benchmark_gravity(int num_particles) {
    std::cout << "\n=== Gravity-Only Baseline Benchmark ===" << std::endl;
    std::cout << "Particles: " << num_particles << std::endl;
    std::cout << "PM Grid: " << GRID_SIZE << "x" << GRID_SIZE << std::endl;
    
    // Allocate particles
    thrust::device_vector<Particle> d_particles(num_particles);
    thrust::device_vector<float2> d_forces(num_particles);
    
    // Initialize particles (random distribution)
    auto init_particles = [](Particle& p) {
        p.pos.x = BOX_SIZE * (rand() / (float)RAND_MAX);
        p.pos.y = BOX_SIZE * (rand() / (float)RAND_MAX);
        p.vel.x = 10.0f * (rand() / (float)RAND_MAX - 0.5f);
        p.vel.y = 10.0f * (rand() / (float)RAND_MAX - 0.5f);
        p.mass = 1.0f;
        p.radius = 0.5f;
    };
    
    thrust::host_vector<Particle> h_particles(num_particles);
    for (auto& p : h_particles) init_particles(p);
    d_particles = h_particles;
    
    // Create PM solver
    PMSolver2D pm_solver(GRID_SIZE, BOX_SIZE);
    
    // Warmup
    for (int i = 0; i < 10; i++) {
        pm_solver.compute_forces(thrust::raw_pointer_cast(d_particles.data()),
                                num_particles,
                                thrust::raw_pointer_cast(d_forces.data()));
    }
    cudaDeviceSynchronize();
    
    // Benchmark
    auto start = std::chrono::high_resolution_clock::now();
    int num_steps = 100;
    
    for (int step = 0; step < num_steps; step++) {
        // Compute forces
        pm_solver.compute_forces(thrust::raw_pointer_cast(d_particles.data()),
                                num_particles,
                                thrust::raw_pointer_cast(d_forces.data()));
        
        // Integrate
        integrate_particles<<<(num_particles + 255)/256, 256>>>(
            thrust::raw_pointer_cast(d_particles.data()),
            thrust::raw_pointer_cast(d_forces.data()),
            num_particles, DT);
    }
    
    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();
    
    // Report performance
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    float ms_per_step = duration.count() / (float)num_steps;
    float fps = 1000.0f / ms_per_step;
    
    std::cout << "Time per step: " << ms_per_step << " ms" << std::endl;
    std::cout << "Effective FPS: " << fps << std::endl;
    std::cout << "Particles per second: " << (num_particles * fps) / 1e6 << " million" << std::endl;
    
    // Memory usage
    size_t free_mem, total_mem;
    cudaMemGetInfo(&free_mem, &total_mem);
    std::cout << "GPU Memory used: " << (total_mem - free_mem) / (1024*1024) << " MB" << std::endl;
}

int main() {
    // Test various particle counts
    std::vector<int> test_counts = {
        100000,    // 100K
        1000000,   // 1M
        5000000,   // 5M
        10000000,  // 10M
        20000000,  // 20M
        50000000,  // 50M
        100000000  // 100M - ambitious!
    };
    
    for (int count : test_counts) {
        try {
            benchmark_gravity(count);
        } catch (std::exception& e) {
            std::cout << "Failed at " << count << " particles: " << e.what() << std::endl;
            break;
        }
    }
    
    return 0;
}
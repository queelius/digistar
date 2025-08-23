#include "CUDABackend.h"
#include <cuda_runtime.h>
#include <cufft.h>
#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <iostream>
#include <stdexcept>

// CUDA kernels (from gravity_baseline.cu)
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
    
    float2 grid_pos;
    grid_pos.x = p.pos.x * (grid_size / box_size);
    grid_pos.y = p.pos.y * (grid_size / box_size);
    
    int gx = (int)grid_pos.x;
    int gy = (int)grid_pos.y;
    
    gx = (gx + grid_size) % grid_size;
    gy = (gy + grid_size) % grid_size;
    
    float fx = grid_pos.x - gx;
    float fy = grid_pos.y - gy;
    
    int gxp = (gx + 1) % grid_size;
    int gyp = (gy + 1) % grid_size;
    
    atomicAdd(&density_grid[gy * grid_size + gx], p.mass * (1-fx) * (1-fy));
    atomicAdd(&density_grid[gy * grid_size + gxp], p.mass * fx * (1-fy));
    atomicAdd(&density_grid[gyp * grid_size + gx], p.mass * (1-fx) * fy);
    atomicAdd(&density_grid[gyp * grid_size + gxp], p.mass * fx * fy);
}

__global__ void solve_poisson_fourier(cufftComplex* density_fft, 
                                      cufftComplex* potential_fft,
                                      int grid_size, float G) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = (grid_size/2 + 1) * grid_size;
    if (idx >= total) return;
    
    int kx = idx % (grid_size/2 + 1);
    int ky = idx / (grid_size/2 + 1);
    
    if (ky > grid_size/2) ky -= grid_size;
    
    float k2 = kx * kx + ky * ky;
    if (k2 == 0) {
        potential_fft[idx].x = 0;
        potential_fft[idx].y = 0;
        return;
    }
    
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
    
    int gxm = (gx - 1 + grid_size) % grid_size;
    int gxp = (gx + 1) % grid_size;
    int gym = (gy - 1 + grid_size) % grid_size;
    int gyp = (gy + 1) % grid_size;
    
    float dx = box_size / grid_size;
    
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
    
    float2 grid_pos;
    grid_pos.x = p.pos.x * (grid_size / box_size);
    grid_pos.y = p.pos.y * (grid_size / box_size);
    
    int gx = (int)grid_pos.x;
    int gy = (int)grid_pos.y;
    
    gx = (gx + grid_size) % grid_size;
    gy = (gy + grid_size) % grid_size;
    
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
    
    forces[idx].x *= p.mass;
    forces[idx].y *= p.mass;
}

__global__ void integrate_particles(Particle* particles, float2* forces, 
                                   int n, float dt, float box_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    
    Particle& p = particles[idx];
    
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
    if (p.pos.x < 0) p.pos.x += box_size;
    if (p.pos.x >= box_size) p.pos.x -= box_size;
    if (p.pos.y < 0) p.pos.y += box_size;
    if (p.pos.y >= box_size) p.pos.y -= box_size;
}

// CUDABackend implementation
class CUDABackendImpl {
private:
    // Device pointers
    Particle* d_particles;
    float2* d_forces;
    float2* d_density_grid;
    float2* d_potential_grid;
    float2* d_force_grid;
    
    // FFT plans
    cufftHandle plan_forward;
    cufftHandle plan_inverse;
    
    // Parameters
    size_t num_particles;
    SimulationParams params;
    
    // Memory tracking
    size_t allocated_memory;
    
public:
    CUDABackend() : d_particles(nullptr), d_forces(nullptr), 
                    d_density_grid(nullptr), d_potential_grid(nullptr),
                    d_force_grid(nullptr), num_particles(0), allocated_memory(0) {
        // Check CUDA availability
        int device_count;
        cudaGetDeviceCount(&device_count);
        if (device_count == 0) {
            throw std::runtime_error("No CUDA devices found");
        }
        
        // Set device
        cudaSetDevice(0);
    }
    
    ~CUDABackend() {
        cleanup();
    }
    
    void initialize(size_t n, const SimulationParams& p) override {
        cleanup();  // Clean up any existing allocation
        
        num_particles = n;
        params = p;
        
        // Allocate particle data
        size_t particle_bytes = n * sizeof(Particle);
        size_t force_bytes = n * sizeof(float2);
        cudaMalloc(&d_particles, particle_bytes);
        cudaMalloc(&d_forces, force_bytes);
        
        // Allocate grid data
        int grid_total = params.grid_size * params.grid_size;
        size_t grid_bytes = grid_total * sizeof(float2);
        cudaMalloc(&d_density_grid, grid_bytes);
        cudaMalloc(&d_potential_grid, grid_bytes);
        cudaMalloc(&d_force_grid, grid_bytes);
        
        // Create FFT plans
        cufftPlan2d(&plan_forward, params.grid_size, params.grid_size, CUFFT_R2C);
        cufftPlan2d(&plan_inverse, params.grid_size, params.grid_size, CUFFT_C2R);
        
        allocated_memory = particle_bytes + force_bytes + 3 * grid_bytes;
    }
    
    void setParticles(const std::vector<Particle>& particles) override {
        if (particles.size() != num_particles) {
            throw std::runtime_error("Particle count mismatch");
        }
        
        cudaMemcpy(d_particles, particles.data(), 
                   num_particles * sizeof(Particle), 
                   cudaMemcpyHostToDevice);
    }
    
    void getParticles(std::vector<Particle>& particles) override {
        particles.resize(num_particles);
        cudaMemcpy(particles.data(), d_particles, 
                   num_particles * sizeof(Particle), 
                   cudaMemcpyDeviceToHost);
    }
    
    void computeForces() override {
        int grid_total = params.grid_size * params.grid_size;
        
        // Clear density grid
        clear_grid<<<(grid_total * 2 + 255)/256, 256>>>((float*)d_density_grid, grid_total * 2);
        
        // Project particles to density grid
        project_to_grid<<<(num_particles + 255)/256, 256>>>(
            d_particles, num_particles, (float*)d_density_grid, 
            params.grid_size, params.box_size);
        
        // FFT density to frequency space
        cufftExecR2C(plan_forward, (float*)d_density_grid, (cufftComplex*)d_density_grid);
        
        // Solve Poisson equation
        int freq_size = (params.grid_size/2 + 1) * params.grid_size;
        solve_poisson_fourier<<<(freq_size + 255)/256, 256>>>(
            (cufftComplex*)d_density_grid,
            (cufftComplex*)d_potential_grid,
            params.grid_size, params.gravity_constant);
        
        // Inverse FFT to get potential
        cufftExecC2R(plan_inverse, (cufftComplex*)d_potential_grid, (float*)d_potential_grid);
        
        // Normalize FFT result
        float normalization = 1.0f / (grid_total);
        thrust::device_ptr<float> potential_ptr((float*)d_potential_grid);
        thrust::transform(potential_ptr, potential_ptr + grid_total,
                         potential_ptr,
                         [normalization] __device__ (float x) { return x * normalization; });
        
        // Compute forces from potential gradient
        dim3 block(16, 16);
        dim3 grid((params.grid_size + 15)/16, (params.grid_size + 15)/16);
        compute_forces_from_potential<<<grid, block>>>(
            (float*)d_potential_grid, d_force_grid,
            params.grid_size, params.box_size);
        
        // Interpolate forces back to particles
        interpolate_forces<<<(num_particles + 255)/256, 256>>>(
            d_particles, num_particles, d_force_grid, d_forces,
            params.grid_size, params.box_size);
    }
    
    void integrate(float dt) override {
        integrate_particles<<<(num_particles + 255)/256, 256>>>(
            d_particles, d_forces, num_particles, dt, params.box_size);
    }
    
    size_t getMaxParticles() const override {
        // Estimate based on available GPU memory
        size_t free_mem, total_mem;
        cudaMemGetInfo(&free_mem, &total_mem);
        
        // Assume we can use 80% of free memory
        size_t usable = free_mem * 0.8;
        
        // Each particle needs: particle struct + force + grid contribution
        size_t per_particle = sizeof(Particle) + sizeof(float2) + 100; // rough estimate
        
        return usable / per_particle;
    }
    
    std::string getBackendName() const override {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, 0);
        return std::string("CUDA (") + prop.name + ")";
    }
    
    bool isGPU() const override { return true; }
    
    size_t getMemoryUsage() const override { return allocated_memory; }
    
    void cleanup() override {
        if (d_particles) {
            cudaFree(d_particles);
            cudaFree(d_forces);
            cudaFree(d_density_grid);
            cudaFree(d_potential_grid);
            cudaFree(d_force_grid);
            
            cufftDestroy(plan_forward);
            cufftDestroy(plan_inverse);
            
            d_particles = nullptr;
            allocated_memory = 0;
        }
    }
};

// Wrapper class that implements the interface
CUDABackend::CUDABackend() : pImpl(std::make_unique<CUDABackendImpl>()) {}
CUDABackend::~CUDABackend() = default;

void CUDABackend::initialize(size_t num_particles, const SimulationParams& params) {
    pImpl->initialize(num_particles, params);
}

void CUDABackend::setParticles(const std::vector<Particle>& particles) {
    pImpl->setParticles(particles);
}

void CUDABackend::getParticles(std::vector<Particle>& particles) {
    pImpl->getParticles(particles);
}

void CUDABackend::computeForces() {
    pImpl->computeForces();
}

void CUDABackend::integrate(float dt) {
    pImpl->integrate(dt);
}

size_t CUDABackend::getMaxParticles() const {
    return pImpl->getMaxParticles();
}

std::string CUDABackend::getBackendName() const {
    return pImpl->getBackendName();
}

bool CUDABackend::isGPU() const {
    return pImpl->isGPU();
}

size_t CUDABackend::getMemoryUsage() const {
    return pImpl->getMemoryUsage();
}

void CUDABackend::cleanup() {
    pImpl->cleanup();
}
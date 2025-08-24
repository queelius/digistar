# GPU/CPU Optimization Strategy

## Overview

DigiStar targets 10+ million particles at 60 FPS. This requires GPU acceleration for physics, but CPU handles game logic, AI, and special cases. This document unifies our optimization strategy across both platforms.

## Force Calculation Methods

### Algorithm Complexity Hierarchy

| Method | Complexity | Particles | Use Case |
|--------|------------|-----------|----------|
| Direct N-body | O(N²) | < 10K | Perfect accuracy, debugging |
| Barnes-Hut | O(N log N) | 100K-1M | Non-periodic boundaries |
| Particle-Mesh (PM) | O(N) | 10M+ | **Our primary method** |
| TreePM Hybrid | O(N) | 10M+ | Short+long range accuracy |
| Fast Multipole | O(N) | 10M+ | Complex, high memory |

### Why PM (Particle-Mesh) Wins

**Advantages:**
- O(N) scaling to millions of particles
- FFT naturally handles toroidal topology
- Very GPU-friendly (cuFFT optimized)
- Smooth long-range forces
- No tree rebuilding overhead

**Limitations:**
- Fixed grid resolution
- Loses short-range detail
- Requires separate collision detection

## GPU Implementation

### Core Architecture

```cuda
class GPUSimulation {
    // Three-tier spatial indexing
    SpatialHash contact_hash;      // 2r cells for collisions
    NeighborList local_neighbors;  // 100r for springs
    PMGrid gravity_grid;           // Global gravity via FFT
    SpringNetwork springs;         // Dynamic spring system
};
```

### Unified Force Kernel

```cuda
__global__ void compute_all_forces(
    Particle* particles,
    SpatialHash* contact_hash,
    PMGrid* pm_grid,
    SpringNetwork* springs,
    float3* forces_out
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float3 total_force = make_float3(0);
    
    // 1. Contact forces (soft repulsion)
    total_force += compute_contact_forces(idx, contact_hash);
    
    // 2. Spring forces (local bonds)
    total_force += compute_spring_forces(idx, springs);
    
    // 3. Global gravity (PM interpolation)
    total_force += interpolate_pm_field(particles[idx].pos, pm_grid);
    
    forces_out[idx] = total_force;
}
```

### GPU-Specific Optimizations

#### Memory Coalescing (SoA Layout)
```cuda
struct ParticleDataGPU {
    float* pos_x;  // Contiguous for coalesced access
    float* pos_y;
    float* vel_x;
    float* vel_y;
    float* mass;
    float* temp;
};
```

#### Morton Ordering for Cache Efficiency
```cuda
__global__ void morton_sort_particles(Particle* particles, uint32_t* morton_codes) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    morton_codes[idx] = morton2D(particles[idx].pos);
}
// Then sort particles by morton code for spatial locality
```

#### Warp-Level Primitives
```cuda
__device__ void warp_reduce_force(float3& force) {
    // Sum across warp using shuffles (no shared memory needed)
    force.x += __shfl_down_sync(0xffffffff, force.x, 16);
    force.x += __shfl_down_sync(0xffffffff, force.x, 8);
    force.x += __shfl_down_sync(0xffffffff, force.x, 4);
    force.x += __shfl_down_sync(0xffffffff, force.x, 2);
    force.x += __shfl_down_sync(0xffffffff, force.x, 1);
}
```

#### Mixed Precision
```cuda
// Positions need FP32, forces can use FP16
__global__ void force_calculation_mixed() {
    float2 pos = particles.get_position_fp32(idx);
    
    // Forces in half precision for 2x throughput
    __half2 force = compute_force_fp16(pos);
    
    // Accumulate in FP32 for accuracy
    forces[idx] += __half22float2(force);
}
```

#### Concurrent Streams
```cuda
void build_structures_async() {
    // Three structures built in parallel
    cudaStream_t streams[3];
    
    build_spatial_hash<<<..., streams[0]>>>(particles);
    build_neighbor_lists<<<..., streams[1]>>>(particles);
    project_to_pm_grid<<<..., streams[2]>>>(particles);
    
    cudaDeviceSynchronize();
}
```

## CPU Implementation

### SIMD/AVX Optimizations

```cpp
// Process 8 particles at once with AVX2
void update_particles_avx2(ParticlesSoA& particles, float dt) {
    for (int i = 0; i < n_particles; i += 8) {
        __m256 pos_x = _mm256_load_ps(&particles.pos_x[i]);
        __m256 vel_x = _mm256_load_ps(&particles.vel_x[i]);
        __m256 force_x = _mm256_load_ps(&forces[i]);
        
        // Update velocity: v += F/m * dt
        __m256 mass = _mm256_load_ps(&particles.mass[i]);
        __m256 accel = _mm256_div_ps(force_x, mass);
        vel_x = _mm256_fmadd_ps(accel, _mm256_set1_ps(dt), vel_x);
        
        // Update position: x += v * dt
        pos_x = _mm256_fmadd_ps(vel_x, _mm256_set1_ps(dt), pos_x);
        
        _mm256_store_ps(&particles.pos_x[i], pos_x);
        _mm256_store_ps(&particles.vel_x[i], vel_x);
    }
}
```

### OpenMP Parallelization

```cpp
void compute_forces_cpu() {
    #pragma omp parallel for schedule(dynamic, 1024)
    for (int i = 0; i < n_particles; i++) {
        // Each thread handles chunk of particles
        compute_particle_forces(i);
    }
}
```

### Cache-Optimized Data Layout

```cpp
struct alignas(64) ParticlesSoA {  // Cache line aligned
    float* pos_x;  
    float* pos_y;
    float* vel_x;
    float* vel_y;
    float* mass;
    
    // Prefetch next cache line
    void prefetch(int idx) {
        _mm_prefetch(&pos_x[idx + 16], _MM_HINT_T0);
        _mm_prefetch(&vel_x[idx + 16], _MM_HINT_T0);
    }
};
```

## Hybrid CPU+GPU Strategy

### Work Division

```cpp
class HybridBackend {
    // GPU: Bulk physics (10M particles)
    CUDABackend gpu_sim;
    
    // CPU: Special objects, AI, game logic
    CPUBackend cpu_sim;
    
    void update(float dt) {
        // Start GPU work asynchronously
        gpu_sim.compute_forces_async();
        
        // CPU work in parallel
        #pragma omp parallel sections
        {
            #pragma omp section
            cpu_sim.update_ai_agents();
            
            #pragma omp section
            cpu_sim.handle_player_input();
            
            #pragma omp section
            cpu_sim.update_special_objects();
        }
        
        // Sync and exchange boundary data
        gpu_sim.wait();
        exchange_forces_via_shm();
    }
};
```

### Platform Selection

```cpp
std::unique_ptr<Backend> create_backend() {
    if (cuda_available() && gpu_memory >= 6_GB) {
        return std::make_unique<CUDABackend>();
    } else if (cpu_has_avx512()) {
        return std::make_unique<AVX512Backend>();
    } else if (cpu_has_avx2()) {
        return std::make_unique<AVX2Backend>();
    } else {
        return std::make_unique<ScalarBackend>();
    }
}
```

## Performance Targets

### GPU Performance

| GPU | Particles | FPS | Method | Memory |
|-----|-----------|-----|--------|---------|
| RTX 3060 | 10M | 60 | PM+TreePM | 6 GB |
| RTX 3060 | 20M | 30 | PM only | 8 GB |
| RTX 4090 | 50M | 60 | PM+TreePM | 12 GB |
| RTX 4090 | 100M | 30 | PM only | 20 GB |

### CPU Performance

| CPU | Particles | FPS | Method | Cores |
|-----|-----------|-----|--------|-------|
| Ryzen 9 5950X | 100K | 60 | Direct | 16 |
| Ryzen 9 5950X | 500K | 30 | Barnes-Hut | 16 |
| Intel i9-12900K | 200K | 60 | Direct+AVX512 | 24 |
| Apple M2 Max | 1M | 30 | Metal Compute | 12 |

### Memory Requirements (20M particles)

```
Particles:      20M × 32 bytes = 640 MB
Spatial Hash:   10M cells × 16 bytes = 160 MB
Neighbor Lists: 20M × 50 × 4 bytes = 4 GB
PM Grid:        512³ × 8 bytes = 1 GB
Springs:        5M × 24 bytes = 120 MB
Forces:         20M × 12 bytes = 240 MB
----------------------------------------
Total: ~6.2 GB GPU memory required
```

### Bandwidth Analysis

```
Per frame at 60 FPS:
- Read particles: 640 MB
- Write forces: 240 MB
- Spring updates: 120 MB
- PM grid FFT: 1 GB
- Neighbor access: 2 GB (cached)
Total: ~4 GB/frame = 240 GB/s (within RTX 3060's 360 GB/s)
```

## Optimization Checklist

### GPU Profiling Targets
- SM Occupancy > 50%
- Memory Bandwidth Utilization > 60%
- Warp Execution Efficiency > 70%
- L2 Cache Hit Rate > 80%
- No uncoalesced memory access

### CPU Profiling Targets
- Vectorization ratio > 75%
- Cache miss rate < 5%
- Thread scaling efficiency > 80%
- Branch misprediction < 2%

## Libraries and Tools

### GPU Stack
- **cuFFT**: PM solver FFTs
- **CUB**: Parallel primitives (sort, scan)
- **Thrust**: High-level algorithms
- **CUDA Graphs**: Reduce kernel launch overhead
- **NCCL**: Multi-GPU communication

### CPU Stack
- **FFTW**: CPU-optimized FFTs
- **Intel MKL**: BLAS/LAPACK operations
- **OpenMP**: Thread parallelization
- **TBB**: Task-based parallelism
- **ISPC**: Intel SPMD compiler

### Profiling Tools
- **Nsight Compute**: GPU kernel analysis
- **Nsight Systems**: System-wide profiling
- **VTune**: CPU performance analysis
- **perf**: Linux performance counters

## Implementation Roadmap

### Phase 1: GPU Foundation (Weeks 1-2)
- [x] Basic CUDA backend structure
- [ ] PM solver with cuFFT
- [ ] Simple spatial hash for collisions
- [ ] Test with 1M particles

### Phase 2: CPU Fallback (Weeks 3-4)
- [ ] OpenMP CPU backend
- [ ] AVX2 particle updates
- [ ] FFTW integration
- [ ] Test with 100K particles

### Phase 3: Optimization (Weeks 5-6)
- [ ] Morton ordering
- [ ] Mixed precision
- [ ] Kernel fusion
- [ ] Memory pooling

### Phase 4: Hybrid System (Weeks 7-8)
- [ ] CPU/GPU work division
- [ ] Shared memory IPC
- [ ] Dynamic load balancing
- [ ] Unified API

### Phase 5: Production (Weeks 9-10)
- [ ] Multi-GPU support
- [ ] Adaptive quality
- [ ] Platform detection
- [ ] Performance validation

## Key Insights

1. **GPU is mandatory for 10M+ particles** - No CPU can match GPU parallelism
2. **PM method is ideal for our toroidal space** - Natural periodicity
3. **Three-tier spatial indexing** - Right algorithm for each force scale
4. **CPU excels at complex logic** - AI, game rules, player input
5. **Hybrid approach maximizes hardware** - GPU physics, CPU intelligence
6. **Cache-friendly layouts crucial** - SoA for GPU, AoS for CPU logic
7. **Mixed precision acceptable** - FP32 positions, FP16 forces
8. **Visual plausibility > accuracy** - Games can cheat, simulations cannot

## Summary

The optimal architecture:
- **GPU**: PM gravity, collision detection, spring forces, particle integration
- **CPU**: AI agents, game logic, special objects, network, UI
- **IPC**: posix_shm for zero-copy CPU↔GPU communication
- **Fallback**: CPU-only mode for development with < 100K particles

This gives us massive scale on GPU while maintaining flexibility for game features on CPU.
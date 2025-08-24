# Multi-Range Force Calculation System

## Force Ranges and Methods

We need to handle THREE distinct force ranges efficiently:

1. **Contact Range (0 - 2r)**: Soft-body repulsion, spring formation
2. **Local Range (2r - 100r)**: Springs, local interactions  
3. **Global Range (100r - ∞)**: Gravity, electrostatics

Each requires different algorithms for efficiency!

## Unified Architecture

```cuda
// Key insight: Different data structures for different ranges
struct ParticleSystem {
    // Range 1: Contact forces (spatial hash)
    SpatialHash contact_hash;     // 2r cell size
    
    // Range 2: Local forces (neighbor lists)  
    NeighborList local_neighbors; // 100r cutoff
    
    // Range 3: Global forces (PM grid)
    PMGrid gravity_grid;          // Full domain
    
    // Spring network (sparse matrix on GPU)
    SpringNetwork springs;
};
```

## Algorithm Pipeline

### Phase 1: Spatial Sorting (Once per frame)
```cuda
__global__ void morton_sort_particles(Particle* particles, uint32_t* morton_codes) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float3 pos = particles[idx].pos;
    
    // Z-order curve for cache-friendly access
    morton_codes[idx] = morton3D(pos);
}
// Then: cub::DeviceRadixSort::SortPairs(morton_codes, particle_indices)
```

### Phase 2: Build Acceleration Structures (Parallel)

```cuda
__global__ void build_structures(Particle* particles) {
    // Three kernels run concurrently on different streams
    
    // Stream 1: Contact hash (fine-grained)
    build_spatial_hash<<<>>>(particles, cell_size=2*radius);
    
    // Stream 2: Neighbor lists (medium-range)
    build_neighbor_lists<<<>>>(particles, cutoff=100*radius);
    
    // Stream 3: PM grid (coarse)
    project_to_pm_grid<<<>>>(particles, grid_256x256x256);
}
```

### Phase 3: Force Calculation (Fused Kernel)

```cuda
__global__ void compute_all_forces(
    Particle* particles,
    SpatialHash* contact_hash,
    NeighborList* neighbors,
    PMGrid* pm_grid,
    SpringNetwork* springs,
    float3* forces_out
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float3 pos = particles[idx].pos;
    float3 total_force = make_float3(0);
    
    // 1. CONTACT FORCES (Soft-body repulsion)
    uint cell = hash_position(pos, 2*radius);
    for (int neighbor_cell : get_27_neighbors(cell)) {
        for (int j : contact_hash[neighbor_cell]) {
            if (j <= idx) continue; // Skip self and duplicates
            
            float3 r = particles[j].pos - pos;
            float dist = length(r);
            
            if (dist < 2*radius) {
                // Soft exponential repulsion
                float overlap = 2*radius - dist;
                float force_mag = REPULSION_STRENGTH * exp(-dist/radius);
                total_force -= normalize(r) * force_mag;
                
                // Check for spring formation (virtual spring field)
                if (should_form_spring(particles[idx], particles[j])) {
                    springs->try_add_spring(idx, j, dist);
                }
            }
        }
    }
    
    // 2. LOCAL FORCES (Springs + medium-range)
    // Process existing springs
    for (Spring s : springs->get_springs_for(idx)) {
        float3 r = particles[s.other].pos - pos;
        float dist = length(r);
        
        // Hooke's law with damping
        float3 spring_force = s.k * (dist - s.rest_length) * normalize(r);
        float3 damping = -s.damping * (particles[idx].vel - particles[s.other].vel);
        total_force += spring_force + damping;
        
        // Break springs if overstretched
        if (dist > s.break_distance) {
            springs->mark_for_removal(s.id);
        }
    }
    
    // Medium-range interactions (stored in neighbor list)
    for (int j : neighbors[idx]) {
        float3 r = particles[j].pos - pos;
        float dist2 = dot(r, r);
        
        if (dist2 < MEDIUM_CUTOFF2) {
            // Electrostatics, Van der Waals, etc.
            float3 force = compute_medium_range_force(particles[idx], particles[j], r);
            total_force += force;
        }
    }
    
    // 3. GLOBAL FORCES (Gravity via PM)
    // Interpolate from pre-computed PM grid
    float3 gravity = trilinear_interpolate(pm_grid, pos);
    total_force += gravity * particles[idx].mass;
    
    // Write once to global memory
    forces_out[idx] = total_force;
}
```

## Spring Network on GPU

### Sparse Spring Storage
```cuda
// COO format for dynamic springs (easy add/remove)
struct SpringNetwork {
    int* particle1;      // Source particle index
    int* particle2;      // Target particle index  
    float* rest_length;
    float* stiffness;
    float* damping;
    float* break_force;
    
    int* num_springs;    // Atomic counter
    int max_springs;     // Pre-allocated maximum
    
    // Per-particle spring lists (for fast lookup)
    int* spring_offsets; // CSR format offsets
    int* spring_indices; // CSR format indices
};
```

### Dynamic Spring Formation
```cuda
__device__ bool should_form_spring(Particle& p1, Particle& p2) {
    float3 r = p2.pos - p1.pos;
    float dist = length(r);
    
    // Distance criterion
    if (dist > SPRING_FORM_DISTANCE) return false;
    
    // Relative velocity criterion (must be slow)
    float3 v_rel = p2.vel - p1.vel;
    if (length(v_rel) > SPRING_FORM_VEL_THRESHOLD) return false;
    
    // Temperature criterion (cold particles stick)
    if (p1.temp + p2.temp > SPRING_FORM_TEMP_THRESHOLD) return false;
    
    return true;
}

__device__ void try_add_spring(int idx1, int idx2, float dist) {
    // Atomic add to spring list
    int spring_id = atomicAdd(num_springs, 1);
    if (spring_id >= max_springs) {
        atomicSub(num_springs, 1);
        return;
    }
    
    particle1[spring_id] = idx1;
    particle2[spring_id] = idx2;
    rest_length[spring_id] = dist;
    stiffness[spring_id] = calculate_stiffness(idx1, idx2);
    damping[spring_id] = DEFAULT_DAMPING;
}
```

## Optimization Strategies

### 1. Temporal Coherence
```cuda
// Reuse neighbor lists across frames
if (frame_count % NEIGHBOR_REBUILD_FREQUENCY == 0) {
    rebuild_neighbor_lists();
} else {
    verify_and_patch_neighbor_lists();  // Just check boundaries
}
```

### 2. Mixed Precision
```cuda
// Positions need FP32, forces can use FP16
__device__ half3 compute_repulsion_force_fp16(float3 r, float dist) {
    half force_mag = __float2half(REPULSION_STRENGTH * exp(-dist/radius));
    half3 direction = __float2half3_rn(normalize(r));
    return direction * force_mag;
}
```

### 3. Warp-Level Primitives
```cuda
// Use warp shuffles to reduce atomic pressure
__device__ void warp_accumulate_forces(float3& force) {
    // Sum across warp first
    force.x += __shfl_down_sync(0xffffffff, force.x, 16);
    force.x += __shfl_down_sync(0xffffffff, force.x, 8);
    force.x += __shfl_down_sync(0xffffffff, force.x, 4);
    force.x += __shfl_down_sync(0xffffffff, force.x, 2);
    force.x += __shfl_down_sync(0xffffffff, force.x, 1);
    
    // Only lane 0 does atomic
    if (threadIdx.x % 32 == 0) {
        atomicAdd(&global_force.x, force.x);
    }
}
```

### 4. Adaptive Cell Sizes
```cuda
struct AdaptiveSpatialHash {
    // Different cell sizes for different densities
    float get_cell_size(float3 pos) {
        float density = estimate_local_density(pos);
        if (density > HIGH_DENSITY) return 1.0f * radius;
        if (density > MEDIUM_DENSITY) return 2.0f * radius;
        return 4.0f * radius;
    }
};
```

## Performance Estimates

### Memory Requirements (20M particles)
```
Particles: 20M × 64 bytes = 1.28 GB
Spatial Hash: 10M cells × 16 bytes = 160 MB  
Neighbor Lists: 20M × 50 neighbors × 4 bytes = 4 GB
PM Grid: 256³ × 16 bytes = 256 MB
Springs: 5M springs × 32 bytes = 160 MB
-----------------------------------------
Total: ~6 GB (fits easily in 12 GB)
```

### Bandwidth Analysis
```
Per frame (60 FPS):
- Read particles: 1.28 GB
- Write forces: 240 MB
- Spring updates: 160 MB
- Neighbor access: 2 GB (cached)
Total: ~3.7 GB/frame = 222 GB/s (within 360 GB/s limit)
```

### Compute Distribution
```
Contact forces (5%): ~1000 GFLOPS
Spring forces (15%): ~3000 GFLOPS  
Local forces (20%): ~4000 GFLOPS
PM interpolation (10%): ~2000 GFLOPS
Integration (5%): ~1000 GFLOPS
Spring management (5%): ~1000 GFLOPS
-----------------------------------------
Total: ~12 TFLOPS (within 13 TFLOP budget)
```

## Implementation Priority

1. **Week 1**: Basic spatial hash + repulsion forces
2. **Week 2**: PM grid for gravity
3. **Week 3**: Dynamic spring formation
4. **Week 4**: Neighbor lists for medium-range
5. **Week 5**: Optimization and tuning

## Key Innovations

1. **Three-tier spatial indexing** - Right algorithm for each scale
2. **Unified force kernel** - Single pass, all forces
3. **Dynamic spring network** - Forms and breaks in parallel
4. **Adaptive structures** - Adjusts to particle distribution

This gives us soft-body physics with emergent structures at massive scale!
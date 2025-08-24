# Spatial Indexing Design

## Overview

DigiStar targets 10+ million particles at 60 FPS. Our experiments show that dense grids fail catastrophically at this scale - requiring 6TB of RAM for a realistic world. This document describes our **sparse hierarchical grid** approach that scales to millions of particles using only hundreds of MB.

## The Scale Problem

For 10M particles in realistic space:
- World size: 1,000,000 × 1,000,000 units
- Cell size for collisions: 2 units
- Dense grid cells needed: 250 BILLION
- Memory for dense grid: 6,000 GB
- **Actual occupied cells: ~1 million**
- Memory for sparse grid: 100 MB

**Conclusion: Sparse storage is mandatory, not optional.**

## Sparse Grid Architecture

### Core Data Structure

```cpp
class SparseGrid {
    std::unordered_map<uint64_t, std::vector<uint32_t>> cells;
    float cell_size;
    float world_size;
    
    uint64_t hash_cell(int x, int y) const {
        // Handle toroidal wraparound
        x = (x + grid_resolution) % grid_resolution;
        y = (y + grid_resolution) % grid_resolution;
        return (uint64_t(x) << 32) | uint64_t(y);
    }
};
```

### Why Hash Maps Win

**Memory Efficiency:**
- Only stores occupied cells (~1M vs 250B)
- 100MB vs 6TB memory usage
- O(1) average case lookup

**Cache Performance:**
- Better locality than nested vectors
- No pointer chasing through empty cells
- Predictable memory access patterns

## Incremental Updates

Our experiments show that **only ~1% of particles change cells per frame**. This enables massive optimization:

### Particle Tracking

```cpp
struct Particle {
    float2 pos, vel;
    float mass, radius;
    int32_t current_cell;  // Track which cell particle is in
};
```

### Incremental Update Algorithm

```cpp
void incremental_update(std::vector<Particle>& particles) {
    #pragma omp parallel for
    for (uint32_t i = 0; i < particles.size(); i++) {
        int32_t new_cell = get_cell_index(particles[i].pos);
        int32_t old_cell = particles[i].current_cell;
        
        if (new_cell != old_cell) {  // Only ~1% of particles
            #pragma omp critical
            {
                // Remove from old cell
                cells[old_cell].erase(particle_id);
                // Add to new cell
                cells[new_cell].push_back(particle_id);
            }
            particles[i].current_cell = new_cell;
        }
    }
}
```

**Performance Impact:**
- Full rebuild: 1140ms for 2M particles
- Incremental update: 28ms for 2M particles
- **40x speedup!**

## Hierarchical Multi-Scale Grids

Different forces need different resolutions:

| Force Type | Range | Cell Size | Typical Queries/Frame |
|------------|-------|-----------|----------------------|
| Collisions | ~2 units | 2-4 units | All particles |
| Springs | ~10 units | 10-20 units | Spring endpoints |
| Radiation | ~100 units | 100-200 units | Hot particles |
| Long-range | ~1000 units | 1000+ units | Massive bodies |

### Implementation

```cpp
class HierarchicalSparseGrid {
    struct Level {
        std::unordered_map<uint64_t, std::vector<uint32_t>> cells;
        float cell_size;
        
        // Each level tracks different particle properties
        void add_particle(uint32_t id, float2 pos) {
            uint64_t key = hash_cell(pos / cell_size);
            cells[key].push_back(id);
        }
    };
    
    Level collision_level;  // Fine: 2-unit cells
    Level spring_level;     // Medium: 20-unit cells
    Level radiation_level;  // Coarse: 200-unit cells
};
```

## Toroidal Space Handling

Our space wraps around, requiring special care:

### Cell Size Must Divide World Size

```cpp
// CRITICAL: Ensure even division for proper wraparound
int resolution = round_to_power_of_2(world_size / desired_cell_size);
float actual_cell_size = world_size / resolution;  // Evenly divides
```

### Wraparound Distance Calculation

```cpp
float2 min_distance(float2 a, float2 b, float world_size) {
    float dx = b.x - a.x;
    float dy = b.y - a.y;
    
    // Find shortest path through wraparound
    if (dx > world_size / 2) dx -= world_size;
    if (dx < -world_size / 2) dx += world_size;
    if (dy > world_size / 2) dy -= world_size;
    if (dy < -world_size / 2) dy += world_size;
    
    return float2(dx, dy);
}
```

## Performance Optimizations

### 1. Parallel Construction with OpenMP

```cpp
// Thread-local maps to avoid contention
#pragma omp parallel
{
    std::unordered_map<uint64_t, std::vector<uint32_t>> local_cells;
    
    #pragma omp for
    for (int i = 0; i < num_particles; i++) {
        uint64_t key = hash_cell(particles[i].pos);
        local_cells[key].push_back(i);
    }
    
    // Merge thread-local maps
    #pragma omp critical
    {
        for (auto& [key, indices] : local_cells) {
            cells[key].insert(cells[key].end(), 
                            indices.begin(), indices.end());
        }
    }
}
```

### 2. Morton Ordering for Cache Efficiency

```cpp
// Sort particles by Morton code for spatial locality
uint32_t morton2D(uint16_t x, uint16_t y) {
    uint32_t result = 0;
    for (int i = 0; i < 16; i++) {
        result |= (x & (1u << i)) << i | (y & (1u << i)) << (i + 1);
    }
    return result;
}

// Sort particles once per N frames
if (frame % 100 == 0) {
    sort_particles_by_morton_code();
}
```

### 3. Reserve Hash Map Capacity

```cpp
// Pre-allocate to avoid rehashing
cells.reserve(num_particles / 10);  // ~10 particles per cell
```

## Scaling Analysis

### CPU Performance (Measured)

| Particles | Full Rebuild | Incremental | Memory | FPS Potential |
|-----------|-------------|-------------|---------|---------------|
| 100K | 20ms | 1.7ms | 12 MB | 580 FPS |
| 500K | 222ms | 8.9ms | 64 MB | 112 FPS |
| 1M | 486ms | 14.4ms | 129 MB | 69 FPS |
| 2M | 1140ms | 27.8ms | 260 MB | 36 FPS |

### GPU Scaling (Projected)

For 10M+ particles, GPU becomes necessary:

**GPU Advantages:**
- Parallel hash table construction (100x faster)
- No atomic contention with proper design
- Shared memory for cell data
- 10,000+ threads in flight

**Expected Performance:**
- 10M particles: < 5ms update (200 FPS)
- 50M particles: < 20ms update (50 FPS)
- 100M particles: < 40ms update (25 FPS)

## Memory Requirements

For 10M particles:

```
Particles:     10M × 32 bytes = 320 MB
Sparse Grid:   1M cells × 64 bytes = 64 MB per level
4 Grid Levels: 256 MB
Springs:       5M × 24 bytes = 120 MB
--------------------------------------
Total:         ~700 MB (fits in 1GB)
```

Compare to dense grid: **6,000 GB** → Sparse is **8,500x smaller**

## Implementation Roadmap

### Phase 1: CPU Sparse Grid ✓
- [x] Hash map implementation
- [x] Incremental updates
- [x] 2M particles at 36 FPS

### Phase 2: Multi-Level Hierarchy
- [ ] Separate grids per force type
- [ ] Automatic level selection
- [ ] Cross-level queries

### Phase 3: GPU Acceleration
- [ ] CUDA sparse grid
- [ ] Parallel hash construction
- [ ] 10M+ particles at 60 FPS

### Phase 4: Advanced Features
- [ ] Concurrent hash maps (no locks)
- [ ] Predictive cell updates
- [ ] Adaptive cell sizing

## Key Insights

1. **Dense grids are impossible at scale** - 6TB for realistic worlds
2. **Sparse grids are mandatory** - 100MB for same functionality
3. **Incremental updates are critical** - 40x speedup from ~1% change rate
4. **Toroidal space needs even division** - cell_size must divide world_size
5. **CPU can handle 2M particles** - with proper data structures
6. **GPU needed for 10M+** - parallel hash construction is key

## Integration with Physics Engine

```cpp
class PhysicsEngine {
    SparseGrid collision_grid;   // Fine resolution
    SparseGrid spring_grid;      // Medium resolution  
    PMSolver gravity_solver;     // Separate FFT grid
    
    void update(float dt) {
        // Parallel updates
        #pragma omp parallel sections
        {
            #pragma omp section
            collision_grid.incremental_update(particles);
            
            #pragma omp section
            spring_grid.incremental_update(particles);
            
            #pragma omp section
            gravity_solver.update_density_field(particles);
        }
        
        // Calculate forces using appropriate grids
        calculate_collision_forces(collision_grid);
        calculate_spring_forces(spring_grid);
        calculate_gravity_forces(gravity_solver);
    }
};
```

## Conclusion

Sparse hierarchical grids with incremental updates are the **only viable approach** for millions of particles. Dense grids that work fine for thousands of particles fail catastrophically at scale. Our experiments prove that with proper data structures, CPU can handle 2M particles at interactive rates, with clear path to 10M+ on GPU.
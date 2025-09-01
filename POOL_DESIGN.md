# Pool Data Structure Design

## Current Issues with ParticlePool

1. **No particle removal** - Can add but not remove particles
2. **active_indices not maintained** - Initialized but never updated
3. **Fragmentation** - Dead particles waste memory and computation
4. **ID/Index confusion** - No stable particle IDs when array is compacted
5. **Inefficient iteration** - Must check `alive` flag for every particle

## Design Goals

1. **Cache Efficiency** - Keep active particles contiguous in memory
2. **SIMD Friendly** - Structure of Arrays (SoA) layout for vectorization
3. **Stable IDs** - Track particles even as they move in memory
4. **Fast Iteration** - Iterate only over active particles, no branching
5. **Dynamic** - Efficient add/remove operations
6. **Toroidal-aware** - Handle periodic boundary conditions

## Proposed Design

### Core Concepts

```
MEMORY LAYOUT:
[ACTIVE PARTICLES.....................][DEAD/UNUSED...................]
^                                      ^                               ^
0                                      count                           capacity

- All active particles are kept contiguous at the front
- No gaps between active particles
- O(1) iteration over active particles
```

### ID vs Index
- **Index**: Position in the array (changes when compacted)
- **ID**: Stable identifier (never changes for a particle's lifetime)
- Need bidirectional mapping: ID ↔ Index

### Key Operations

```cpp
class ParticlePool {
    // Core API
    uint32_t create(float x, float y, float vx, float vy, float mass, float radius);
    void destroy(uint32_t id);
    bool exists(uint32_t id) const;
    
    // Bulk operations
    void clear_forces();
    void apply_boundaries(float world_size);  // Toroidal wrapping
    
    // Iteration (only active particles)
    size_t size() const { return count; }
    
    // Direct array access for physics kernels
    float* x() { return pos_x; }
    float* y() { return pos_y; }
    // ... etc
    
    // ID/Index mapping
    uint32_t id_to_index(uint32_t id) const;
    uint32_t index_to_id(uint32_t index) const;
    
private:
    // Memory management
    void compact();  // Remove gaps
    void grow();     // Expand capacity
};
```

## Implementation Strategy

### Option 1: Swap-and-Pop (Recommended)
When removing particle at index `i`:
1. Swap particle `i` with particle at `count-1`
2. Decrement `count`
3. Update ID→Index mapping

**Pros:**
- Always compact, no fragmentation
- O(1) removal
- Best cache usage

**Cons:**
- Changes particle order (usually fine for physics)
- Must update external references

### Option 2: Generation-based IDs
```cpp
struct ParticleID {
    uint32_t index : 24;     // Array index
    uint32_t generation : 8; // Increment on reuse
};
```
- Detect stale references
- Reuse array slots safely

### Option 3: Indirect Index Table
```cpp
uint32_t* id_to_index;  // Maps ID → current index
uint32_t* index_to_id;  // Maps index → ID
```
- Stable IDs even with compaction
- Small overhead for indirection

## Pool Types Needed

### 1. ParticlePool
- Largest, most performance-critical
- Needs all optimizations
- Used in tight physics loops

### 2. SpringPool
```cpp
struct SpringPool {
    uint32_t* particle1_id;  // Use IDs, not indices!
    uint32_t* particle2_id;
    float* rest_length;
    float* stiffness;
    float* damping;
    float* current_length;  // Cached for performance
    float* strain;          // Cached
    bool* broken;           // Lazy deletion
};
```

### 3. ContactPool
- Rebuilt every frame
- No need for stable IDs
- Can use simple array

### 4. CompositePool
- Maps to groups of particles
- Needs careful handling during particle removal

## Usage Examples

```cpp
// Creating particles
uint32_t sun_id = particles.create(0, 0, 0, 0, 1e6, 50);
uint32_t earth_id = particles.create(100, 0, 0, 10, 1, 5);

// Physics loop (cache-friendly)
size_t n = particles.size();
float* x = particles.x();
float* y = particles.y();
float* fx = particles.fx();
float* fy = particles.fy();

#pragma omp simd
for (size_t i = 0; i < n; i++) {
    fx[i] = 0;
    fy[i] = 0;
}

// Apply forces
for (size_t i = 0; i < n; i++) {
    for (size_t j = i+1; j < n; j++) {
        // Calculate forces...
    }
}

// Integration
#pragma omp simd
for (size_t i = 0; i < n; i++) {
    vx[i] += fx[i] / mass[i] * dt;
    x[i] += vx[i] * dt;
}

// Toroidal boundaries
particles.apply_boundaries(world_size);

// Remove particle
particles.destroy(earth_id);
```

## Performance Considerations

### Memory Layout
- 64-byte aligned for AVX-512
- Consider memory prefetching
- Group frequently accessed fields

### Parallelization
- Ensure thread-safe operations
- Consider per-thread pools with periodic merging
- Read-only access during physics calculations

### GPU Considerations
- Need to mirror structure on GPU
- Minimize host-device transfers
- Consider pinned memory for async transfers

## Testing Strategy

1. **Unit tests**: Each operation in isolation
2. **Stress tests**: Add/remove many particles
3. **Invariant checks**: No gaps, correct count, valid mappings
4. **Performance benchmarks**: Measure cache misses, SIMD usage
5. **Integration tests**: With physics algorithms

## Migration Path

1. Start with simple implementation (current + remove operation)
2. Add ID/Index mapping
3. Implement swap-and-pop removal
4. Add generation-based IDs for safety
5. Optimize based on profiling
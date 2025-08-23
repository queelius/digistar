// test_spatial_index_scale.cpp - Testing scalability to millions of particles

#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <chrono>
#include <iomanip>
#include <unordered_map>

// Analyze memory and performance at scale
void analyze_scalability() {
    std::cout << "=== Spatial Index Scalability Analysis ===\n\n";
    
    // Current implementation analysis
    std::cout << "CURRENT IMPLEMENTATION ISSUES:\n";
    std::cout << "-------------------------------\n";
    
    // Issue 1: Resolution scaling
    std::cout << "1. GRID RESOLUTION PROBLEM:\n";
    std::cout << "   World size: 1,000,000 x 1,000,000 units (realistic for 10M particles)\n";
    std::cout << "   Cell size for collisions: 2 units\n";
    std::cout << "   Required resolution: 500,000 x 500,000 = 250 BILLION cells!\n";
    std::cout << "   Memory per cell (vector overhead): ~24 bytes minimum\n";
    std::cout << "   Total memory: 6,000 GB just for empty cells!\n\n";
    
    // Issue 2: Memory layout
    std::cout << "2. MEMORY LAYOUT ISSUES:\n";
    std::cout << "   - std::vector<std::vector<uint32_t>> is pointer-heavy\n";
    std::cout << "   - Poor cache locality (pointer chasing)\n";
    std::cout << "   - Dynamic allocation overhead per cell\n";
    std::cout << "   - Can't use for 250B cells\n\n";
    
    // Issue 3: Rebuild cost
    std::cout << "3. REBUILD COST AT SCALE:\n";
    std::cout << "   Current: 50 particles/ms (from benchmark)\n";
    std::cout << "   10M particles: 200,000 ms = 200 seconds per frame!\n";
    std::cout << "   Need: 10M / 16ms = 625,000 particles/ms (12,500x faster)\n\n";
    
    // Solutions needed
    std::cout << "\nSOLUTIONS NEEDED:\n";
    std::cout << "-----------------\n";
    
    std::cout << "1. SPARSE GRID STORAGE:\n";
    std::cout << "   - Use hash map for occupied cells only\n";
    std::cout << "   - 10M particles, ~10 per cell = 1M cells (not 250B)\n";
    std::cout << "   - Memory: 1M cells * 100 bytes = 100 MB (manageable)\n\n";
    
    std::cout << "2. BETTER DATA STRUCTURE:\n";
    std::cout << "```cpp\n";
    std::cout << "   struct SparseGrid {\n";
    std::cout << "       std::unordered_map<uint64_t, std::vector<uint32_t>> cells;\n";
    std::cout << "       \n";
    std::cout << "       uint64_t hash_cell(int x, int y) {\n";
    std::cout << "           return (uint64_t(x) << 32) | uint64_t(y);\n";
    std::cout << "       }\n";
    std::cout << "   };\n";
    std::cout << "```\n\n";
    
    std::cout << "3. PARALLEL REBUILD:\n";
    std::cout << "   - Partition particles into chunks\n";
    std::cout << "   - Each thread builds local hash map\n";
    std::cout << "   - Merge thread-local maps\n";
    std::cout << "   - Or use concurrent hash map\n\n";
    
    std::cout << "4. INCREMENTAL UPDATES:\n";
    std::cout << "   - Track which particles moved cells\n";
    std::cout << "   - Only update those cells\n";
    std::cout << "   - Most particles stay in same cell per frame\n\n";
    
    // Memory calculations
    std::cout << "\nMEMORY REQUIREMENTS (10M particles):\n";
    std::cout << "------------------------------------\n";
    
    size_t particle_size = 32;  // pos, vel, mass, radius, temp, id
    size_t particles_mem = 10000000ULL * particle_size;
    std::cout << "Particles array: " << (particles_mem / 1024 / 1024) << " MB\n";
    
    size_t avg_particles_per_cell = 10;
    size_t occupied_cells = 10000000ULL / avg_particles_per_cell;
    size_t hash_overhead = 32;  // hash map overhead per entry
    size_t vector_overhead = 24;  // vector overhead
    size_t indices_size = 4 * avg_particles_per_cell;  // uint32_t indices
    size_t cell_memory = occupied_cells * (hash_overhead + vector_overhead + indices_size);
    std::cout << "Sparse grid (1M occupied cells): " << (cell_memory / 1024 / 1024) << " MB\n";
    
    // For 4 grid levels
    size_t total_grid_mem = cell_memory * 4;
    std::cout << "All 4 grid levels: " << (total_grid_mem / 1024 / 1024) << " MB\n";
    
    size_t total_mem = particles_mem + total_grid_mem;
    std::cout << "Total memory: " << (total_mem / 1024 / 1024) << " MB\n\n";
    
    // Performance requirements
    std::cout << "PERFORMANCE REQUIREMENTS (60 FPS):\n";
    std::cout << "---------------------------------\n";
    std::cout << "Frame budget: 16.67 ms\n";
    std::cout << "Grid rebuild: < 2 ms\n";
    std::cout << "Force calculation: < 10 ms\n";
    std::cout << "Integration: < 2 ms\n";
    std::cout << "Other: < 2 ms\n\n";
    
    std::cout << "Required throughput:\n";
    std::cout << "- Rebuild: 5M particles/ms\n";
    std::cout << "- Neighbor queries: 1M queries/ms\n";
    std::cout << "- Force updates: 1M particles/ms\n\n";
    
    // GPU considerations
    std::cout << "GPU ADVANTAGES:\n";
    std::cout << "---------------\n";
    std::cout << "- Parallel hash table construction (100x faster)\n";
    std::cout << "- Shared memory for cell data\n";
    std::cout << "- Coalesced memory access\n";
    std::cout << "- 10,000+ threads in flight\n";
    std::cout << "- No pointer chasing\n\n";
    
    std::cout << "RECOMMENDED APPROACH:\n";
    std::cout << "--------------------\n";
    std::cout << "1. Use sparse hash grid (unordered_map)\n";
    std::cout << "2. Implement parallel rebuild with OpenMP\n";
    std::cout << "3. Use incremental updates where possible\n";
    std::cout << "4. Consider switching to GPU for 10M+ particles\n";
    std::cout << "5. Use fixed-size pools for particle indices\n";
    std::cout << "6. Implement Z-order curve for better cache locality\n";
}

// Benchmark sparse vs dense grid
void benchmark_sparse_grid() {
    std::cout << "\n=== Sparse Grid Benchmark ===\n\n";
    
    const int num_particles = 100000;
    const float world_size = 100000.0f;
    const float cell_size = 10.0f;
    const int grid_resolution = int(world_size / cell_size);
    
    std::cout << "Particles: " << num_particles << "\n";
    std::cout << "World: " << world_size << " x " << world_size << "\n";
    std::cout << "Cell size: " << cell_size << "\n";
    std::cout << "Grid resolution: " << grid_resolution << " x " << grid_resolution << "\n";
    std::cout << "Total cells if dense: " << (grid_resolution * grid_resolution) << "\n\n";
    
    // Simulate particle positions
    std::vector<std::pair<float, float>> positions(num_particles);
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(0, world_size);
    
    for (auto& pos : positions) {
        pos.first = dist(rng);
        pos.second = dist(rng);
    }
    
    // Test sparse grid with unordered_map
    {
        auto start = std::chrono::high_resolution_clock::now();
        
        std::unordered_map<uint64_t, std::vector<uint32_t>> sparse_grid;
        sparse_grid.reserve(num_particles / 10);  // Assume ~10 particles per cell
        
        for (uint32_t i = 0; i < num_particles; i++) {
            int cx = int(positions[i].first / cell_size);
            int cy = int(positions[i].second / cell_size);
            uint64_t key = (uint64_t(cx) << 32) | uint64_t(cy);
            sparse_grid[key].push_back(i);
        }
        
        auto end = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(end - start).count();
        
        std::cout << "SPARSE GRID (unordered_map):\n";
        std::cout << "  Build time: " << ms << " ms\n";
        std::cout << "  Occupied cells: " << sparse_grid.size() << "\n";
        std::cout << "  Memory (estimate): " << 
            (sparse_grid.size() * 64 + num_particles * 4) / 1024 << " KB\n";
        std::cout << "  Particles/ms: " << (num_particles / ms) << "\n\n";
    }
    
    // Dense grid would use too much memory at this scale
    std::cout << "DENSE GRID (std::vector<std::vector>):\n";
    std::cout << "  Would require: " << 
        (size_t(grid_resolution) * grid_resolution * 24) / 1024 / 1024 << " MB\n";
    std::cout << "  Status: SKIPPED (too much memory)\n\n";
    
    std::cout << "CONCLUSION:\n";
    std::cout << "-----------\n";
    std::cout << "Sparse grid is REQUIRED for large worlds\n";
    std::cout << "Dense grid only works for small worlds (<1000x1000)\n";
}

int main() {
    analyze_scalability();
    benchmark_sparse_grid();
    
    std::cout << "\n=== FINAL RECOMMENDATIONS ===\n";
    std::cout << "1. Current implementation works for <100K particles\n";
    std::cout << "2. Need sparse grid for 1M+ particles\n";
    std::cout << "3. Need GPU for 10M+ particles\n";
    std::cout << "4. Consider using existing libraries:\n";
    std::cout << "   - cuSpatial (NVIDIA)\n";
    std::cout << "   - parallel_hashmap\n";
    std::cout << "   - tbb::concurrent_hash_map\n";
    
    return 0;
}
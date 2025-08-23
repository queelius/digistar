// test_spatial_index.cpp - Hierarchical uniform grid spatial indexing

#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <chrono>
#include <algorithm>
#include <cassert>
#include <iomanip>
#include <functional>
#include <omp.h>

struct float2 {
    float x, y;
    
    float2(float x_ = 0, float y_ = 0) : x(x_), y(y_) {}
    float2 operator+(const float2& b) const { return float2(x + b.x, y + b.y); }
    float2 operator-(const float2& b) const { return float2(x - b.x, y - b.y); }
    float2 operator*(float s) const { return float2(x * s, y * s); }
    float2& operator+=(const float2& b) { x += b.x; y += b.y; return *this; }
    float length() const { return sqrtf(x * x + y * y); }
    float length_sq() const { return x * x + y * y; }
};

struct Particle {
    float2 pos;
    float2 vel;
    float mass;
    float radius;
    float temp;
    uint32_t id;
};

// Morton encoding for 2D (Z-order curve)
uint32_t morton2D(uint16_t x, uint16_t y) {
    uint32_t result = 0;
    for (int i = 0; i < 16; i++) {
        result |= (x & (1u << i)) << i | (y & (1u << i)) << (i + 1);
    }
    return result;
}

// Single level of hierarchical grid
class GridLevel {
private:
    int resolution;
    float cell_size;
    float world_size;
    std::vector<std::vector<uint32_t>> cells;
    
public:
    GridLevel(int res, float cell_sz, float world_sz) 
        : resolution(res), cell_size(cell_sz), world_size(world_sz) {
        cells.resize(resolution * resolution);
    }
    
    void clear() {
        for (auto& cell : cells) {
            cell.clear();
        }
    }
    
    int get_cell_index(float2 pos) const {
        // Handle toroidal wraparound - ensure positive coordinates
        float x = pos.x;
        float y = pos.y;
        
        // Wrap negative coordinates
        while (x < 0) x += world_size;
        while (y < 0) y += world_size;
        
        // Wrap positive coordinates
        x = fmodf(x, world_size);
        y = fmodf(y, world_size);
        
        int ix = int(x / cell_size) % resolution;
        int iy = int(y / cell_size) % resolution;
        
        // Safety check (shouldn't be needed but just in case)
        if (ix < 0) ix += resolution;
        if (iy < 0) iy += resolution;
        
        return iy * resolution + ix;
    }
    
    void add_particle(uint32_t id, float2 pos) {
        int cell_idx = get_cell_index(pos);
        cells[cell_idx].push_back(id);
    }
    
    void get_neighbors(int x, int y, std::vector<int>& neighbors) const {
        neighbors.clear();
        for (int dy = -1; dy <= 1; dy++) {
            for (int dx = -1; dx <= 1; dx++) {
                int nx = (x + dx + resolution) % resolution;
                int ny = (y + dy + resolution) % resolution;
                neighbors.push_back(ny * resolution + nx);
            }
        }
    }
    
    const std::vector<uint32_t>& get_cell(int idx) const {
        return cells[idx];
    }
    
    float get_cell_size() const { return cell_size; }
    int get_resolution() const { return resolution; }
    
    // Statistics
    void print_stats() const {
        int empty_cells = 0;
        int max_occupancy = 0;
        float avg_occupancy = 0;
        
        for (const auto& cell : cells) {
            if (cell.empty()) empty_cells++;
            max_occupancy = std::max(max_occupancy, (int)cell.size());
            avg_occupancy += cell.size();
        }
        
        avg_occupancy /= cells.size();
        
        std::cout << "  Resolution: " << resolution << "x" << resolution
                  << ", Cell size: " << cell_size << " units\n";
        std::cout << "  Empty cells: " << empty_cells << "/" << cells.size()
                  << " (" << (100.0f * empty_cells / cells.size()) << "%)\n";
        std::cout << "  Avg occupancy: " << avg_occupancy
                  << ", Max occupancy: " << max_occupancy << "\n";
    }
};

// Hierarchical grid with multiple levels
class HierarchicalGrid {
private:
    static constexpr int NUM_LEVELS = 4;
    static constexpr float CELL_SIZES[NUM_LEVELS] = {2.0f, 20.0f, 200.0f, 2000.0f};
    
    std::vector<GridLevel> levels;
    float world_size;
    std::vector<Particle>* particles_ptr;
    
public:
    HierarchicalGrid(float world_sz, std::vector<Particle>* particles) 
        : world_size(world_sz), particles_ptr(particles) {
        
        // Create grid levels with appropriate resolutions
        for (int i = 0; i < NUM_LEVELS; i++) {
            // Calculate ideal resolution
            int ideal_resolution = int(world_size / CELL_SIZES[i]);
            
            // Round to nearest power of 2 for performance
            int resolution = 1 << (int)round(log2(ideal_resolution));
            
            // Adjust cell size to evenly divide world size
            float actual_cell_size = world_size / resolution;
            
            levels.emplace_back(resolution, actual_cell_size, world_size);
        }
    }
    
    void rebuild() {
        // Clear all levels
        for (auto& level : levels) {
            level.clear();
        }
        
        // Add particles to all levels
        for (uint32_t i = 0; i < particles_ptr->size(); i++) {
            const float2& pos = (*particles_ptr)[i].pos;
            for (auto& level : levels) {
                level.add_particle(i, pos);
            }
        }
    }
    
    // Find particles within radius using appropriate grid level
    void find_neighbors(float2 pos, float radius, std::vector<uint32_t>& result) {
        result.clear();
        
        // Choose appropriate grid level based on search radius
        int level_idx = 0;
        for (int i = 0; i < NUM_LEVELS; i++) {
            if (radius <= CELL_SIZES[i] * 2) {
                level_idx = i;
                break;
            }
        }
        
        GridLevel& level = levels[level_idx];
        
        // Get center cell and its 9 neighbors (3x3 grid)
        int cell_idx = level.get_cell_index(pos);
        int cx = cell_idx % level.get_resolution();
        int cy = cell_idx / level.get_resolution();
        
        std::vector<int> neighbor_cells;
        level.get_neighbors(cx, cy, neighbor_cells);
        
        float radius_sq = radius * radius;
        
        // Search the 9 cells
        for (int ncell : neighbor_cells) {
            const auto& particles_in_cell = level.get_cell(ncell);
            for (uint32_t pid : particles_in_cell) {
                // Calculate minimum distance considering wraparound
                float2 ppos = (*particles_ptr)[pid].pos;
                float dx = ppos.x - pos.x;
                float dy = ppos.y - pos.y;
                
                // Find shortest path through wraparound
                if (dx > world_size / 2) dx -= world_size;
                if (dx < -world_size / 2) dx += world_size;
                if (dy > world_size / 2) dy -= world_size;
                if (dy < -world_size / 2) dy += world_size;
                
                float dist_sq = dx * dx + dy * dy;
                if (dist_sq <= radius_sq) {
                    result.push_back(pid);
                }
            }
        }
    }
    
    void print_stats() const {
        std::cout << "\nHierarchical Grid Statistics:\n";
        std::cout << "World size: " << world_size << " x " << world_size << "\n";
        
        for (int i = 0; i < levels.size(); i++) {
            std::cout << "\nLevel " << i << " (for " 
                      << ((i == 0) ? "collisions" : 
                          (i == 1) ? "springs" :
                          (i == 2) ? "radiation" : "long-range") << "):\n";
            levels[i].print_stats();
        }
    }
    
    GridLevel& get_level(int idx) { return levels[idx]; }
};

// Benchmark different operations
class SpatialIndexBenchmark {
private:
    std::vector<Particle> particles;
    HierarchicalGrid grid;
    float world_size;
    
    double time_operation(std::function<void()> op) {
        auto start = std::chrono::high_resolution_clock::now();
        op();
        auto end = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double, std::milli>(end - start).count();
    }
    
public:
    SpatialIndexBenchmark(int num_particles, float world_sz) 
        : world_size(world_sz), grid(world_sz, &particles) {
        
        // Initialize particles
        std::mt19937 rng(42);
        std::uniform_real_distribution<float> pos_dist(0, world_size);
        std::uniform_real_distribution<float> vel_dist(-10, 10);
        std::uniform_real_distribution<float> mass_dist(1, 10);
        
        particles.resize(num_particles);
        for (int i = 0; i < num_particles; i++) {
            particles[i].pos = float2(pos_dist(rng), pos_dist(rng));
            particles[i].vel = float2(vel_dist(rng), vel_dist(rng));
            particles[i].mass = mass_dist(rng);
            particles[i].radius = 1.0f;
            particles[i].temp = 100.0f;
            particles[i].id = i;
        }
    }
    
    void benchmark_grid_rebuild() {
        std::cout << "\n=== Grid Rebuild Benchmark ===\n";
        
        double time = time_operation([this]() {
            grid.rebuild();
        });
        
        std::cout << "Time to rebuild grid: " << time << " ms\n";
        std::cout << "Particles per millisecond: " 
                  << (particles.size() / time) << "\n";
    }
    
    void benchmark_neighbor_search() {
        std::cout << "\n=== Neighbor Search Benchmark ===\n";
        
        grid.rebuild();
        
        // Test different search radii
        float radii[] = {2.0f, 10.0f, 50.0f, 200.0f};
        const char* names[] = {"Collision", "Spring", "Medium", "Long-range"};
        
        for (int i = 0; i < 4; i++) {
            std::vector<uint32_t> neighbors;
            int total_neighbors = 0;
            
            double time = time_operation([&]() {
                for (int j = 0; j < 1000; j++) {
                    float2 test_pos = particles[j % particles.size()].pos;
                    grid.find_neighbors(test_pos, radii[i], neighbors);
                    total_neighbors += neighbors.size();
                }
            });
            
            std::cout << names[i] << " range (" << radii[i] << " units):\n";
            std::cout << "  Time for 1000 queries: " << time << " ms\n";
            std::cout << "  Avg neighbors found: " << (total_neighbors / 1000.0f) << "\n";
            std::cout << "  Queries per millisecond: " << (1000.0 / time) << "\n";
        }
    }
    
    void benchmark_collision_detection() {
        std::cout << "\n=== Collision Detection Benchmark ===\n";
        
        grid.rebuild();
        
        int collisions = 0;
        double time = time_operation([&]() {
            GridLevel& collision_grid = grid.get_level(0);
            
            #pragma omp parallel for reduction(+:collisions)
            for (int cell_idx = 0; cell_idx < collision_grid.get_resolution() * 
                                               collision_grid.get_resolution(); cell_idx++) {
                const auto& cell = collision_grid.get_cell(cell_idx);
                
                // Check within cell
                for (int i = 0; i < cell.size(); i++) {
                    for (int j = i + 1; j < cell.size(); j++) {
                        float2 diff = particles[cell[i]].pos - particles[cell[j]].pos;
                        float dist_sq = diff.length_sq();
                        if (dist_sq < 4.0f) { // 2 * radius
                            collisions++;
                        }
                    }
                }
            }
        });
        
        std::cout << "Time to detect all collisions: " << time << " ms\n";
        std::cout << "Collisions found: " << collisions << "\n";
        std::cout << "Collision checks per millisecond: " 
                  << (particles.size() * particles.size() / 2.0 / time) << "\n";
    }
    
    void benchmark_morton_sort() {
        std::cout << "\n=== Morton Sort Benchmark ===\n";
        
        // Create morton codes
        std::vector<std::pair<uint32_t, uint32_t>> morton_pairs;
        morton_pairs.reserve(particles.size());
        
        double encode_time = time_operation([&]() {
            for (uint32_t i = 0; i < particles.size(); i++) {
                uint16_t x = uint16_t(particles[i].pos.x / world_size * 65535);
                uint16_t y = uint16_t(particles[i].pos.y / world_size * 65535);
                morton_pairs.push_back({morton2D(x, y), i});
            }
        });
        
        double sort_time = time_operation([&]() {
            std::sort(morton_pairs.begin(), morton_pairs.end());
        });
        
        // Reorder particles
        std::vector<Particle> sorted_particles(particles.size());
        double reorder_time = time_operation([&]() {
            for (size_t i = 0; i < morton_pairs.size(); i++) {
                sorted_particles[i] = particles[morton_pairs[i].second];
            }
            particles = std::move(sorted_particles);
        });
        
        std::cout << "Morton encoding time: " << encode_time << " ms\n";
        std::cout << "Sorting time: " << sort_time << " ms\n";
        std::cout << "Reordering time: " << reorder_time << " ms\n";
        std::cout << "Total time: " << (encode_time + sort_time + reorder_time) << " ms\n";
    }
    
    void run_all_benchmarks() {
        std::cout << "\n==================================================\n";
        std::cout << "Spatial Indexing Benchmark\n";
        std::cout << "Particles: " << particles.size() << "\n";
        std::cout << "World size: " << world_size << " x " << world_size << "\n";
        std::cout << "Threads: " << omp_get_max_threads() << "\n";
        std::cout << "==================================================\n";
        
        benchmark_grid_rebuild();
        grid.print_stats();
        benchmark_neighbor_search();
        benchmark_collision_detection();
        benchmark_morton_sort();
        
        std::cout << "\n==================================================\n";
    }
};

// Simple test to verify correctness
void test_correctness() {
    std::cout << "Running correctness tests...\n";
    
    std::vector<Particle> particles(4);  // Only 4 particles for testing
    float world_size = 1000.0f;
    
    // Create known particle positions
    particles[0].pos = float2(100, 100);
    particles[0].id = 0;
    particles[1].pos = float2(101, 100);  // Close to 0
    particles[1].id = 1;
    particles[2].pos = float2(200, 200);  // Far from 0
    particles[2].id = 2;
    particles[3].pos = float2(999, 100);  // Wraparound test
    particles[3].id = 3;
    
    HierarchicalGrid grid(world_size, &particles);
    grid.rebuild();
    
    // Test neighbor finding
    std::vector<uint32_t> neighbors;
    grid.find_neighbors(float2(100, 100), 5.0f, neighbors);
    
    assert(std::find(neighbors.begin(), neighbors.end(), 0) != neighbors.end());
    assert(std::find(neighbors.begin(), neighbors.end(), 1) != neighbors.end());
    assert(std::find(neighbors.begin(), neighbors.end(), 2) == neighbors.end());
    
    // Test wraparound - particle 3 is at (999, 100)
    // Distance from (1, 100) to (999, 100) through wraparound is 2 units
    grid.find_neighbors(float2(1, 100), 5.0f, neighbors);
    assert(std::find(neighbors.begin(), neighbors.end(), 3) != neighbors.end() && "Wraparound detection failed");
    
    std::cout << "All correctness tests passed!\n\n";
}

int main() {
    // Run correctness tests first
    test_correctness();
    
    // Benchmark with different particle counts
    int particle_counts[] = {1000, 10000, 100000};
    
    for (int count : particle_counts) {
        SpatialIndexBenchmark benchmark(count, 10000.0f);
        benchmark.run_all_benchmarks();
    }
    
    return 0;
}
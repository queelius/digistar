#pragma once

#include <unordered_map>
#include <vector>
#include <cstdint>
#include "types.h"

namespace digistar {

// Abstract spatial index interface
class SpatialIndex {
public:
    virtual ~SpatialIndex() = default;
    
    // Clear all data
    virtual void clear() = 0;
    
    // Add particle to index
    virtual void insert(uint32_t particle_id, float x, float y) = 0;
    
    // Remove particle from index
    virtual void remove(uint32_t particle_id, float x, float y) = 0;
    
    // Update particle position (optimized for small movements)
    virtual void update(uint32_t particle_id, float old_x, float old_y, float new_x, float new_y) = 0;
    
    // Query neighbors within radius
    virtual std::vector<uint32_t> query_radius(float x, float y, float radius) const = 0;
    
    // Query all particles in a cell
    virtual std::vector<uint32_t> query_cell(float x, float y) const = 0;
    
    // Get all neighboring cells (including diagonal)
    virtual std::vector<uint64_t> get_neighbor_cells(float x, float y) const = 0;
    
    // Statistics
    virtual size_t get_particle_count() const = 0;
    virtual size_t get_cell_count() const = 0;
    virtual float get_cell_size() const = 0;
};

// Sparse grid implementation using hash map
class SparseGrid : public SpatialIndex {
private:
    std::unordered_map<uint64_t, std::vector<uint32_t>> cells;
    std::unordered_map<uint32_t, uint64_t> particle_cells;  // Track which cell each particle is in
    
    float cell_size;
    float world_size;
    int grid_resolution;
    
    // Hash 2D cell coordinates to unique key
    uint64_t hash_cell(float x, float y) const {
        int cell_x = static_cast<int>(x / cell_size);
        int cell_y = static_cast<int>(y / cell_size);
        
        // Handle wrapping for toroidal space
        cell_x = ((cell_x % grid_resolution) + grid_resolution) % grid_resolution;
        cell_y = ((cell_y % grid_resolution) + grid_resolution) % grid_resolution;
        
        return (static_cast<uint64_t>(cell_x) << 32) | static_cast<uint64_t>(cell_y);
    }
    
    // Extract cell coordinates from hash
    void unhash_cell(uint64_t hash, int& x, int& y) const {
        x = static_cast<int>(hash >> 32);
        y = static_cast<int>(hash & 0xFFFFFFFF);
    }
    
public:
    SparseGrid(float cell_size_, float world_size_) 
        : cell_size(cell_size_), world_size(world_size_) {
        grid_resolution = static_cast<int>(world_size / cell_size);
    }
    
    void clear() override {
        cells.clear();
        particle_cells.clear();
    }
    
    void insert(uint32_t particle_id, float x, float y) override {
        uint64_t cell_hash = hash_cell(x, y);
        cells[cell_hash].push_back(particle_id);
        particle_cells[particle_id] = cell_hash;
    }
    
    void remove(uint32_t particle_id, float x, float y) override {
        auto it = particle_cells.find(particle_id);
        if (it != particle_cells.end()) {
            uint64_t cell_hash = it->second;
            auto& cell = cells[cell_hash];
            cell.erase(std::remove(cell.begin(), cell.end(), particle_id), cell.end());
            
            // Remove empty cells to save memory
            if (cell.empty()) {
                cells.erase(cell_hash);
            }
            
            particle_cells.erase(it);
        }
    }
    
    void update(uint32_t particle_id, float old_x, float old_y, float new_x, float new_y) override {
        uint64_t old_cell = hash_cell(old_x, old_y);
        uint64_t new_cell = hash_cell(new_x, new_y);
        
        if (old_cell != new_cell) {
            // Remove from old cell
            auto& old_particles = cells[old_cell];
            old_particles.erase(
                std::remove(old_particles.begin(), old_particles.end(), particle_id),
                old_particles.end()
            );
            
            if (old_particles.empty()) {
                cells.erase(old_cell);
            }
            
            // Add to new cell
            cells[new_cell].push_back(particle_id);
            particle_cells[particle_id] = new_cell;
        }
    }
    
    std::vector<uint32_t> query_radius(float x, float y, float radius) const override {
        std::vector<uint32_t> result;
        
        int cells_to_check = static_cast<int>(std::ceil(radius / cell_size));
        int center_x, center_y;
        unhash_cell(hash_cell(x, y), center_x, center_y);
        
        for (int dy = -cells_to_check; dy <= cells_to_check; dy++) {
            for (int dx = -cells_to_check; dx <= cells_to_check; dx++) {
                int cell_x = center_x + dx;
                int cell_y = center_y + dy;
                
                // Wrap for toroidal space
                cell_x = ((cell_x % grid_resolution) + grid_resolution) % grid_resolution;
                cell_y = ((cell_y % grid_resolution) + grid_resolution) % grid_resolution;
                
                uint64_t cell_hash = (static_cast<uint64_t>(cell_x) << 32) | static_cast<uint64_t>(cell_y);
                
                auto it = cells.find(cell_hash);
                if (it != cells.end()) {
                    result.insert(result.end(), it->second.begin(), it->second.end());
                }
            }
        }
        
        return result;
    }
    
    std::vector<uint32_t> query_cell(float x, float y) const override {
        uint64_t cell_hash = hash_cell(x, y);
        auto it = cells.find(cell_hash);
        if (it != cells.end()) {
            return it->second;
        }
        return {};
    }
    
    std::vector<uint64_t> get_neighbor_cells(float x, float y) const override {
        std::vector<uint64_t> neighbors;
        neighbors.reserve(9);  // 3x3 grid in 2D
        
        int center_x, center_y;
        unhash_cell(hash_cell(x, y), center_x, center_y);
        
        for (int dy = -1; dy <= 1; dy++) {
            for (int dx = -1; dx <= 1; dx++) {
                int cell_x = center_x + dx;
                int cell_y = center_y + dy;
                
                // Wrap for toroidal space
                cell_x = ((cell_x % grid_resolution) + grid_resolution) % grid_resolution;
                cell_y = ((cell_y % grid_resolution) + grid_resolution) % grid_resolution;
                
                uint64_t cell_hash = (static_cast<uint64_t>(cell_x) << 32) | static_cast<uint64_t>(cell_y);
                neighbors.push_back(cell_hash);
            }
        }
        
        return neighbors;
    }
    
    size_t get_particle_count() const override {
        return particle_cells.size();
    }
    
    size_t get_cell_count() const override {
        return cells.size();
    }
    
    float get_cell_size() const override {
        return cell_size;
    }
    
    // Direct access for performance-critical code
    const std::unordered_map<uint64_t, std::vector<uint32_t>>& get_cells() const {
        return cells;
    }
};

// Union-Find for composite detection
class UnionFind {
private:
    std::vector<uint32_t> parent;
    std::vector<uint32_t> rank;
    
public:
    void reset(size_t n) {
        parent.resize(n);
        rank.resize(n);
        for (size_t i = 0; i < n; i++) {
            parent[i] = i;
            rank[i] = 0;
        }
    }
    
    uint32_t find(uint32_t x) {
        if (parent[x] != x) {
            parent[x] = find(parent[x]);  // Path compression
        }
        return parent[x];
    }
    
    void unite(uint32_t x, uint32_t y) {
        uint32_t root_x = find(x);
        uint32_t root_y = find(y);
        
        if (root_x == root_y) return;
        
        // Union by rank
        if (rank[root_x] < rank[root_y]) {
            parent[root_x] = root_y;
        } else if (rank[root_x] > rank[root_y]) {
            parent[root_y] = root_x;
        } else {
            parent[root_y] = root_x;
            rank[root_x]++;
        }
    }
    
    bool connected(uint32_t x, uint32_t y) {
        return find(x) == find(y);
    }
};

} // namespace digistar
#pragma once

#include <cstdint>
#include <vector>

namespace digistar {

// Base class for spatial indexing structures
class SpatialIndex {
public:
    virtual ~SpatialIndex() = default;
    
    // Clear all entries
    virtual void clear() = 0;
    
    // Add a particle to the index
    virtual void insert(uint32_t particle_id, float x, float y) = 0;
    
    // Find particles within radius of a point
    virtual std::vector<uint32_t> query(float x, float y, float radius) const = 0;
    
    // Find potential collision pairs
    virtual std::vector<std::pair<uint32_t, uint32_t>> findPairs(float max_distance) const = 0;
    
    // Update after particles have moved
    virtual void update() = 0;
};

// Simple grid-based spatial hash
class GridSpatialIndex : public SpatialIndex {
private:
    struct Cell {
        std::vector<uint32_t> particles;
    };
    
    float cell_size;
    size_t grid_width;
    size_t grid_height;
    std::vector<Cell> cells;
    
    size_t getCellIndex(float x, float y) const;
    
public:
    GridSpatialIndex(float world_size, float cell_size);
    ~GridSpatialIndex() = default;
    
    void clear() override;
    void insert(uint32_t particle_id, float x, float y) override;
    std::vector<uint32_t> query(float x, float y, float radius) const override;
    std::vector<std::pair<uint32_t, uint32_t>> findPairs(float max_distance) const override;
    void update() override;
};

} // namespace digistar
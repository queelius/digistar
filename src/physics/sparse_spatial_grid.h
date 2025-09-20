/**
 * Sparse Spatial Grid System - Following SPATIAL_INDEXING_DESIGN.md
 *
 * MANDATORY: Uses sparse hash maps, NOT dense arrays
 * - Dense grids need 6TB RAM for realistic worlds
 * - Sparse grids use only ~100MB for same functionality
 * - Only stores occupied cells
 */

#pragma once

#include <unordered_map>
#include <vector>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <functional>
#include <memory>

namespace digistar {

/**
 * Sparse spatial grid using hash map as per design docs
 * Only stores occupied cells, not empty space
 */
template<typename Particle>
class SparseSpatialGrid {
public:
    struct Config {
        float cell_size = 10.0f;       // Size of each grid cell
        float world_size = 10000.0f;   // Total world size
        bool toroidal = true;          // Use toroidal topology
    };

private:
public:  // Made public for backend access
    Config config;
    int grid_resolution;               // Number of cells per dimension

    // SPARSE storage - only occupied cells (as per SPATIAL_INDEXING_DESIGN.md)
    std::unordered_map<uint64_t, std::vector<uint32_t>> cells;

    // Track which cell each particle is in for incremental updates
    std::vector<int32_t> particle_cells;  // -1 = not tracked

    // Hash function for cell coordinates
    uint64_t hashCell(int cx, int cy) const {
        // Handle toroidal wraparound
        if (config.toroidal) {
            cx = ((cx % grid_resolution) + grid_resolution) % grid_resolution;
            cy = ((cy % grid_resolution) + grid_resolution) % grid_resolution;
        } else {
            cx = std::max(0, std::min(grid_resolution - 1, cx));
            cy = std::max(0, std::min(grid_resolution - 1, cy));
        }
        return (uint64_t(cx) << 32) | uint64_t(cy);
    }

    // Get cell coordinates from position
    std::pair<int, int> getCellCoords(float x, float y) const {
        float gx = (x + config.world_size * 0.5f) / config.cell_size;
        float gy = (y + config.world_size * 0.5f) / config.cell_size;
        return {int(gx), int(gy)};
    }

public:
    SparseSpatialGrid(const Config& cfg = Config()) : config(cfg) {
        // Ensure cell size evenly divides world size for proper toroidal wrap
        grid_resolution = std::round(config.world_size / config.cell_size);
        config.cell_size = config.world_size / grid_resolution;  // Adjust for even division

        // Reserve capacity to avoid rehashing (estimate ~10% occupancy)
        cells.reserve(grid_resolution * grid_resolution / 10);
    }

    // Initialize particle tracking
    void initParticleTracking(size_t num_particles) {
        particle_cells.resize(num_particles, -1);
    }

    // Clear all cells
    void clear() {
        cells.clear();
        std::fill(particle_cells.begin(), particle_cells.end(), -1);
    }

    // Full rebuild - O(N)
    void rebuild(const std::vector<Particle>& particles) {
        cells.clear();

        for (uint32_t i = 0; i < particles.size(); i++) {
            auto [cx, cy] = getCellCoords(particles[i].x, particles[i].y);
            uint64_t key = hashCell(cx, cy);
            cells[key].push_back(i);

            if (i < particle_cells.size()) {
                particle_cells[i] = key;
            }
        }
    }

    // Incremental update - O(k) where k = particles that changed cells (~1% per frame)
    void incrementalUpdate(const std::vector<Particle>& particles) {
        // Ensure tracking is initialized
        if (particle_cells.size() != particles.size()) {
            initParticleTracking(particles.size());
            rebuild(particles);  // First time - do full rebuild
            return;
        }

        // Update only particles that changed cells
        for (uint32_t i = 0; i < particles.size(); i++) {
            auto [cx, cy] = getCellCoords(particles[i].x, particles[i].y);
            uint64_t new_key = hashCell(cx, cy);
            uint64_t old_key = particle_cells[i];

            if (new_key != old_key) {  // Only ~1% of particles per frame
                // Remove from old cell
                if (old_key != -1) {
                    auto it = cells.find(old_key);
                    if (it != cells.end()) {
                        auto& vec = it->second;
                        vec.erase(std::remove(vec.begin(), vec.end(), i), vec.end());

                        // Remove cell if empty
                        if (vec.empty()) {
                            cells.erase(it);
                        }
                    }
                }

                // Add to new cell
                cells[new_key].push_back(i);
                particle_cells[i] = new_key;
            }
        }
    }

    // Process particle pairs within radius - PARALLEL VERSION
    template<typename Callback>
    void processPairs(const std::vector<Particle>& particles, float radius,
                      Callback callback) {
        int cell_radius = std::ceil(radius / config.cell_size);
        float radius2 = radius * radius;

        // Convert hash map to vector for parallel iteration
        std::vector<std::pair<uint64_t, std::vector<uint32_t>>> cell_list;
        cell_list.reserve(cells.size());
        for (const auto& [key, particles] : cells) {
            if (!particles.empty()) {
                cell_list.push_back({key, particles});
            }
        }

        // Parallel processing with OpenMP
        #pragma omp parallel for schedule(dynamic, 64)
        for (size_t cell_idx = 0; cell_idx < cell_list.size(); cell_idx++) {
            const auto& [center_key, center_particles] = cell_list[cell_idx];

            // Decode cell coordinates
            int cx = (center_key >> 32) & 0xFFFFFFFF;
            int cy = center_key & 0xFFFFFFFF;

            // Process pairs within the same cell
            for (size_t i = 0; i < center_particles.size(); i++) {
                for (size_t j = i + 1; j < center_particles.size(); j++) {
                    uint32_t idx1 = center_particles[i];
                    uint32_t idx2 = center_particles[j];

                    float dist2 = getDistanceSquared(particles[idx1], particles[idx2]);
                    if (dist2 <= radius2 && dist2 > 0.0001f) {
                        callback(idx1, idx2, std::sqrt(dist2));
                    }
                }
            }

            // Check neighboring cells
            for (int dy = -cell_radius; dy <= cell_radius; dy++) {
                for (int dx = -cell_radius; dx <= cell_radius; dx++) {
                    if (dx == 0 && dy == 0) continue;

                    // Only process half of neighbors to avoid duplicates
                    if (dx < 0 || (dx == 0 && dy < 0)) continue;

                    uint64_t neighbor_key = hashCell(cx + dx, cy + dy);

                    // Check if neighbor cell exists (sparse!)
                    auto neighbor_it = cells.find(neighbor_key);
                    if (neighbor_it == cells.end()) continue;

                    const auto& neighbor_particles = neighbor_it->second;

                    // Check all pairs between cells
                    for (uint32_t idx1 : center_particles) {
                        for (uint32_t idx2 : neighbor_particles) {
                            float dist2 = getDistanceSquared(particles[idx1], particles[idx2]);
                            if (dist2 <= radius2 && dist2 > 0.0001f) {
                                callback(idx1, idx2, std::sqrt(dist2));
                            }
                        }
                    }
                }
            }
        }
    }

    // Find neighbors of a specific particle
    std::vector<uint32_t> findNeighbors(const Particle& p, float radius,
                                        const std::vector<Particle>& particles) const {
        std::vector<uint32_t> neighbors;
        int cell_radius = std::ceil(radius / config.cell_size);
        float radius2 = radius * radius;

        auto [cx, cy] = getCellCoords(p.x, p.y);

        // Check all cells within radius
        for (int dy = -cell_radius; dy <= cell_radius; dy++) {
            for (int dx = -cell_radius; dx <= cell_radius; dx++) {
                uint64_t key = hashCell(cx + dx, cy + dy);

                // Check if cell exists (sparse!)
                auto it = cells.find(key);
                if (it == cells.end()) continue;

                for (uint32_t idx : it->second) {
                    float dist2 = getDistanceSquared(p, particles[idx]);
                    if (dist2 <= radius2 && dist2 > 0.0001f) {
                        neighbors.push_back(idx);
                    }
                }
            }
        }

        return neighbors;
    }

    // Get statistics
    struct Stats {
        size_t occupied_cells = 0;
        size_t total_particles = 0;
        size_t max_particles_per_cell = 0;
        float avg_particles_per_cell = 0;
        float occupancy_ratio = 0;  // occupied / total possible cells
    };

    Stats getStats() const {
        Stats stats;
        stats.occupied_cells = cells.size();

        size_t max_per_cell = 0;
        for (const auto& [key, particles] : cells) {
            stats.total_particles += particles.size();
            max_per_cell = std::max(max_per_cell, particles.size());
        }

        stats.max_particles_per_cell = max_per_cell;

        if (stats.occupied_cells > 0) {
            stats.avg_particles_per_cell = float(stats.total_particles) / stats.occupied_cells;
        }

        size_t total_possible = grid_resolution * grid_resolution;
        stats.occupancy_ratio = float(stats.occupied_cells) / total_possible;

        return stats;
    }

private:
    // Calculate distance squared with toroidal wrapping
    float getDistanceSquared(const Particle& p1, const Particle& p2) const {
        float dx = p1.x - p2.x;
        float dy = p1.y - p2.y;

        if (config.toroidal) {
            // Find shortest path through wraparound
            if (dx > config.world_size * 0.5f) dx -= config.world_size;
            if (dx < -config.world_size * 0.5f) dx += config.world_size;
            if (dy > config.world_size * 0.5f) dy -= config.world_size;
            if (dy < -config.world_size * 0.5f) dy += config.world_size;
        }

        return dx * dx + dy * dy;
    }
};

/**
 * Multi-Resolution Sparse Grid Manager
 *
 * Manages multiple sparse grids at different resolutions
 * as specified in SPATIAL_INDEXING_DESIGN.md
 */
template<typename Particle>
class SparseMultiResolutionGrid {
public:
    enum GridType {
        CONTACT = 0,    // 4 units - collision detection
        SPRING = 1,     // 20 units - spring formation
        THERMAL = 2,    // 100 units - heat transfer
        RADIATION = 3   // 500 units - radiation
    };

    struct Config {
        float world_size = 10000.0f;
        bool toroidal = true;

        // Cell sizes as per design doc
        float contact_cell_size = 4.0f;
        float spring_cell_size = 20.0f;
        float thermal_cell_size = 100.0f;
        float radiation_cell_size = 500.0f;
    };

private:
    Config config;
    std::vector<std::unique_ptr<SparseSpatialGrid<Particle>>> grids;

    // Interaction ranges for each grid type (must match cell sizes for efficiency)
    std::vector<float> ranges = {16.0f, 32.0f, 128.0f, 512.0f};

public:
    SparseMultiResolutionGrid(const Config& cfg = Config()) : config(cfg) {
        typename SparseSpatialGrid<Particle>::Config grid_config;
        grid_config.world_size = config.world_size;
        grid_config.toroidal = config.toroidal;

        // Create sparse grids with appropriate resolutions
        grid_config.cell_size = config.contact_cell_size;
        grids.push_back(std::make_unique<SparseSpatialGrid<Particle>>(grid_config));

        grid_config.cell_size = config.spring_cell_size;
        grids.push_back(std::make_unique<SparseSpatialGrid<Particle>>(grid_config));

        grid_config.cell_size = config.thermal_cell_size;
        grids.push_back(std::make_unique<SparseSpatialGrid<Particle>>(grid_config));

        grid_config.cell_size = config.radiation_cell_size;
        grids.push_back(std::make_unique<SparseSpatialGrid<Particle>>(grid_config));
    }

    // Update all grids - use incremental for better performance
    void update(const std::vector<Particle>& particles, bool incremental = true) {
        for (auto& grid : grids) {
            if (incremental) {
                grid->incrementalUpdate(particles);
            } else {
                grid->rebuild(particles);
            }
        }
    }

    // Process pairs for specific interaction type
    template<typename Callback>
    void processPairs(GridType type, const std::vector<Particle>& particles,
                     Callback callback) {
        grids[type]->processPairs(particles, ranges[type], callback);
    }

    // Get grid for direct access
    SparseSpatialGrid<Particle>& getGrid(GridType type) {
        return *grids[type];
    }

    // Get statistics for all grids
    struct MultiGridStats {
        typename SparseSpatialGrid<Particle>::Stats contact_stats;
        typename SparseSpatialGrid<Particle>::Stats spring_stats;
        typename SparseSpatialGrid<Particle>::Stats thermal_stats;
        typename SparseSpatialGrid<Particle>::Stats radiation_stats;
    };

    MultiGridStats getStats() const {
        MultiGridStats stats;
        stats.contact_stats = grids[CONTACT]->getStats();
        stats.spring_stats = grids[SPRING]->getStats();
        stats.thermal_stats = grids[THERMAL]->getStats();
        stats.radiation_stats = grids[RADIATION]->getStats();
        return stats;
    }
};

} // namespace digistar
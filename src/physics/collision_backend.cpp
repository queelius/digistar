/**
 * Collision Detection Backend Implementations
 */

#include "collision_backend.h"
#include <omp.h>
#include <atomic>
#include <cstring>

namespace digistar {

/**
 * Thread-Local Backend Implementation
 * Each thread accumulates forces locally, then merges
 */
template<typename Particle>
void ThreadLocalBackend<Particle>::computeCollisions(
    std::vector<Particle>& particles,
    SparseSpatialGrid<Particle>& grid,
    float dt) {

    auto start = this->startTimer();
    this->stats_.pairs_checked = 0;
    this->stats_.collisions_found = 0;

    size_t num_particles = particles.size();
    int num_threads = this->config_.num_threads > 0 ? this->config_.num_threads : omp_get_max_threads();

    // Thread-local storage for forces
    std::vector<std::vector<float>> fx_local(num_threads, std::vector<float>(num_particles, 0.0f));
    std::vector<std::vector<float>> fy_local(num_threads, std::vector<float>(num_particles, 0.0f));

    // Thread-local stats
    std::vector<size_t> pairs_checked_local(num_threads, 0);
    std::vector<size_t> collisions_found_local(num_threads, 0);

    // Get all occupied cells for parallel processing
    std::vector<std::pair<uint64_t, std::vector<uint32_t>>> cell_list;
    for (const auto& [key, cell_particles] : grid.cells) {
        if (!cell_particles.empty()) {
            cell_list.push_back({key, cell_particles});
        }
    }

    #pragma omp parallel num_threads(num_threads)
    {
        int tid = omp_get_thread_num();
        auto& fx_thread = fx_local[tid];
        auto& fy_thread = fy_local[tid];

        #pragma omp for schedule(dynamic, 16)
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

                    pairs_checked_local[tid]++;

                    float dx = particles[idx1].x - particles[idx2].x;
                    float dy = particles[idx1].y - particles[idx2].y;
                    float dist2 = dx * dx + dy * dy;

                    float min_dist = particles[idx1].radius + particles[idx2].radius;
                    float min_dist2 = min_dist * min_dist;

                    if (dist2 < min_dist2 && dist2 > 0.001f) {
                        collisions_found_local[tid]++;

                        float dist = std::sqrt(dist2);
                        float overlap = min_dist - dist;
                        dx /= dist;
                        dy /= dist;

                        // Hertzian contact force
                        float force = this->calculateContactForce(overlap);

                        // Relative velocity for damping
                        float vrel_x = particles[idx1].vx - particles[idx2].vx;
                        float vrel_y = particles[idx1].vy - particles[idx2].vy;
                        float vrel_normal = vrel_x * dx + vrel_y * dy;

                        // Add damping
                        force -= this->config_.damping_coefficient * vrel_normal * std::sqrt(overlap);

                        // Apply forces to thread-local storage
                        float fx = force * dx;
                        float fy = force * dy;

                        fx_thread[idx1] += fx;
                        fy_thread[idx1] += fy;
                        fx_thread[idx2] -= fx;
                        fy_thread[idx2] -= fy;
                    }
                }
            }

            // Check neighboring cells
            int cell_radius = std::ceil(this->config_.contact_radius / grid.config.cell_size);

            for (int dy = -cell_radius; dy <= cell_radius; dy++) {
                for (int dx = -cell_radius; dx <= cell_radius; dx++) {
                    if (dx == 0 && dy == 0) continue;

                    // Only process half of neighbors to avoid duplicates
                    if (dx < 0 || (dx == 0 && dy < 0)) continue;

                    uint64_t neighbor_key = grid.hashCell(cx + dx, cy + dy);

                    // Find neighbor in our cell list (since we can't access grid directly)
                    const std::vector<uint32_t>* neighbor_particles = nullptr;
                    for (const auto& [key, particles] : cell_list) {
                        if (key == neighbor_key) {
                            neighbor_particles = &particles;
                            break;
                        }
                    }

                    if (!neighbor_particles) continue;

                    // Check all pairs between cells
                    for (uint32_t idx1 : center_particles) {
                        for (uint32_t idx2 : *neighbor_particles) {
                            pairs_checked_local[tid]++;

                            float dx = particles[idx1].x - particles[idx2].x;
                            float dy = particles[idx1].y - particles[idx2].y;

                            // Handle toroidal wrapping
                            if (grid.config.toroidal) {
                                if (dx > grid.config.world_size * 0.5f) dx -= grid.config.world_size;
                                if (dx < -grid.config.world_size * 0.5f) dx += grid.config.world_size;
                                if (dy > grid.config.world_size * 0.5f) dy -= grid.config.world_size;
                                if (dy < -grid.config.world_size * 0.5f) dy += grid.config.world_size;
                            }

                            float dist2 = dx * dx + dy * dy;
                            float min_dist = particles[idx1].radius + particles[idx2].radius;
                            float min_dist2 = min_dist * min_dist;

                            if (dist2 < min_dist2 && dist2 > 0.001f) {
                                collisions_found_local[tid]++;

                                float dist = std::sqrt(dist2);
                                float overlap = min_dist - dist;
                                dx /= dist;
                                dy /= dist;

                                float force = this->calculateContactForce(overlap);

                                float vrel_x = particles[idx1].vx - particles[idx2].vx;
                                float vrel_y = particles[idx1].vy - particles[idx2].vy;
                                float vrel_normal = vrel_x * dx + vrel_y * dy;

                                force -= this->config_.damping_coefficient * vrel_normal * std::sqrt(overlap);

                                float fx = force * dx;
                                float fy = force * dy;

                                fx_thread[idx1] += fx;
                                fy_thread[idx1] += fy;
                                fx_thread[idx2] -= fx;
                                fy_thread[idx2] -= fy;
                            }
                        }
                    }
                }
            }
        }
    }

    // Merge thread-local forces back to particles
    #pragma omp parallel for
    for (size_t i = 0; i < num_particles; i++) {
        float fx_sum = 0.0f;
        float fy_sum = 0.0f;

        for (int tid = 0; tid < num_threads; tid++) {
            fx_sum += fx_local[tid][i];
            fy_sum += fy_local[tid][i];
        }

        particles[i].fx += fx_sum;
        particles[i].fy += fy_sum;
    }

    // Merge stats
    for (int tid = 0; tid < num_threads; tid++) {
        this->stats_.pairs_checked += pairs_checked_local[tid];
        this->stats_.collisions_found += collisions_found_local[tid];
    }

    this->endTimer(start);

    if (this->stats_.pairs_checked > 0) {
        this->stats_.efficiency = double(this->stats_.collisions_found) /
                                  double(this->stats_.pairs_checked);
    }

    // Calculate memory usage
    this->stats_.memory_bytes_used =
        num_threads * num_particles * 2 * sizeof(float) +  // fx_local, fy_local
        num_threads * 2 * sizeof(size_t);                   // stats
}

/**
 * Tiled Backend Implementation
 * Divides space into tiles, one thread per tile
 */
template<typename Particle>
void TiledBackend<Particle>::computeCollisions(
    std::vector<Particle>& particles,
    SparseSpatialGrid<Particle>& grid,
    float dt) {

    auto start = this->startTimer();
    this->stats_.pairs_checked = 0;
    this->stats_.collisions_found = 0;

    int num_threads = this->config_.num_threads > 0 ? this->config_.num_threads : omp_get_max_threads();

    // Divide world into tiles
    int tiles_per_dim = std::ceil(std::sqrt(num_threads));
    float tile_size = grid.config.world_size / tiles_per_dim;

    // Assign cells to tiles
    std::vector<std::vector<std::pair<uint64_t, std::vector<uint32_t>>>> tiles(num_threads);

    for (const auto& [key, cell_particles] : grid.cells) {
        if (cell_particles.empty()) continue;

        // Decode cell coordinates
        int cx = (key >> 32) & 0xFFFFFFFF;
        int cy = key & 0xFFFFFFFF;

        // Convert to world coordinates
        float world_x = (cx * grid.config.cell_size) - grid.config.world_size * 0.5f;
        float world_y = (cy * grid.config.cell_size) - grid.config.world_size * 0.5f;

        // Determine tile
        int tile_x = (world_x + grid.config.world_size * 0.5f) / tile_size;
        int tile_y = (world_y + grid.config.world_size * 0.5f) / tile_size;

        tile_x = std::max(0, std::min(tiles_per_dim - 1, tile_x));
        tile_y = std::max(0, std::min(tiles_per_dim - 1, tile_y));

        int tile_id = tile_y * tiles_per_dim + tile_x;
        if (tile_id < num_threads) {
            tiles[tile_id].push_back({key, cell_particles});
        }
    }

    // Thread-local stats
    std::vector<size_t> pairs_checked_local(num_threads, 0);
    std::vector<size_t> collisions_found_local(num_threads, 0);

    #pragma omp parallel num_threads(num_threads)
    {
        int tid = omp_get_thread_num();

        // Process cells in this tile
        for (const auto& [center_key, center_particles] : tiles[tid]) {

            // Process pairs within the same cell
            for (size_t i = 0; i < center_particles.size(); i++) {
                for (size_t j = i + 1; j < center_particles.size(); j++) {
                    uint32_t idx1 = center_particles[i];
                    uint32_t idx2 = center_particles[j];

                    pairs_checked_local[tid]++;

                    float dx = particles[idx1].x - particles[idx2].x;
                    float dy = particles[idx1].y - particles[idx2].y;
                    float dist2 = dx * dx + dy * dy;

                    float min_dist = particles[idx1].radius + particles[idx2].radius;
                    float min_dist2 = min_dist * min_dist;

                    if (dist2 < min_dist2 && dist2 > 0.001f) {
                        collisions_found_local[tid]++;

                        float dist = std::sqrt(dist2);
                        float overlap = min_dist - dist;
                        dx /= dist;
                        dy /= dist;

                        float force = this->calculateContactForce(overlap);

                        float vrel_x = particles[idx1].vx - particles[idx2].vx;
                        float vrel_y = particles[idx1].vy - particles[idx2].vy;
                        float vrel_normal = vrel_x * dx + vrel_y * dy;

                        force -= this->config_.damping_coefficient * vrel_normal * std::sqrt(overlap);

                        float fx = force * dx;
                        float fy = force * dy;

                        // Use atomics for simplicity in tiled approach
                        #pragma omp atomic
                        particles[idx1].fx += fx;
                        #pragma omp atomic
                        particles[idx1].fy += fy;
                        #pragma omp atomic
                        particles[idx2].fx -= fx;
                        #pragma omp atomic
                        particles[idx2].fy -= fy;
                    }
                }
            }

            // Check with neighbor cells in same tile
            int cx = (center_key >> 32) & 0xFFFFFFFF;
            int cy = center_key & 0xFFFFFFFF;
            int cell_radius = std::ceil(this->config_.contact_radius / grid.config.cell_size);

            for (int dy = -cell_radius; dy <= cell_radius; dy++) {
                for (int dx = -cell_radius; dx <= cell_radius; dx++) {
                    if (dx == 0 && dy == 0) continue;
                    if (dx < 0 || (dx == 0 && dy < 0)) continue;

                    uint64_t neighbor_key = grid.hashCell(cx + dx, cy + dy);

                    // Find neighbor in our tile
                    const std::vector<uint32_t>* neighbor_particles = nullptr;
                    for (const auto& [key, particles] : tiles[tid]) {
                        if (key == neighbor_key) {
                            neighbor_particles = &particles;
                            break;
                        }
                    }

                    if (!neighbor_particles) continue;

                    // Check all pairs between cells
                    for (uint32_t idx1 : center_particles) {
                        for (uint32_t idx2 : *neighbor_particles) {
                            pairs_checked_local[tid]++;

                            float dx = particles[idx1].x - particles[idx2].x;
                            float dy = particles[idx1].y - particles[idx2].y;

                            if (grid.config.toroidal) {
                                if (dx > grid.config.world_size * 0.5f) dx -= grid.config.world_size;
                                if (dx < -grid.config.world_size * 0.5f) dx += grid.config.world_size;
                                if (dy > grid.config.world_size * 0.5f) dy -= grid.config.world_size;
                                if (dy < -grid.config.world_size * 0.5f) dy += grid.config.world_size;
                            }

                            float dist2 = dx * dx + dy * dy;
                            float min_dist = particles[idx1].radius + particles[idx2].radius;
                            float min_dist2 = min_dist * min_dist;

                            if (dist2 < min_dist2 && dist2 > 0.001f) {
                                collisions_found_local[tid]++;

                                float dist = std::sqrt(dist2);
                                float overlap = min_dist - dist;
                                dx /= dist;
                                dy /= dist;

                                float force = this->calculateContactForce(overlap);

                                float vrel_x = particles[idx1].vx - particles[idx2].vx;
                                float vrel_y = particles[idx1].vy - particles[idx2].vy;
                                float vrel_normal = vrel_x * dx + vrel_y * dy;

                                force -= this->config_.damping_coefficient * vrel_normal * std::sqrt(overlap);

                                float fx = force * dx;
                                float fy = force * dy;

                                #pragma omp atomic
                                particles[idx1].fx += fx;
                                #pragma omp atomic
                                particles[idx1].fy += fy;
                                #pragma omp atomic
                                particles[idx2].fx -= fx;
                                #pragma omp atomic
                                particles[idx2].fy -= fy;
                            }
                        }
                    }
                }
            }
        }
    }

    // Merge stats
    for (int tid = 0; tid < num_threads; tid++) {
        this->stats_.pairs_checked += pairs_checked_local[tid];
        this->stats_.collisions_found += collisions_found_local[tid];
    }

    this->endTimer(start);

    if (this->stats_.pairs_checked > 0) {
        this->stats_.efficiency = double(this->stats_.collisions_found) /
                                  double(this->stats_.pairs_checked);
    }

    this->stats_.memory_bytes_used = tiles_per_dim * tiles_per_dim * sizeof(void*);
}

/**
 * Atomic Backend Implementation (for comparison)
 * Uses OpenMP parallel with atomic operations
 */
template<typename Particle>
void AtomicBackend<Particle>::computeCollisions(
    std::vector<Particle>& particles,
    SparseSpatialGrid<Particle>& grid,
    float dt) {

    auto start = this->startTimer();

    std::atomic<size_t> pairs_checked(0);
    std::atomic<size_t> collisions_found(0);

    int num_threads = this->config_.num_threads > 0 ? this->config_.num_threads : omp_get_max_threads();

    // Get all occupied cells for parallel processing
    std::vector<std::pair<uint64_t, std::vector<uint32_t>>> cell_list;
    for (const auto& [key, cell_particles] : grid.cells) {
        if (!cell_particles.empty()) {
            cell_list.push_back({key, cell_particles});
        }
    }

    #pragma omp parallel for num_threads(num_threads) schedule(dynamic, 32)
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

                pairs_checked++;

                float dx = particles[idx1].x - particles[idx2].x;
                float dy = particles[idx1].y - particles[idx2].y;
                float dist2 = dx * dx + dy * dy;

                float min_dist = particles[idx1].radius + particles[idx2].radius;
                float min_dist2 = min_dist * min_dist;

                if (dist2 < min_dist2 && dist2 > 0.001f) {
                    collisions_found++;

                    float dist = std::sqrt(dist2);
                    float overlap = min_dist - dist;
                    dx /= dist;
                    dy /= dist;

                    float force = this->calculateContactForce(overlap);

                    float vrel_x = particles[idx1].vx - particles[idx2].vx;
                    float vrel_y = particles[idx1].vy - particles[idx2].vy;
                    float vrel_normal = vrel_x * dx + vrel_y * dy;

                    force -= this->config_.damping_coefficient * vrel_normal * std::sqrt(overlap);

                    float fx = force * dx;
                    float fy = force * dy;

                    #pragma omp atomic
                    particles[idx1].fx += fx;
                    #pragma omp atomic
                    particles[idx1].fy += fy;
                    #pragma omp atomic
                    particles[idx2].fx -= fx;
                    #pragma omp atomic
                    particles[idx2].fy -= fy;
                }
            }
        }

        // Check neighboring cells
        int cell_radius = std::ceil(this->config_.contact_radius / grid.config.cell_size);

        for (int dy = -cell_radius; dy <= cell_radius; dy++) {
            for (int dx = -cell_radius; dx <= cell_radius; dx++) {
                if (dx == 0 && dy == 0) continue;
                if (dx < 0 || (dx == 0 && dy < 0)) continue;

                uint64_t neighbor_key = grid.hashCell(cx + dx, cy + dy);

                auto neighbor_it = grid.cells.find(neighbor_key);
                if (neighbor_it == grid.cells.end()) continue;

                const auto& neighbor_particles = neighbor_it->second;

                // Check all pairs between cells
                for (uint32_t idx1 : center_particles) {
                    for (uint32_t idx2 : neighbor_particles) {
                        pairs_checked++;

                        float dx = particles[idx1].x - particles[idx2].x;
                        float dy = particles[idx1].y - particles[idx2].y;

                        if (grid.config.toroidal) {
                            if (dx > grid.config.world_size * 0.5f) dx -= grid.config.world_size;
                            if (dx < -grid.config.world_size * 0.5f) dx += grid.config.world_size;
                            if (dy > grid.config.world_size * 0.5f) dy -= grid.config.world_size;
                            if (dy < -grid.config.world_size * 0.5f) dy += grid.config.world_size;
                        }

                        float dist2 = dx * dx + dy * dy;
                        float min_dist = particles[idx1].radius + particles[idx2].radius;
                        float min_dist2 = min_dist * min_dist;

                        if (dist2 < min_dist2 && dist2 > 0.001f) {
                            collisions_found++;

                            float dist = std::sqrt(dist2);
                            float overlap = min_dist - dist;
                            dx /= dist;
                            dy /= dist;

                            float force = this->calculateContactForce(overlap);

                            float vrel_x = particles[idx1].vx - particles[idx2].vx;
                            float vrel_y = particles[idx1].vy - particles[idx2].vy;
                            float vrel_normal = vrel_x * dx + vrel_y * dy;

                            force -= this->config_.damping_coefficient * vrel_normal * std::sqrt(overlap);

                            float fx = force * dx;
                            float fy = force * dy;

                            #pragma omp atomic
                            particles[idx1].fx += fx;
                            #pragma omp atomic
                            particles[idx1].fy += fy;
                            #pragma omp atomic
                            particles[idx2].fx -= fx;
                            #pragma omp atomic
                            particles[idx2].fy -= fy;
                        }
                    }
                }
            }
        }
    }

    this->stats_.pairs_checked = pairs_checked;
    this->stats_.collisions_found = collisions_found;

    this->endTimer(start);

    if (this->stats_.pairs_checked > 0) {
        this->stats_.efficiency = double(this->stats_.collisions_found) /
                                  double(this->stats_.pairs_checked);
    }
}

// Explicit template instantiations
struct TestParticle {
    float x, y, vx, vy, ax, ay, fx, fy, mass, radius;
};

template class ThreadLocalBackend<TestParticle>;
template class TiledBackend<TestParticle>;
template class AtomicBackend<TestParticle>;

} // namespace digistar

// Instantiation for the benchmark Particle (outside namespace)
struct Particle {
    float x, y, vx, vy, ax, ay, fx, fy, mass, radius;
};

namespace digistar {
template class ThreadLocalBackend<::Particle>;
template class TiledBackend<::Particle>;
template class AtomicBackend<::Particle>;
}
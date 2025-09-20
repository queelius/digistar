/**
 * CPU Material-Based Virtual Spring Backend
 *
 * Enhanced virtual spring network that uses material properties to create
 * diverse emergent structures. Combines contact forces for all particles
 * with material-specific spring formation for rich physics.
 *
 * Key Features:
 * - Dual force system (contact + springs)
 * - Material compatibility matrix
 * - Directional bonding patterns
 * - Temperature-dependent mechanics
 * - Phase transitions
 * - Optimized for millions of particles
 */

#pragma once

#include <vector>
#include <memory>
#include <algorithm>
#include <cmath>
#include <omp.h>
#include "virtual_spring_network_backend.h"
#include "sparse_spatial_grid.h"
#include "material_system.h"

namespace digistar {

/**
 * Extended particle structure with material properties
 */
template<typename BaseParticle>
struct MaterialParticle : public BaseParticle {
    // Material properties (8 bytes total)
    ParticleMaterial material;
    uint8_t bond_count;     // Current number of bonds
    uint8_t cluster_id;     // Contact-based cluster
    uint16_t flags;         // Status flags

    // Thermal properties
    float internal_energy;  // For temperature calculation
};

/**
 * Enhanced spring with material-aware properties
 */
struct MaterialSpring : public Spring {
    float base_stiffness;   // Unmodified stiffness
    float base_damping;     // Unmodified damping
    uint8_t mat_type1;      // Material type of p1
    uint8_t mat_type2;      // Material type of p2
    float formation_temp;   // Temperature when formed
};

/**
 * CPU Material Spring Backend
 */
template<typename Particle>
class CpuMaterialSpringBackend : public IVirtualSpringNetworkBackend<MaterialParticle<Particle>> {
public:
    using MParticle = MaterialParticle<Particle>;
    using Base = IVirtualSpringNetworkBackend<MParticle>;
    using typename Base::Config;

    struct MaterialConfig : public Config {
        // Material-specific parameters
        bool enable_directional_bonding = true;
        bool enable_temperature_effects = true;
        bool enable_phase_transitions = true;
        float directional_threshold = 0.3f;
        float thermal_diffusion_rate = 0.01f;
        float ambient_temperature = 293.0f;  // Room temperature

        // Contact force parameters
        float contact_radius = 4.0f;
        float contact_stiffness = 500.0f;
        float contact_damping = 0.2f;
    };

    CpuMaterialSpringBackend(const MaterialConfig& config = MaterialConfig())
        : config_(config), bonding_matrix_(), spring_ctx_() {

        Base::config_ = config;  // Copy base config

        if (config_.num_threads <= 0) {
            config_.num_threads = omp_get_max_threads();
        }

        // Initialize spring formation context
        spring_ctx_.max_distance = config.formation_distance;
        spring_ctx_.max_velocity = config.formation_velocity;
        spring_ctx_.directional_threshold = config.directional_threshold;

        springs_.reserve(config.max_total_springs);
        contact_union_find_ = std::make_unique<UnionFind>(0);
    }

    void updateVirtualSprings(
        std::vector<MParticle>& particles,
        SparseSpatialGrid<MParticle>& grid,
        float dt) override {

        auto frame_start = std::chrono::high_resolution_clock::now();

        // 0. Initialize or resize data structures
        ensureCapacity(particles.size());

        // 1. Apply contact forces to ALL particles (fluid behavior)
        auto t1 = startTimer();
        computeContactForces(particles, grid, dt);
        contact_force_ms_ = elapsedMs(t1);

        // 2. Update thermal properties and phase transitions
        if (config_.enable_temperature_effects) {
            t1 = startTimer();
            updateThermalProperties(particles, dt);
            thermal_update_ms_ = elapsedMs(t1);
        }

        // 3. Form new material-specific springs
        t1 = startTimer();
        formMaterialSprings(particles, grid);
        Base::stats_.spring_update_ms = elapsedMs(t1);

        // 4. Compute spring forces (material-aware)
        t1 = startTimer();
        computeMaterialSpringForces(particles, dt);
        Base::stats_.force_calc_ms = elapsedMs(t1);

        // 5. Break overstressed or incompatible springs
        t1 = startTimer();
        breakMaterialSprings(particles);
        Base::stats_.spring_update_ms += elapsedMs(t1);

        // 6. Update clusters (both contact and spring-based)
        t1 = startTimer();
        updateClusters(particles);
        Base::stats_.composite_calc_ms = elapsedMs(t1);

        // 7. Update material groups for next frame
        t1 = startTimer();
        if (frame_count_ % 10 == 0) {  // Update every 10 frames
            updateMaterialGroups(particles);
        }
        material_group_ms_ = elapsedMs(t1);

        // Update statistics
        Base::stats_.total_time_ms = elapsedMs(frame_start);
        Base::stats_.active_springs = countActiveSprings();
        frame_count_++;
    }

    const std::vector<Spring>& getSprings() const override {
        // Convert MaterialSpring to Spring for interface compatibility
        base_springs_.clear();
        for (const auto& ms : springs_) {
            base_springs_.push_back(static_cast<Spring>(ms));
        }
        return base_springs_;
    }

    std::string getName() const override {
        return "CpuMaterialSpringBackend";
    }

    // Material-specific methods
    void setMaterialType(uint32_t particle_idx, uint8_t material_type,
                         std::vector<MParticle>& particles) {
        if (particle_idx < particles.size()) {
            auto& p = particles[particle_idx];
            p.material.type = material_type;

            // Set default bonding pattern for material
            auto props = getMaterialProperties(material_type);
            p.material.pattern = props.bonding_pattern;

            // Break existing springs if material becomes incompatible
            for (auto& spring : springs_) {
                if (!spring.active) continue;
                if (spring.p1 == particle_idx || spring.p2 == particle_idx) {
                    auto mat1 = particles[spring.p1].material.type;
                    auto mat2 = particles[spring.p2].material.type;
                    auto rule = bonding_matrix_.getRule(mat1, mat2);
                    if (!rule.can_bond) {
                        spring.active = false;
                    }
                }
            }
        }
    }

    float getParticleTemperature(uint32_t idx, const std::vector<MParticle>& particles) const {
        if (idx < particles.size()) {
            return particles[idx].material.temperature;
        }
        return config_.ambient_temperature;
    }

private:
    MaterialConfig config_;
    BondingMatrix bonding_matrix_;
    MaterialGroupManager material_groups_;
    SpringFormationContext spring_ctx_;

    std::vector<MaterialSpring> springs_;
    mutable std::vector<Spring> base_springs_;  // For interface compatibility

    std::unique_ptr<UnionFind> contact_union_find_;  // Contact-based clustering
    std::unique_ptr<UnionFind> spring_union_find_;   // Spring-based clustering

    std::vector<std::unordered_set<uint32_t>> particle_springs_;
    std::vector<uint8_t> particle_bond_counts_;

    // Performance metrics
    double contact_force_ms_ = 0;
    double thermal_update_ms_ = 0;
    double material_group_ms_ = 0;
    uint64_t frame_count_ = 0;

    // === Contact Forces (Always Active) ===
    void computeContactForces(std::vector<MParticle>& particles,
                              SparseSpatialGrid<MParticle>& grid, float dt) {

        int cell_radius = std::ceil(config_.contact_radius / grid.config.cell_size);
        float radius2 = config_.contact_radius * config_.contact_radius;

        // Reset contact clustering
        contact_union_find_->reset(particles.size());

        // Convert to vector for parallel processing
        std::vector<std::pair<uint64_t, std::vector<uint32_t>>> cell_list;
        cell_list.reserve(grid.cells.size());
        for (const auto& [key, indices] : grid.cells) {
            cell_list.push_back({key, indices});
        }

        #pragma omp parallel for schedule(dynamic, 32) num_threads(config_.num_threads)
        for (size_t cell_idx = 0; cell_idx < cell_list.size(); cell_idx++) {
            const auto& [center_key, center_particles] = cell_list[cell_idx];

            int cx = (center_key >> 32) & 0xFFFFFFFF;
            int cy = center_key & 0xFFFFFFFF;

            // Process all neighboring cells for contact
            for (int dy = -cell_radius; dy <= cell_radius; dy++) {
                for (int dx = -cell_radius; dx <= cell_radius; dx++) {
                    uint64_t neighbor_key = grid.hashCell(cx + dx, cy + dy);
                    auto neighbor_it = grid.cells.find(neighbor_key);
                    if (neighbor_it == grid.cells.end()) continue;

                    for (uint32_t i : center_particles) {
                        for (uint32_t j : neighbor_it->second) {
                            if (i >= j) continue;

                            auto& p1 = particles[i];
                            auto& p2 = particles[j];

                            float dx = p2.x - p1.x;
                            float dy = p2.y - p1.y;

                            // Handle toroidal wrapping
                            if (grid.config.toroidal) {
                                float half_world = grid.config.world_size * 0.5f;
                                if (dx > half_world) dx -= grid.config.world_size;
                                if (dx < -half_world) dx += grid.config.world_size;
                                if (dy > half_world) dy -= grid.config.world_size;
                                if (dy < -half_world) dy += grid.config.world_size;
                            }

                            float dist2 = dx*dx + dy*dy;
                            if (dist2 > radius2 || dist2 < 0.01f) continue;

                            float dist = sqrtf(dist2);
                            float overlap = (p1.radius + p2.radius) - dist;

                            if (overlap > 0) {
                                // Hertzian contact force
                                float force_mag = config_.contact_stiffness * powf(overlap, 1.5f);

                                // Damping
                                float dvx = p2.vx - p1.vx;
                                float dvy = p2.vy - p1.vy;
                                float vel_along = (dvx * dx + dvy * dy) / dist;
                                force_mag += config_.contact_damping * vel_along;

                                // Apply forces
                                float fx = force_mag * dx / dist;
                                float fy = force_mag * dy / dist;

                                #pragma omp atomic
                                p1.fx += fx;
                                #pragma omp atomic
                                p1.fy += fy;
                                #pragma omp atomic
                                p2.fx -= fx;
                                #pragma omp atomic
                                p2.fy -= fy;

                                // Mark as in contact (for clustering)
                                #pragma omp critical
                                {
                                    contact_union_find_->unite(i, j);
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    // === Material Spring Formation ===
    void formMaterialSprings(const std::vector<MParticle>& particles,
                             SparseSpatialGrid<MParticle>& grid) {

        // Use material groups for efficient compatible pair checking
        material_groups_.forEachCompatiblePair(bonding_matrix_,
            [&](const MaterialGroup& group1, const MaterialGroup& group2,
                const BondingRule& rule) {

                // Only process nearby particles in compatible groups
                processCompatibleMaterialGroup(particles, grid, group1, group2, rule);
            });
    }

    void processCompatibleMaterialGroup(const std::vector<MParticle>& particles,
                                       SparseSpatialGrid<MParticle>& grid,
                                       const MaterialGroup& group1,
                                       const MaterialGroup& group2,
                                       const BondingRule& rule) {

        float max_dist2 = spring_ctx_.max_distance * spring_ctx_.max_distance;

        // Thread-local spring buffers
        std::vector<std::vector<MaterialSpring>> thread_springs(config_.num_threads);

        #pragma omp parallel num_threads(config_.num_threads)
        {
            int tid = omp_get_thread_num();
            auto& local_springs = thread_springs[tid];

            #pragma omp for schedule(dynamic, 64)
            for (size_t idx1 = 0; idx1 < group1.particle_indices.size(); idx1++) {
                uint32_t i = group1.particle_indices[idx1];
                const auto& p1 = particles[i];

                // Check bond saturation
                if (particle_bond_counts_[i] >= getMaterialProperties(p1.material.type).max_bonds)
                    continue;

                // Find nearby particles from group2
                auto neighbors = grid.getNeighbors(p1.x, p1.y, spring_ctx_.max_distance);

                for (uint32_t j : neighbors) {
                    if (i >= j) continue;  // Avoid duplicates

                    // Check if j is in group2
                    bool in_group2 = (group1.material_type == group2.material_type) ||
                                    std::binary_search(group2.particle_indices.begin(),
                                                      group2.particle_indices.end(), j);
                    if (!in_group2) continue;

                    const auto& p2 = particles[j];

                    // Check bond saturation for p2
                    if (particle_bond_counts_[j] >= getMaterialProperties(p2.material.type).max_bonds)
                        continue;

                    // Check if spring already exists
                    bool exists = false;
                    for (const auto& spring_idx : particle_springs_[i]) {
                        if (springs_[spring_idx].p2 == j || springs_[spring_idx].p1 == j) {
                            exists = true;
                            break;
                        }
                    }
                    if (exists) continue;

                    // Calculate distance and velocity
                    float dx = p2.x - p1.x;
                    float dy = p2.y - p1.y;
                    float dist2 = dx*dx + dy*dy;

                    if (dist2 > max_dist2 || dist2 < 0.01f) continue;

                    float dvx = p2.vx - p1.vx;
                    float dvy = p2.vy - p1.vy;
                    float rel_vel = sqrtf(dvx*dvx + dvy*dvy);

                    // Comprehensive spring formation check
                    if (shouldFormSpring(sqrtf(dist2), rel_vel,
                                       p1.material, p2.material,
                                       bonding_matrix_, spring_ctx_,
                                       dx, dy,
                                       particle_bond_counts_[i],
                                       particle_bond_counts_[j])) {

                        // Create material spring
                        MaterialSpring spring;
                        spring.p1 = i;
                        spring.p2 = j;
                        spring.rest_length = sqrtf(dist2);
                        spring.base_stiffness = Base::config_.spring_stiffness * (1.0f + rule.strength * 0.1f);
                        spring.base_damping = Base::config_.spring_damping;
                        spring.stiffness = spring.base_stiffness;
                        spring.damping = spring.base_damping;
                        spring.max_force = Base::config_.max_force * (1.0f + rule.strength * 0.05f);
                        spring.current_force = 0;
                        spring.active = true;
                        spring.mat_type1 = p1.material.type;
                        spring.mat_type2 = p2.material.type;
                        spring.formation_temp = (p1.material.temperature + p2.material.temperature) * 0.5f;

                        local_springs.push_back(spring);
                    }
                }
            }
        }

        // Merge thread-local springs
        for (const auto& local_springs : thread_springs) {
            for (const auto& spring : local_springs) {
                if (springs_.size() < Base::config_.max_total_springs) {
                    uint32_t spring_idx = springs_.size();
                    springs_.push_back(spring);
                    particle_springs_[spring.p1].insert(spring_idx);
                    particle_springs_[spring.p2].insert(spring_idx);
                    particle_bond_counts_[spring.p1]++;
                    particle_bond_counts_[spring.p2]++;
                    Base::stats_.springs_formed++;
                }
            }
        }
    }

    // === Material Spring Forces ===
    void computeMaterialSpringForces(std::vector<MParticle>& particles, float dt) {

        #pragma omp parallel for schedule(static) num_threads(config_.num_threads)
        for (size_t i = 0; i < springs_.size(); i++) {
            auto& spring = springs_[i];
            if (!spring.active) continue;

            const auto& p1 = particles[spring.p1];
            const auto& p2 = particles[spring.p2];

            // Update spring properties based on temperature
            if (config_.enable_temperature_effects) {
                float avg_temp = (p1.material.temperature + p2.material.temperature) * 0.5f;
                auto props1 = getMaterialProperties(spring.mat_type1);
                auto props2 = getMaterialProperties(spring.mat_type2);
                float avg_melt = (props1.melting_point + props2.melting_point) * 0.5f;
                float avg_boil = (props1.boiling_point + props2.boiling_point) * 0.5f;

                spring.stiffness = spring.base_stiffness *
                    getTemperatureModifiedStrength(1.0f, avg_temp, avg_melt, avg_boil);
            }

            // Calculate spring force
            float dx = p2.x - p1.x;
            float dy = p2.y - p1.y;

            // Handle toroidal wrapping
            const float world_size = 10000.0f;  // TODO: Get from grid
            const float half_world = world_size * 0.5f;
            if (dx > half_world) dx -= world_size;
            if (dx < -half_world) dx += world_size;
            if (dy > half_world) dy -= world_size;
            if (dy < -half_world) dy += world_size;

            float dist = sqrtf(dx*dx + dy*dy);
            if (dist < 0.001f) continue;

            float stretch = dist - spring.rest_length;
            float force_magnitude = spring.stiffness * stretch;

            // Damping
            float dvx = p2.vx - p1.vx;
            float dvy = p2.vy - p1.vy;
            float vel_along = (dvx * dx + dvy * dy) / dist;
            force_magnitude += spring.damping * vel_along;

            spring.current_force = fabsf(force_magnitude);

            // Apply forces
            float fx = force_magnitude * dx / dist;
            float fy = force_magnitude * dy / dist;

            #pragma omp atomic
            particles[spring.p1].fx += fx;
            #pragma omp atomic
            particles[spring.p1].fy += fy;
            #pragma omp atomic
            particles[spring.p2].fx -= fx;
            #pragma omp atomic
            particles[spring.p2].fy -= fy;
        }
    }

    // === Spring Breaking ===
    void breakMaterialSprings(const std::vector<MParticle>& particles) {

        size_t springs_broken = 0;

        #pragma omp parallel for reduction(+:springs_broken) num_threads(config_.num_threads)
        for (size_t i = 0; i < springs_.size(); i++) {
            auto& spring = springs_[i];
            if (!spring.active) continue;

            const auto& p1 = particles[spring.p1];
            const auto& p2 = particles[spring.p2];

            bool should_break = false;

            // Check material compatibility (may have changed)
            auto rule = bonding_matrix_.getRule(p1.material.type, p2.material.type);
            if (!rule.can_bond) {
                should_break = true;
            }

            // Force threshold
            if (spring.current_force > spring.max_force) {
                should_break = true;
            }

            // Stretch threshold
            float dx = p2.x - p1.x;
            float dy = p2.y - p1.y;
            float dist = sqrtf(dx*dx + dy*dy);
            if (dist > spring.rest_length * Base::config_.max_stretch) {
                should_break = true;
            }

            // Temperature breaking (bonds melt)
            if (config_.enable_temperature_effects) {
                float avg_temp = (p1.material.temperature + p2.material.temperature) * 0.5f;
                auto props1 = getMaterialProperties(spring.mat_type1);
                auto props2 = getMaterialProperties(spring.mat_type2);
                float avg_melt = (props1.melting_point + props2.melting_point) * 0.5f;
                if (avg_temp > avg_melt * 1.1f) {
                    should_break = true;
                }
            }

            if (should_break) {
                spring.active = false;
                springs_broken++;
            }
        }

        Base::stats_.springs_broken = springs_broken;

        // Update bond counts and particle_springs
        for (size_t i = 0; i < springs_.size(); i++) {
            if (!springs_[i].active) {
                particle_springs_[springs_[i].p1].erase(i);
                particle_springs_[springs_[i].p2].erase(i);
                particle_bond_counts_[springs_[i].p1]--;
                particle_bond_counts_[springs_[i].p2]--;
            }
        }
    }

    // === Thermal Updates ===
    void updateThermalProperties(std::vector<MParticle>& particles, float dt) {

        #pragma omp parallel for num_threads(config_.num_threads)
        for (size_t i = 0; i < particles.size(); i++) {
            auto& p = particles[i];

            // Calculate temperature from kinetic energy
            float ke = 0.5f * p.mass * (p.vx*p.vx + p.vy*p.vy);
            float temp_from_ke = ke / (1.38e-23f * p.mass);  // Simplified

            // Blend with current temperature
            float new_temp = 0.9f * p.material.temperature + 0.1f * temp_from_ke;

            // Thermal diffusion to neighbors
            float temp_sum = 0;
            int neighbor_count = 0;
            for (uint32_t spring_idx : particle_springs_[i]) {
                const auto& spring = springs_[spring_idx];
                if (!spring.active) continue;

                uint32_t other = (spring.p1 == i) ? spring.p2 : spring.p1;
                temp_sum += particles[other].material.temperature;
                neighbor_count++;
            }

            if (neighbor_count > 0) {
                float avg_neighbor_temp = temp_sum / neighbor_count;
                new_temp += config_.thermal_diffusion_rate * (avg_neighbor_temp - new_temp);
            }

            // Apply ambient temperature influence
            new_temp += 0.01f * (config_.ambient_temperature - new_temp);

            p.material.temperature = new_temp;

            // Check phase transitions
            if (config_.enable_phase_transitions) {
                checkPhaseTransition(p.material, new_temp);
            }
        }
    }

    // === Clustering ===
    void updateClusters(const std::vector<MParticle>& particles) {
        // Update spring-based clusters
        spring_union_find_->reset(particles.size());
        for (const auto& spring : springs_) {
            if (spring.active) {
                spring_union_find_->unite(spring.p1, spring.p2);
            }
        }

        // Update composite bodies based on spring clusters
        Base::composites_.clear();
        auto spring_components = spring_union_find_->getComponents();

        for (const auto& [root, indices] : spring_components) {
            if (indices.size() < Base::config_.min_composite_size) continue;

            CompositeBody composite;
            composite.id = Base::composites_.size();
            composite.particle_indices = indices;

            // Calculate properties...
            // (Similar to base implementation)

            Base::composites_.push_back(composite);
        }

        Base::stats_.num_composites = Base::composites_.size();
    }

    // === Material Groups ===
    void updateMaterialGroups(const std::vector<MParticle>& particles) {
        std::vector<ParticleMaterial> materials(particles.size());
        for (size_t i = 0; i < particles.size(); i++) {
            materials[i] = particles[i].material;
        }
        material_groups_.rebuild(materials.data(), materials.size());
    }

    // === Helper Functions ===
    void ensureCapacity(size_t particle_count) {
        if (particle_springs_.size() < particle_count) {
            particle_springs_.resize(particle_count);
            particle_bond_counts_.resize(particle_count, 0);
        }

        if (!contact_union_find_ || contact_union_find_->getNumComponents() != particle_count) {
            contact_union_find_ = std::make_unique<UnionFind>(particle_count);
            spring_union_find_ = std::make_unique<UnionFind>(particle_count);
        }
    }

    size_t countActiveSprings() const {
        return std::count_if(springs_.begin(), springs_.end(),
                            [](const MaterialSpring& s) { return s.active; });
    }

    inline auto startTimer() const {
        return std::chrono::high_resolution_clock::now();
    }

    inline double elapsedMs(const std::chrono::high_resolution_clock::time_point& start) const {
        auto end = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double, std::milli>(end - start).count();
    }
};

} // namespace digistar
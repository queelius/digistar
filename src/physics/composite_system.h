/**
 * Composite System - Dynamic Clustering with Union-Find
 *
 * Implements emergent composite body formation from spring networks.
 * Uses efficient Union-Find (Disjoint Set Union) data structure to track
 * connected components in real-time as springs form and break.
 */

#pragma once

#include <vector>
#include <unordered_map>
#include <cmath>
#include <algorithm>
#include <numeric>

namespace digistar {

/**
 * Union-Find (Disjoint Set Union) Data Structure
 *
 * Efficiently tracks connected components with nearly O(1) operations
 * using path compression and union by rank optimizations.
 */
class UnionFind {
private:
    std::vector<int> parent;  // Parent of each node
    std::vector<int> rank;    // Rank for union by rank
    std::vector<int> size;    // Size of each component
    int num_components;       // Number of disjoint sets

public:
    UnionFind(int n) : parent(n), rank(n, 0), size(n, 1), num_components(n) {
        // Initially, each particle is its own component
        std::iota(parent.begin(), parent.end(), 0);
    }

    // Find with path compression
    int find(int x) {
        if (parent[x] != x) {
            parent[x] = find(parent[x]);  // Path compression
        }
        return parent[x];
    }

    // Union by rank
    bool unite(int x, int y) {
        int root_x = find(x);
        int root_y = find(y);

        if (root_x == root_y) return false;  // Already connected

        // Union by rank - attach smaller tree under larger
        if (rank[root_x] < rank[root_y]) {
            parent[root_x] = root_y;
            size[root_y] += size[root_x];
        } else if (rank[root_x] > rank[root_y]) {
            parent[root_y] = root_x;
            size[root_x] += size[root_y];
        } else {
            parent[root_y] = root_x;
            size[root_x] += size[root_y];
            rank[root_x]++;
        }

        num_components--;
        return true;
    }

    // Check if two particles are in same component
    bool connected(int x, int y) {
        return find(x) == find(y);
    }

    // Get size of component containing x
    int componentSize(int x) {
        return size[find(x)];
    }

    // Get total number of components
    int getNumComponents() const {
        return num_components;
    }

    // Get all components as groups
    std::unordered_map<int, std::vector<int>> getComponents() {
        std::unordered_map<int, std::vector<int>> components;
        for (size_t i = 0; i < parent.size(); i++) {
            int root = find(i);
            components[root].push_back(i);
        }
        return components;
    }

    // Reset to initial state (all singletons)
    void reset() {
        std::iota(parent.begin(), parent.end(), 0);
        std::fill(rank.begin(), rank.end(), 0);
        std::fill(size.begin(), size.end(), 1);
        num_components = parent.size();
    }
};

/**
 * Composite Body - Emergent structure from connected particles
 */
struct CompositeBody {
    // Composition
    std::vector<int> particle_indices;  // Indices of particles in this composite
    int id;                             // Unique composite ID

    // Mass properties
    float total_mass = 0;
    float center_of_mass_x = 0;
    float center_of_mass_y = 0;

    // Motion properties
    float mean_velocity_x = 0;
    float mean_velocity_y = 0;
    float angular_velocity = 0;
    float angular_momentum = 0;

    // Temperature/Energy
    float internal_temperature = 0;  // Velocity dispersion
    float kinetic_energy = 0;
    float rotational_energy = 0;

    // Structural properties
    float moment_of_inertia = 0;
    float radius = 0;               // Bounding radius from COM
    float compactness = 0;          // Mass/area ratio
    float elongation = 1.0;         // Aspect ratio
    float principal_angle = 0;      // Orientation of principal axis

    // Spring network properties
    int num_internal_springs = 0;
    float avg_spring_stress = 0;
    float structural_integrity = 1.0;  // 0 = falling apart, 1 = solid
    float max_spring_stress = 0;

    // Visual properties (for rendering)
    uint32_t color = 0;  // Assigned color for this composite
    bool is_rigid = false;  // True if behaves as rigid body

    // Dynamic sphere tree for better collision detection
    struct BoundingSphere {
        float center_x, center_y;
        float radius;
        std::vector<int> particle_indices;  // Particles in this sphere
    };
    std::vector<BoundingSphere> sphere_tree;  // Multiple bounding spheres

    // Compute all properties from particle data
    template<typename Particle>
    void computeProperties(const std::vector<Particle>& particles) {
        if (particle_indices.empty()) return;

        // Reset aggregates
        total_mass = 0;
        center_of_mass_x = center_of_mass_y = 0;
        mean_velocity_x = mean_velocity_y = 0;

        // First pass: compute center of mass and total mass
        for (int idx : particle_indices) {
            const auto& p = particles[idx];
            total_mass += p.mass;
            center_of_mass_x += p.x * p.mass;
            center_of_mass_y += p.y * p.mass;
            mean_velocity_x += p.vx * p.mass;
            mean_velocity_y += p.vy * p.mass;
        }

        if (total_mass > 0) {
            center_of_mass_x /= total_mass;
            center_of_mass_y /= total_mass;
            mean_velocity_x /= total_mass;
            mean_velocity_y /= total_mass;
        }

        // Second pass: compute moment of inertia and velocity dispersion
        moment_of_inertia = 0;
        radius = 0;
        float velocity_variance = 0;
        angular_momentum = 0;

        for (int idx : particle_indices) {
            const auto& p = particles[idx];

            // Distance from center of mass
            float dx = p.x - center_of_mass_x;
            float dy = p.y - center_of_mass_y;
            float r2 = dx * dx + dy * dy;
            float r = std::sqrt(r2);

            // Update properties
            moment_of_inertia += p.mass * r2;
            radius = std::max(radius, r);

            // Velocity relative to COM motion
            float dvx = p.vx - mean_velocity_x;
            float dvy = p.vy - mean_velocity_y;
            velocity_variance += p.mass * (dvx * dvx + dvy * dvy);

            // Angular momentum (r × v)
            angular_momentum += p.mass * (dx * p.vy - dy * p.vx);
        }

        // Compute derived properties
        if (moment_of_inertia > 0) {
            angular_velocity = angular_momentum / moment_of_inertia;
            rotational_energy = 0.5f * moment_of_inertia * angular_velocity * angular_velocity;
        }

        // Internal temperature from velocity dispersion
        if (particle_indices.size() > 1) {
            internal_temperature = velocity_variance / (2.0f * total_mass);
        }

        // Kinetic energy
        kinetic_energy = 0.5f * total_mass * (mean_velocity_x * mean_velocity_x +
                                              mean_velocity_y * mean_velocity_y);

        // Compactness (density-like measure)
        if (radius > 0) {
            float area = M_PI * radius * radius;
            compactness = total_mass / area;
        }

        // Check if rigid (low velocity dispersion relative to rotational motion)
        is_rigid = (internal_temperature < 10.0f && particle_indices.size() > 3);

        // Build dynamic sphere tree for better collision detection
        buildSphereTree(particles);
    }

    // Build a sphere tree that captures the shape better than a single bounding sphere
    template<typename Particle>
    void buildSphereTree(const std::vector<Particle>& particles) {
        sphere_tree.clear();

        // Strategy: Use k-means clustering to create multiple spheres
        // This captures concavities and sparse structures better

        const int num_spheres = std::min(5, std::max(1, (int)particle_indices.size() / 10));

        if (particle_indices.size() <= 3 || num_spheres == 1) {
            // Too few particles, just use one sphere
            BoundingSphere sphere;
            sphere.center_x = center_of_mass_x;
            sphere.center_y = center_of_mass_y;
            sphere.radius = radius;
            sphere.particle_indices = particle_indices;
            sphere_tree.push_back(sphere);
            return;
        }

        // Simple spatial subdivision: divide particles into regions
        std::vector<BoundingSphere> spheres(num_spheres);
        std::vector<std::vector<int>> clusters(num_spheres);

        // Initialize cluster centers using furthest-point sampling
        std::vector<int> centers;
        centers.push_back(particle_indices[0]);

        for (int i = 1; i < num_spheres; i++) {
            float max_min_dist = 0;
            int furthest = -1;

            for (int idx : particle_indices) {
                float min_dist = 1e10f;
                for (int center : centers) {
                    float dx = particles[idx].x - particles[center].x;
                    float dy = particles[idx].y - particles[center].y;
                    float dist = dx*dx + dy*dy;
                    min_dist = std::min(min_dist, dist);
                }

                if (min_dist > max_min_dist) {
                    max_min_dist = min_dist;
                    furthest = idx;
                }
            }

            if (furthest >= 0) {
                centers.push_back(furthest);
            }
        }

        // Assign particles to nearest center
        for (int idx : particle_indices) {
            float min_dist = 1e10f;
            int best_cluster = 0;

            for (int i = 0; i < centers.size(); i++) {
                float dx = particles[idx].x - particles[centers[i]].x;
                float dy = particles[idx].y - particles[centers[i]].y;
                float dist = dx*dx + dy*dy;

                if (dist < min_dist) {
                    min_dist = dist;
                    best_cluster = i;
                }
            }

            clusters[best_cluster].push_back(idx);
        }

        // Create bounding sphere for each cluster
        for (int i = 0; i < num_spheres; i++) {
            if (clusters[i].empty()) continue;

            BoundingSphere sphere;
            sphere.particle_indices = clusters[i];

            // Compute center
            sphere.center_x = 0;
            sphere.center_y = 0;
            float total_mass = 0;

            for (int idx : clusters[i]) {
                float m = particles[idx].mass;
                sphere.center_x += particles[idx].x * m;
                sphere.center_y += particles[idx].y * m;
                total_mass += m;
            }

            if (total_mass > 0) {
                sphere.center_x /= total_mass;
                sphere.center_y /= total_mass;
            }

            // Compute radius
            sphere.radius = 0;
            for (int idx : clusters[i]) {
                float dx = particles[idx].x - sphere.center_x;
                float dy = particles[idx].y - sphere.center_y;
                float dist = std::sqrt(dx*dx + dy*dy) + particles[idx].radius;
                sphere.radius = std::max(sphere.radius, dist);
            }

            // Add some padding for safety
            sphere.radius *= 1.1f;

            sphere_tree.push_back(sphere);
        }
    }

    // Compute spring-related properties
    template<typename Spring>
    void computeSpringProperties(const std::vector<Spring>& springs) {
        num_internal_springs = 0;
        avg_spring_stress = 0;
        max_spring_stress = 0;

        for (const auto& spring : springs) {
            if (!spring.active) continue;

            // Check if both ends are in this composite
            bool p1_in = std::find(particle_indices.begin(), particle_indices.end(),
                                  spring.p1) != particle_indices.end();
            bool p2_in = std::find(particle_indices.begin(), particle_indices.end(),
                                  spring.p2) != particle_indices.end();

            if (p1_in && p2_in) {
                num_internal_springs++;

                // Estimate stress (simplified - would need actual spring data)
                float stress = 0.1f;  // Placeholder
                avg_spring_stress += stress;
                max_spring_stress = std::max(max_spring_stress, stress);
            }
        }

        if (num_internal_springs > 0) {
            avg_spring_stress /= num_internal_springs;

            // Structural integrity based on spring stress
            structural_integrity = 1.0f - std::min(1.0f, max_spring_stress / 2.0f);
        }
    }
};

/**
 * Composite Manager - Tracks and updates all composite bodies
 */
class CompositeManager {
private:
    std::vector<CompositeBody> composites;
    UnionFind* union_find = nullptr;
    int next_composite_id = 0;

    // Color palette for visualization
    std::vector<uint32_t> color_palette = {
        0xFF0080FF,  // Blue
        0xFFFF0080,  // Pink
        0xFF80FF00,  // Green
        0xFFFF8000,  // Orange
        0xFF00FFFF,  // Cyan
        0xFFFF00FF,  // Magenta
        0xFFFFFF00,  // Yellow
        0xFF00FF80,  // Teal
        0xFFFF80FF,  // Light purple
        0xFF80FFFF,  // Light blue
    };

public:
    CompositeManager(size_t num_particles) {
        union_find = new UnionFind(num_particles);
    }

    ~CompositeManager() {
        delete union_find;
    }

    // Update composites based on spring network
    template<typename Particle, typename Spring>
    void updateComposites(const std::vector<Particle>& particles,
                         const std::vector<Spring>& springs) {
        // Reset union-find
        union_find->reset();

        // Build connected components from active springs
        for (const auto& spring : springs) {
            if (spring.active &&
                spring.p1 < particles.size() &&
                spring.p2 < particles.size()) {
                union_find->unite(spring.p1, spring.p2);
            }
        }

        // Get all components
        auto components = union_find->getComponents();

        // Clear old composites
        composites.clear();

        // Create composite bodies for non-singleton components
        int color_idx = 0;
        for (const auto& [root, indices] : components) {
            if (indices.size() > 1) {  // Skip single particles
                CompositeBody body;
                body.id = next_composite_id++;
                body.particle_indices = indices;
                body.color = color_palette[color_idx % color_palette.size()];
                color_idx++;

                // Compute properties
                body.computeProperties(particles);
                body.computeSpringProperties(springs);

                composites.push_back(body);
            }
        }

        // Sort by size for consistent rendering
        std::sort(composites.begin(), composites.end(),
                 [](const CompositeBody& a, const CompositeBody& b) {
                     return a.particle_indices.size() > b.particle_indices.size();
                 });
    }

    // Get composite containing a particle (or nullptr)
    const CompositeBody* getCompositeForParticle(int particle_idx) const {
        for (const auto& comp : composites) {
            if (std::find(comp.particle_indices.begin(),
                         comp.particle_indices.end(),
                         particle_idx) != comp.particle_indices.end()) {
                return &comp;
            }
        }
        return nullptr;
    }

    // Get all composites
    const std::vector<CompositeBody>& getComposites() const {
        return composites;
    }

    // Get statistics
    struct Stats {
        int num_composites = 0;
        int largest_composite_size = 0;
        int num_rigid_bodies = 0;
        float avg_composite_size = 0;
        float total_composite_mass = 0;
    };

    Stats getStats() const {
        Stats stats;
        stats.num_composites = composites.size();

        if (!composites.empty()) {
            int total_particles = 0;
            for (const auto& comp : composites) {
                int size = comp.particle_indices.size();
                total_particles += size;
                stats.largest_composite_size = std::max(stats.largest_composite_size, size);
                stats.total_composite_mass += comp.total_mass;
                if (comp.is_rigid) stats.num_rigid_bodies++;
            }
            stats.avg_composite_size = float(total_particles) / composites.size();
        }

        return stats;
    }

    // Check if two particles are connected
    bool areConnected(int p1, int p2) const {
        return union_find->connected(p1, p2);
    }

    // Get number of components
    int getNumComponents() const {
        return union_find->getNumComponents();
    }
};

} // namespace digistar
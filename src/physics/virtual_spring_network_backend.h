/**
 * Virtual Spring Network Backend Interface
 *
 * Abstract interface for automatically generated "virtual" spring networks
 * and composite body management. Virtual springs form dynamically based on
 * proximity and velocity criteria, creating emergent clustering behavior.
 *
 * This is distinct from explicit spring networks where springs are
 * manually defined and persistent.
 */

#pragma once

#include <vector>
#include <string>
#include <memory>

namespace digistar {

// Forward declaration
template<typename Particle>
class SparseSpatialGrid;

/**
 * Spring connection between two particles
 */
struct Spring {
    uint32_t p1, p2;           // Particle indices
    float rest_length;         // Equilibrium distance
    float stiffness;           // Spring constant
    float damping;             // Damping coefficient
    float max_force;           // Breaking threshold
    float current_force;       // Current force magnitude
    bool active;               // Is spring intact?
    uint32_t composite_id;     // Which composite this belongs to
};

/**
 * Composite body formed from connected particles
 */
struct CompositeBody {
    std::vector<uint32_t> particle_indices;
    uint32_t id;

    // Aggregate properties
    float total_mass;
    float center_of_mass_x, center_of_mass_y;
    float mean_velocity_x, mean_velocity_y;
    float angular_velocity;
    float moment_of_inertia;
    float radius;  // Bounding radius from COM

    // Spring network properties
    uint32_t num_internal_springs;
    float avg_spring_stress;
    float max_spring_stress;

    // Visual properties
    uint32_t color;
    bool is_rigid;  // High connectivity, low internal motion
};

/**
 * Statistics for spring network performance
 */
struct SpringNetworkStats {
    double spring_update_ms = 0;     // Time to update spring lifecycle
    double force_calc_ms = 0;        // Time to calculate spring forces
    double cluster_update_ms = 0;    // Time to update clusters
    double composite_calc_ms = 0;    // Time to compute composite properties
    double total_time_ms = 0;
    size_t active_springs = 0;
    size_t springs_formed = 0;
    size_t springs_broken = 0;
    size_t num_composites = 0;
    size_t largest_composite = 0;
};

/**
 * Abstract base class for virtual spring network backends
 */
template<typename Particle>
class IVirtualSpringNetworkBackend {
public:
    struct Config {
        // Spring formation criteria
        float formation_distance = 3.0f;      // Max distance to form spring
        float formation_velocity = 2.0f;      // Max relative velocity
        float spring_stiffness = 100.0f;      // Default spring constant
        float spring_damping = 0.5f;          // Default damping

        // Spring breaking criteria
        float max_stretch = 2.0f;             // Break if stretched beyond rest_length * max_stretch
        float max_force = 1000.0f;            // Break if force exceeds threshold

        // Composite detection
        bool track_composites = true;         // Enable composite tracking
        uint32_t min_composite_size = 2;      // Minimum particles for a composite

        // Performance
        int num_threads = 0;                  // 0 = auto-detect
        uint32_t max_springs_per_particle = 12; // Limit connections
        uint32_t max_total_springs = 1000000;   // Global spring limit
    };

    virtual ~IVirtualSpringNetworkBackend() = default;

    /**
     * Update virtual spring network and compute forces
     * 1. Automatically form new springs based on proximity/velocity
     * 2. Compute spring forces
     * 3. Break overstressed springs
     * 4. Update composite bodies incrementally
     *
     * Updates particle fx, fy fields with spring forces
     */
    virtual void updateVirtualSprings(
        std::vector<Particle>& particles,
        SparseSpatialGrid<Particle>& grid,
        float dt) = 0;

    /**
     * Get all active springs
     */
    virtual const std::vector<Spring>& getSprings() const = 0;

    /**
     * Get all composite bodies
     */
    virtual const std::vector<CompositeBody>& getComposites() const = 0;

    /**
     * Check if two particles are connected (same composite)
     */
    virtual bool areConnected(uint32_t p1, uint32_t p2) const = 0;

    /**
     * Get composite ID for a particle (-1 if singleton)
     */
    virtual int getCompositeId(uint32_t particle_idx) const = 0;

    /**
     * Get backend name for identification
     */
    virtual std::string getName() const = 0;

    /**
     * Get last computation statistics
     */
    virtual SpringNetworkStats getStats() const { return stats_; }

    /**
     * Set configuration
     */
    virtual void setConfig(const Config& config) { config_ = config; }

    /**
     * Get configuration
     */
    virtual const Config& getConfig() const { return config_; }

    /**
     * Clear all springs and composites (for reset)
     */
    virtual void clear() = 0;

protected:
    Config config_;
    mutable SpringNetworkStats stats_;
};

} // namespace digistar
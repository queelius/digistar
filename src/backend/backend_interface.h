#pragma once

#include <memory>
#include <string>
#include "../physics/types.h"
#include "../physics/pools.h"
#include "../physics/spatial_index.h"

namespace digistar {

// Simulation statistics
struct SimulationStats {
    // Performance metrics
    float update_time_ms = 0;
    float gravity_time_ms = 0;
    float contact_time_ms = 0;
    float spring_time_ms = 0;
    float integration_time_ms = 0;
    float spatial_index_time = 0;
    float collision_detection_time = 0;
    float composite_detection_time = 0;
    
    // Aliases for consistency
    float update_time = 0;  // Same as update_time_ms
    float integrate_time = 0;  // Same as integration_time_ms
    
    // System metrics
    size_t active_particles = 0;
    size_t active_springs = 0;
    size_t active_contacts = 0;
    size_t active_composites = 0;
    
    // Physical metrics
    float total_energy = 0;
    float total_momentum_x = 0;
    float total_momentum_y = 0;
    float average_temperature = 0;
    float max_velocity = 0;
};

// All simulation data in one place - easy to extend
struct SimulationState {
    // Core particle data (Structure of Arrays for SIMD)
    ParticlePool particles;
    
    // Dynamic structures
    SpringPool springs;
    ContactPool contacts;
    CompositePool composites;
    
    // Spatial indices at different resolutions
    std::unique_ptr<SpatialIndex> contact_index;    // Fine: 2-4 units
    std::unique_ptr<SpatialIndex> spring_index;     // Medium: 10-20 units  
    std::unique_ptr<SpatialIndex> thermal_index;    // Coarse: 50-100 units
    std::unique_ptr<SpatialIndex> radiation_index;  // Very coarse: 200-500 units
    
    // Field data (for GPU, these could be textures)
    RadiationField radiation;
    ThermalField thermal;
    GravityField gravity;  // For PM solver
    
    // Statistics and metadata
    SimulationStats stats;
};

// Configuration for physics updates
struct PhysicsConfig {
    // What to update
    uint32_t enabled_systems = 0;  // Bitmask for fast checking
    
    enum SystemFlags {
        GRAVITY     = 1 << 0,
        CONTACTS    = 1 << 1,
        SPRINGS     = 1 << 2,
        RADIATION   = 1 << 3,
        THERMAL     = 1 << 4,
        SPRING_FIELD = 1 << 5,  // Virtual spring formation
        FUSION      = 1 << 6,
        FISSION     = 1 << 7
    };
    
    // Gravity parameters
    enum GravityMode {
        DIRECT_N2,      // O(N²) for small systems
        PARTICLE_MESH,  // FFT-based for large systems
        BARNES_HUT      // Tree-based (future)
    } gravity_mode = PARTICLE_MESH;
    
    float gravity_strength = 6.67430e-11f;  // Gravitational constant
    
    // Contact parameters
    float contact_stiffness = 1000.0f;
    float contact_damping = 10.0f;
    
    // Spring parameters  
    float spring_break_strain = 0.5f;
    float spring_formation_distance = 10.0f;
    
    // Integration method
    enum IntegratorType {
        VELOCITY_VERLET = 0,
        SEMI_IMPLICIT = 1,
        LEAPFROG = 2,
        FORWARD_EULER = 3
    } default_integrator = SEMI_IMPLICIT;
    
    // Performance hints
    bool prefer_accuracy = false;  // false = prefer speed
    size_t max_contacts_per_particle = 8;  // Helps with allocation
    size_t max_springs_per_particle = 12;
};

// Initialization parameters
struct SimulationConfig {
    size_t max_particles = 10'000'000;
    size_t max_springs = 50'000'000;
    size_t max_contacts = 1'000'000;  // Active contacts at once
    
    float world_size = 10000.0f;  // For toroidal wrapping
    bool use_toroidal = true;
    
    // Grid resolutions
    float contact_cell_size = 2.0f;
    float spring_cell_size = 10.0f;
    float thermal_cell_size = 50.0f;
    float radiation_cell_size = 200.0f;
    
    // PM solver parameters (if using)
    size_t pm_grid_size = 512;  // 512x512 for 2D
    
    // Threading hints
    size_t num_threads = 0;  // 0 = auto-detect
    bool enable_simd = true;
};

// Abstract backend interface - minimal and focused
class IBackend {
public:
    virtual ~IBackend() = default;
    
    // Lifecycle
    virtual void initialize(const SimulationConfig& config) = 0;
    virtual void shutdown() = 0;
    
    // Main simulation loop - backend handles everything internally
    virtual void step(SimulationState& state, 
                     const PhysicsConfig& config,
                     float dt) = 0;
    
    // Performance and debugging
    virtual SimulationStats getStats() const = 0;
    virtual std::string getName() const = 0;
    
    // Backend capabilities query
    virtual uint32_t getSupportedSystems() const = 0;
    virtual size_t getMaxParticles() const = 0;
};


// Factory for creating backends
class BackendFactory {
public:
    enum class Type {
        CPU,
        CPU_SIMD,       // CPU with explicit SIMD
        CUDA,
        OpenCL,
        Distributed
    };
    
    static std::unique_ptr<IBackend> create(Type type, const SimulationConfig& config);
    static bool isSupported(Type type);
    static std::string getDescription(Type type);
};

} // namespace digistar
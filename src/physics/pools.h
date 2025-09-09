#pragma once

#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <algorithm>
#include <unordered_map>
#include <limits>
#include <stdexcept>
#include "types.h"

namespace digistar {

// Improved particle pool with proper add/remove and ID management
class ParticlePool {
public:
    // ===== Data Arrays (Structure of Arrays) =====
    // Position components
    float* pos_x = nullptr;
    float* pos_y = nullptr;
    
    // Velocity components  
    float* vel_x = nullptr;
    float* vel_y = nullptr;
    
    // Force accumulator
    float* force_x = nullptr;
    float* force_y = nullptr;
    
    // Physical properties
    float* mass = nullptr;
    float* radius = nullptr;
    float* temperature = nullptr;
    
    // Material properties
    uint8_t* material_type = nullptr;
    float* charge = nullptr;
    
    // Metadata
    uint32_t* composite_id = nullptr;  // Which composite this belongs to
    
    // ===== Pool Management =====
    size_t capacity = 0;  // Maximum particles
    size_t count = 0;     // Current active particles (always contiguous from 0)
    
    // ID management - provides stable IDs even when particles move
    static constexpr uint32_t INVALID_ID = std::numeric_limits<uint32_t>::max();
    static constexpr uint32_t INVALID_INDEX = std::numeric_limits<uint32_t>::max();
    
private:
    // ID to index mapping
    std::unordered_map<uint32_t, uint32_t> id_to_index_map;
    uint32_t* index_to_id = nullptr;  // Array: index -> ID
    uint32_t next_id = 0;  // Next ID to assign
    
    // Memory allocation helper
    void* alloc_aligned(size_t size) {
        void* ptr = nullptr;
        if (posix_memalign(&ptr, 64, size) != 0) {  // 64-byte alignment for AVX-512
            throw std::bad_alloc();
        }
        std::memset(ptr, 0, size);
        return ptr;
    }
    
public:
    ParticlePool() = default;
    ~ParticlePool() { deallocate(); }
    
    // ===== Memory Management =====
    void allocate(size_t max_particles) {
        deallocate();
        capacity = max_particles;
        
        // Allocate all arrays
        pos_x = (float*)alloc_aligned(capacity * sizeof(float));
        pos_y = (float*)alloc_aligned(capacity * sizeof(float));
        vel_x = (float*)alloc_aligned(capacity * sizeof(float));
        vel_y = (float*)alloc_aligned(capacity * sizeof(float));
        force_x = (float*)alloc_aligned(capacity * sizeof(float));
        force_y = (float*)alloc_aligned(capacity * sizeof(float));
        mass = (float*)alloc_aligned(capacity * sizeof(float));
        radius = (float*)alloc_aligned(capacity * sizeof(float));
        temperature = (float*)alloc_aligned(capacity * sizeof(float));
        charge = (float*)alloc_aligned(capacity * sizeof(float));
        material_type = (uint8_t*)alloc_aligned(capacity * sizeof(uint8_t));
        composite_id = (uint32_t*)alloc_aligned(capacity * sizeof(uint32_t));
        index_to_id = (uint32_t*)alloc_aligned(capacity * sizeof(uint32_t));
        
        // Initialize
        for (size_t i = 0; i < capacity; ++i) {
            temperature[i] = 293.0f;  // Room temperature default
            composite_id[i] = INVALID_ID;
            index_to_id[i] = INVALID_ID;
        }
        
        // Clear mappings
        id_to_index_map.clear();
        id_to_index_map.reserve(capacity);
        count = 0;
        next_id = 0;
        
        // Initialize backwards compatibility
        init_backwards_compat();
    }
    
    void deallocate() {
        free(pos_x); pos_x = nullptr;
        free(pos_y); pos_y = nullptr;
        free(vel_x); vel_x = nullptr;
        free(vel_y); vel_y = nullptr;
        free(force_x); force_x = nullptr;
        free(force_y); force_y = nullptr;
        free(mass); mass = nullptr;
        free(radius); radius = nullptr;
        free(temperature); temperature = nullptr;
        free(charge); charge = nullptr;
        free(material_type); material_type = nullptr;
        free(composite_id); composite_id = nullptr;
        free(index_to_id); index_to_id = nullptr;
        free(active_indices); active_indices = nullptr;  // Free backwards compat array
        
        id_to_index_map.clear();
        capacity = 0;
        count = 0;
    }
    
    // ===== Particle Management =====
    
    // Create a new particle, returns stable ID
    uint32_t create(float x, float y, float vx, float vy, float m, float r) {
        if (count >= capacity) return INVALID_ID;
        
        uint32_t idx = count;
        uint32_t id = next_id++;
        
        // Set data
        pos_x[idx] = x;
        pos_y[idx] = y;
        vel_x[idx] = vx;
        vel_y[idx] = vy;
        mass[idx] = m;
        radius[idx] = r;
        force_x[idx] = 0;
        force_y[idx] = 0;
        temperature[idx] = 293.0f;
        charge[idx] = 0;
        material_type[idx] = 0;
        composite_id[idx] = INVALID_ID;
        
        // Update mappings
        id_to_index_map[id] = idx;
        index_to_id[idx] = id;
        
        count++;
        return id;
    }
    
    // Remove particle by ID (swap-and-pop)
    void destroy(uint32_t id) {
        auto it = id_to_index_map.find(id);
        if (it == id_to_index_map.end()) return;
        
        uint32_t idx = it->second;
        uint32_t last_idx = count - 1;
        
        if (idx != last_idx) {
            // Swap with last particle
            pos_x[idx] = pos_x[last_idx];
            pos_y[idx] = pos_y[last_idx];
            vel_x[idx] = vel_x[last_idx];
            vel_y[idx] = vel_y[last_idx];
            force_x[idx] = force_x[last_idx];
            force_y[idx] = force_y[last_idx];
            mass[idx] = mass[last_idx];
            radius[idx] = radius[last_idx];
            temperature[idx] = temperature[last_idx];
            charge[idx] = charge[last_idx];
            material_type[idx] = material_type[last_idx];
            composite_id[idx] = composite_id[last_idx];
            
            // Update mapping for moved particle
            uint32_t moved_id = index_to_id[last_idx];
            id_to_index_map[moved_id] = idx;
            index_to_id[idx] = moved_id;
        }
        
        // Remove from mappings
        id_to_index_map.erase(id);
        index_to_id[last_idx] = INVALID_ID;
        count--;
    }
    
    // Check if particle exists
    bool exists(uint32_t id) const {
        return id_to_index_map.find(id) != id_to_index_map.end();
    }
    
    // Get index from ID (for advanced usage)
    uint32_t get_index(uint32_t id) const {
        auto it = id_to_index_map.find(id);
        return (it != id_to_index_map.end()) ? it->second : INVALID_INDEX;
    }
    
    // Get ID from index
    uint32_t get_id(uint32_t index) const {
        return (index < count) ? index_to_id[index] : INVALID_ID;
    }
    
    // ===== Bulk Operations =====
    
    // Clear all forces to zero
    void clear_forces() {
        // Only clear active particles for efficiency
        std::memset(force_x, 0, count * sizeof(float));
        std::memset(force_y, 0, count * sizeof(float));
    }
    
    // Apply toroidal boundary conditions
    void apply_boundaries(float world_size) {
        for (size_t i = 0; i < count; i++) {
            // Wrap X
            while (pos_x[i] < 0) pos_x[i] += world_size;
            while (pos_x[i] >= world_size) pos_x[i] -= world_size;
            
            // Wrap Y
            while (pos_y[i] < 0) pos_y[i] += world_size;
            while (pos_y[i] >= world_size) pos_y[i] -= world_size;
        }
    }
    
    // ===== Accessors =====
    
    // Number of active particles
    size_t size() const { return count; }
    
    // Maximum capacity
    size_t max_size() const { return capacity; }
    
    // Direct access to a particle by ID
    struct ParticleRef {
        ParticlePool* pool;
        uint32_t index;
        
        float& x() { return pool->pos_x[index]; }
        float& y() { return pool->pos_y[index]; }
        float& vx() { return pool->vel_x[index]; }
        float& vy() { return pool->vel_y[index]; }
        float& fx() { return pool->force_x[index]; }
        float& fy() { return pool->force_y[index]; }
        float& mass() { return pool->mass[index]; }
        float& radius() { return pool->radius[index]; }
        float& temp() { return pool->temperature[index]; }
        float& charge() { return pool->charge[index]; }
        uint8_t& material() { return pool->material_type[index]; }
        uint32_t& composite() { return pool->composite_id[index]; }
    };
    
    ParticleRef get(uint32_t id) {
        uint32_t idx = get_index(id);
        if (idx == INVALID_INDEX) {
            throw std::runtime_error("Invalid particle ID");
        }
        return ParticleRef{this, idx};
    }
    
    // For backwards compatibility with existing code
    uint32_t* active_indices = nullptr;  // Points to simple [0,1,2,...] array
    bool* alive = nullptr;  // Not used - all particles 0..count-1 are alive
    
    // Initialize backwards compatibility arrays in allocate()
    void init_backwards_compat() {
        if (!active_indices) {
            active_indices = (uint32_t*)alloc_aligned(capacity * sizeof(uint32_t));
            for (size_t i = 0; i < capacity; ++i) {
                active_indices[i] = i;  // Simple identity mapping
            }
        }
    }
};

// Spring pool - references particles by ID
class SpringPool {
public:
    // Spring endpoints (particle IDs, not indices!)
    uint32_t* particle1_id = nullptr;
    uint32_t* particle2_id = nullptr;
    
    // Spring properties
    float* rest_length = nullptr;
    float* stiffness = nullptr;
    float* damping = nullptr;
    
    // Runtime state
    float* current_length = nullptr;  // Cached for performance
    float* strain = nullptr;          // (current - rest) / rest
    float* thermal_conductivity = nullptr;  // Thermal conductivity of spring
    bool* active = nullptr;           // Is spring active?
    
    size_t capacity = 0;
    size_t count = 0;
    
    void allocate(size_t max_springs) {
        deallocate();
        capacity = max_springs;
        
        auto alloc = [](size_t size) {
            void* ptr = nullptr;
            posix_memalign(&ptr, 64, size);
            std::memset(ptr, 0, size);
            return ptr;
        };
        
        particle1_id = (uint32_t*)alloc(capacity * sizeof(uint32_t));
        particle2_id = (uint32_t*)alloc(capacity * sizeof(uint32_t));
        rest_length = (float*)alloc(capacity * sizeof(float));
        stiffness = (float*)alloc(capacity * sizeof(float));
        damping = (float*)alloc(capacity * sizeof(float));
        current_length = (float*)alloc(capacity * sizeof(float));
        strain = (float*)alloc(capacity * sizeof(float));
        thermal_conductivity = (float*)alloc(capacity * sizeof(float));
        active = (bool*)alloc(capacity * sizeof(bool));
        
        count = 0;
    }
    
    void deallocate() {
        free(particle1_id); particle1_id = nullptr;
        free(particle2_id); particle2_id = nullptr;
        free(rest_length); rest_length = nullptr;
        free(stiffness); stiffness = nullptr;
        free(damping); damping = nullptr;
        free(current_length); current_length = nullptr;
        free(strain); strain = nullptr;
        free(thermal_conductivity); thermal_conductivity = nullptr;
        free(active); active = nullptr;
        capacity = 0;
        count = 0;
    }
    
    uint32_t create(uint32_t p1_id, uint32_t p2_id, float rest, float stiff, float damp) {
        if (count >= capacity) return ParticlePool::INVALID_ID;
        
        uint32_t idx = count++;
        particle1_id[idx] = p1_id;
        particle2_id[idx] = p2_id;
        rest_length[idx] = rest;
        stiffness[idx] = stiff;
        damping[idx] = damp;
        current_length[idx] = rest;
        strain[idx] = 0;
        thermal_conductivity[idx] = 1.0f;  // Default thermal conductivity
        active[idx] = true;
        
        return idx;
    }
    
    void deactivate(uint32_t idx) {
        if (idx < count) {
            active[idx] = false;
        }
    }
    
    void break_spring(uint32_t idx) {
        deactivate(idx);
    }
    
    // Compact to remove inactive springs (call periodically)
    void compact() {
        size_t write_idx = 0;
        for (size_t read_idx = 0; read_idx < count; read_idx++) {
            if (active[read_idx]) {
                if (write_idx != read_idx) {
                    particle1_id[write_idx] = particle1_id[read_idx];
                    particle2_id[write_idx] = particle2_id[read_idx];
                    rest_length[write_idx] = rest_length[read_idx];
                    stiffness[write_idx] = stiffness[read_idx];
                    damping[write_idx] = damping[read_idx];
                    current_length[write_idx] = current_length[read_idx];
                    strain[write_idx] = strain[read_idx];
                    active[write_idx] = true;
                }
                write_idx++;
            }
        }
        count = write_idx;
    }
    
    void clear() {
        count = 0;
    }
    
    ~SpringPool() { deallocate(); }
};

// Contact pool - rebuilt each frame, no need for stable IDs
class ContactPool {
public:
    uint32_t* particle1 = nullptr;  // Indices, not IDs (for performance)
    uint32_t* particle2 = nullptr;
    
    float* overlap = nullptr;
    float* normal_x = nullptr;
    float* normal_y = nullptr;
    
    size_t capacity = 0;
    size_t count = 0;
    
    void allocate(size_t max_contacts) {
        deallocate();
        capacity = max_contacts;
        
        auto alloc = [](size_t size) {
            void* ptr = nullptr;
            posix_memalign(&ptr, 64, size);
            std::memset(ptr, 0, size);
            return ptr;
        };
        
        particle1 = (uint32_t*)alloc(capacity * sizeof(uint32_t));
        particle2 = (uint32_t*)alloc(capacity * sizeof(uint32_t));
        overlap = (float*)alloc(capacity * sizeof(float));
        normal_x = (float*)alloc(capacity * sizeof(float));
        normal_y = (float*)alloc(capacity * sizeof(float));
        
        count = 0;
    }
    
    void deallocate() {
        free(particle1); particle1 = nullptr;
        free(particle2); particle2 = nullptr;
        free(overlap); overlap = nullptr;
        free(normal_x); normal_x = nullptr;
        free(normal_y); normal_y = nullptr;
        capacity = 0;
        count = 0;
    }
    
    void clear() {
        count = 0;
    }
    
    bool add(uint32_t p1, uint32_t p2, float ovlp, float nx, float ny) {
        if (count >= capacity) return false;
        
        uint32_t idx = count++;
        particle1[idx] = p1;
        particle2[idx] = p2;
        overlap[idx] = ovlp;
        normal_x[idx] = nx;
        normal_y[idx] = ny;
        return true;
    }
    
    ~ContactPool() { deallocate(); }
};

// Composite pool - groups of particles
class CompositePool {
public:
    // Per-composite data
    uint32_t* first_particle = nullptr;  // Index into particle_ids array
    uint32_t* particle_count = nullptr;  // Number of particles in this composite
    float* center_x = nullptr;
    float* center_y = nullptr;
    float* total_mass = nullptr;
    float* angular_velocity = nullptr;
    
    // Particle membership (flat array, indexed by first_particle)
    uint32_t* particle_ids = nullptr;  // All particle IDs in all composites
    
    size_t capacity = 0;
    size_t count = 0;
    size_t particle_capacity = 0;
    size_t total_particles = 0;
    
    void allocate(size_t max_composites, size_t max_total_particles) {
        deallocate();
        capacity = max_composites;
        particle_capacity = max_total_particles;
        
        auto alloc = [](size_t size) {
            void* ptr = nullptr;
            posix_memalign(&ptr, 64, size);
            std::memset(ptr, 0, size);
            return ptr;
        };
        
        first_particle = (uint32_t*)alloc(capacity * sizeof(uint32_t));
        particle_count = (uint32_t*)alloc(capacity * sizeof(uint32_t));
        center_x = (float*)alloc(capacity * sizeof(float));
        center_y = (float*)alloc(capacity * sizeof(float));
        total_mass = (float*)alloc(capacity * sizeof(float));
        angular_velocity = (float*)alloc(capacity * sizeof(float));
        
        particle_ids = (uint32_t*)alloc(particle_capacity * sizeof(uint32_t));
        
        count = 0;
        total_particles = 0;
    }
    
    void deallocate() {
        free(first_particle); first_particle = nullptr;
        free(particle_count); particle_count = nullptr;
        free(center_x); center_x = nullptr;
        free(center_y); center_y = nullptr;
        free(total_mass); total_mass = nullptr;
        free(angular_velocity); angular_velocity = nullptr;
        free(particle_ids); particle_ids = nullptr;
        capacity = 0;
        count = 0;
        particle_capacity = 0;
        total_particles = 0;
    }
    
    void clear() {
        count = 0;
        total_particles = 0;
    }
    
    ~CompositePool() { deallocate(); }
};

// Field grids for PM solver and radiation
struct GravityField {
    float* density = nullptr;
    float* potential = nullptr;
    float* force_x = nullptr;
    float* force_y = nullptr;
    size_t grid_size = 0;
    
    void allocate(size_t size) {
        deallocate();
        grid_size = size;
        size_t total = size * size;
        
        density = (float*)std::calloc(total, sizeof(float));
        potential = (float*)std::calloc(total, sizeof(float));
        force_x = (float*)std::calloc(total, sizeof(float));
        force_y = (float*)std::calloc(total, sizeof(float));
    }
    
    void deallocate() {
        free(density); density = nullptr;
        free(potential); potential = nullptr;
        free(force_x); force_x = nullptr;
        free(force_y); force_y = nullptr;
        grid_size = 0;
    }
    
    void clear() {
        size_t total = grid_size * grid_size;
        std::memset(density, 0, total * sizeof(float));
        std::memset(potential, 0, total * sizeof(float));
        std::memset(force_x, 0, total * sizeof(float));
        std::memset(force_y, 0, total * sizeof(float));
    }
    
    ~GravityField() { deallocate(); }
};

using RadiationField = GravityField;  // Same structure
using ThermalField = GravityField;    // Same structure

} // namespace digistar
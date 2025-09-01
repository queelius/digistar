#pragma once

#include <cstdint>
#include <vector>
#include <cstring>
#include <cstdlib>
#include <new>
#include "types.h"

namespace digistar {

// Structure of Arrays for SIMD-friendly particle data
class ParticlePool {
public:
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
    float* temp_internal = nullptr;
    
    // Material properties
    uint8_t* material_type = nullptr;
    float* charge = nullptr;
    float* temperature = nullptr;  // External temperature (different from temp_internal)
    
    // Metadata
    uint32_t* particle_id = nullptr;
    uint32_t* composite_id = nullptr;  // Which composite this belongs to
    uint8_t* integrator_type = nullptr;  // Per-particle integration method
    bool* alive = nullptr;  // Is particle active/alive
    
    // Grid tracking (which cell in each spatial index)
    uint64_t* grid_cells = nullptr;  // Packed: 16 bits per grid level
    
    // Pool management
    size_t capacity = 0;
    size_t count = 0;
    uint32_t* active_indices = nullptr;  // For iteration over active particles
    
    ParticlePool() = default;
    ~ParticlePool() { deallocate(); }
    
    void allocate(size_t max_particles) {
        deallocate();
        capacity = max_particles;
        
        // Allocate aligned arrays for SIMD
        auto alloc_aligned = [](size_t size) {
            void* ptr = nullptr;
            if (posix_memalign(&ptr, 64, size) != 0) {  // 64-byte alignment for AVX-512
                throw std::bad_alloc();
            }
            std::memset(ptr, 0, size);
            return ptr;
        };
        
        pos_x = (float*)alloc_aligned(capacity * sizeof(float));
        pos_y = (float*)alloc_aligned(capacity * sizeof(float));
        vel_x = (float*)alloc_aligned(capacity * sizeof(float));
        vel_y = (float*)alloc_aligned(capacity * sizeof(float));
        force_x = (float*)alloc_aligned(capacity * sizeof(float));
        force_y = (float*)alloc_aligned(capacity * sizeof(float));
        mass = (float*)alloc_aligned(capacity * sizeof(float));
        radius = (float*)alloc_aligned(capacity * sizeof(float));
        temp_internal = (float*)alloc_aligned(capacity * sizeof(float));
        charge = (float*)alloc_aligned(capacity * sizeof(float));
        temperature = (float*)alloc_aligned(capacity * sizeof(float));
        
        material_type = (uint8_t*)alloc_aligned(capacity * sizeof(uint8_t));
        particle_id = (uint32_t*)alloc_aligned(capacity * sizeof(uint32_t));
        composite_id = (uint32_t*)alloc_aligned(capacity * sizeof(uint32_t));
        integrator_type = (uint8_t*)alloc_aligned(capacity * sizeof(uint8_t));
        alive = (bool*)alloc_aligned(capacity * sizeof(bool));
        grid_cells = (uint64_t*)alloc_aligned(capacity * sizeof(uint64_t));
        
        active_indices = (uint32_t*)alloc_aligned(capacity * sizeof(uint32_t));
        
        // Initialize active indices
        for (size_t i = 0; i < capacity; ++i) {
            active_indices[i] = i;
            particle_id[i] = i;
            alive[i] = false;  // Start with all particles inactive
            temperature[i] = 293.0f;  // Room temperature default
        }
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
        free(temp_internal); temp_internal = nullptr;
        free(charge); charge = nullptr;
        free(temperature); temperature = nullptr;
        free(material_type); material_type = nullptr;
        free(particle_id); particle_id = nullptr;
        free(composite_id); composite_id = nullptr;
        free(integrator_type); integrator_type = nullptr;
        free(alive); alive = nullptr;
        free(grid_cells); grid_cells = nullptr;
        free(active_indices); active_indices = nullptr;
        capacity = 0;
        count = 0;
    }
    
    uint32_t add_particle(float x, float y, float vx, float vy, float m, float r) {
        if (count >= capacity) return UINT32_MAX;
        
        uint32_t idx = count++;
        pos_x[idx] = x;
        pos_y[idx] = y;
        vel_x[idx] = vx;
        vel_y[idx] = vy;
        mass[idx] = m;
        radius[idx] = r;
        force_x[idx] = 0;
        force_y[idx] = 0;
        temp_internal[idx] = 293.0f;  // Room temperature
        temperature[idx] = 293.0f;  // External temperature
        charge[idx] = 0;
        material_type[idx] = 0;
        composite_id[idx] = UINT32_MAX;  // Not in composite
        integrator_type[idx] = 0;  // Default integrator
        alive[idx] = true;  // New particle is alive
        
        return idx;
    }
    
    // Zero forces for next frame
    void clear_forces() {
        std::memset(force_x, 0, count * sizeof(float));
        std::memset(force_y, 0, count * sizeof(float));
    }
};

// Spring pool - also Structure of Arrays
class SpringPool {
public:
    uint32_t* particle1 = nullptr;
    uint32_t* particle2 = nullptr;
    
    float* rest_length = nullptr;
    float* stiffness = nullptr;
    float* damping = nullptr;
    float* break_strain = nullptr;
    float* max_strain = nullptr;  // Maximum allowed strain before breaking
    float* current_strain = nullptr;
    
    uint8_t* material_type = nullptr;
    uint8_t* is_broken = nullptr;
    bool* alive = nullptr;  // Is spring active
    
    float* thermal_conductivity = nullptr;
    float* damage = nullptr;
    
    size_t capacity = 0;
    size_t count = 0;
    std::vector<uint32_t> free_list;  // Reuse broken spring slots
    
    void allocate(size_t max_springs) {
        deallocate();
        capacity = max_springs;
        
        auto alloc_aligned = [](size_t size) {
            void* ptr = nullptr;
            if (posix_memalign(&ptr, 64, size) != 0) {
                throw std::bad_alloc();
            }
            std::memset(ptr, 0, size);
            return ptr;
        };
        
        particle1 = (uint32_t*)alloc_aligned(capacity * sizeof(uint32_t));
        particle2 = (uint32_t*)alloc_aligned(capacity * sizeof(uint32_t));
        rest_length = (float*)alloc_aligned(capacity * sizeof(float));
        stiffness = (float*)alloc_aligned(capacity * sizeof(float));
        damping = (float*)alloc_aligned(capacity * sizeof(float));
        break_strain = (float*)alloc_aligned(capacity * sizeof(float));
        max_strain = (float*)alloc_aligned(capacity * sizeof(float));
        current_strain = (float*)alloc_aligned(capacity * sizeof(float));
        material_type = (uint8_t*)alloc_aligned(capacity * sizeof(uint8_t));
        is_broken = (uint8_t*)alloc_aligned(capacity * sizeof(uint8_t));
        alive = (bool*)alloc_aligned(capacity * sizeof(bool));
        thermal_conductivity = (float*)alloc_aligned(capacity * sizeof(float));
        damage = (float*)alloc_aligned(capacity * sizeof(float));
        
        free_list.reserve(capacity / 10);  // Expect ~10% breakage
    }
    
    void deallocate() {
        free(particle1); particle1 = nullptr;
        free(particle2); particle2 = nullptr;
        free(rest_length); rest_length = nullptr;
        free(stiffness); stiffness = nullptr;
        free(damping); damping = nullptr;
        free(break_strain); break_strain = nullptr;
        free(max_strain); max_strain = nullptr;
        free(current_strain); current_strain = nullptr;
        free(material_type); material_type = nullptr;
        free(is_broken); is_broken = nullptr;
        free(alive); alive = nullptr;
        free(thermal_conductivity); thermal_conductivity = nullptr;
        free(damage); damage = nullptr;
        capacity = 0;
        count = 0;
    }
    
    uint32_t add_spring(uint32_t p1, uint32_t p2, float rest_len, float stiff, float damp) {
        uint32_t idx;
        
        // Reuse broken spring slot if available
        if (!free_list.empty()) {
            idx = free_list.back();
            free_list.pop_back();
        } else {
            if (count >= capacity) return UINT32_MAX;
            idx = count++;
        }
        
        particle1[idx] = p1;
        particle2[idx] = p2;
        rest_length[idx] = rest_len;
        stiffness[idx] = stiff;
        damping[idx] = damp;
        break_strain[idx] = 0.5f;  // Default 50% strain limit
        current_strain[idx] = 0;
        material_type[idx] = 0;
        is_broken[idx] = 0;
        thermal_conductivity[idx] = 1.0f;
        damage[idx] = 0;
        
        return idx;
    }
    
    void break_spring(uint32_t idx) {
        is_broken[idx] = 1;
        free_list.push_back(idx);
    }
};

// Contact pool - temporary, rebuilt each frame
class ContactPool {
public:
    uint32_t* particle1 = nullptr;
    uint32_t* particle2 = nullptr;
    
    float* overlap = nullptr;
    float* normal_x = nullptr;
    float* normal_y = nullptr;
    float* contact_point_x = nullptr;
    float* contact_point_y = nullptr;
    
    size_t capacity = 0;
    size_t count = 0;
    
    void allocate(size_t max_contacts) {
        deallocate();
        capacity = max_contacts;
        
        auto alloc_aligned = [](size_t size) {
            void* ptr = nullptr;
            posix_memalign(&ptr, 64, size);
            return ptr;
        };
        
        particle1 = (uint32_t*)alloc_aligned(capacity * sizeof(uint32_t));
        particle2 = (uint32_t*)alloc_aligned(capacity * sizeof(uint32_t));
        overlap = (float*)alloc_aligned(capacity * sizeof(float));
        normal_x = (float*)alloc_aligned(capacity * sizeof(float));
        normal_y = (float*)alloc_aligned(capacity * sizeof(float));
        contact_point_x = (float*)alloc_aligned(capacity * sizeof(float));
        contact_point_y = (float*)alloc_aligned(capacity * sizeof(float));
    }
    
    void deallocate() {
        free(particle1); particle1 = nullptr;
        free(particle2); particle2 = nullptr;
        free(overlap); overlap = nullptr;
        free(normal_x); normal_x = nullptr;
        free(normal_y); normal_y = nullptr;
        free(contact_point_x); contact_point_x = nullptr;
        free(contact_point_y); contact_point_y = nullptr;
        capacity = 0;
        count = 0;
    }
    
    void clear() {
        count = 0;
    }
    
    void add_contact(uint32_t p1, uint32_t p2, float ovlp, 
                     float nx, float ny, float cx, float cy) {
        if (count >= capacity) return;
        
        particle1[count] = p1;
        particle2[count] = p2;
        overlap[count] = ovlp;
        normal_x[count] = nx;
        normal_y[count] = ny;
        contact_point_x[count] = cx;
        contact_point_y[count] = cy;
        count++;
    }
};

// Composite pool - identified groups of particles
class CompositePool {
public:
    // Composite properties
    float* center_x = nullptr;  // Current center position
    float* center_y = nullptr;
    float* center_of_mass_x = nullptr;
    float* center_of_mass_y = nullptr;
    float* velocity_x = nullptr;
    float* velocity_y = nullptr;
    float* angular_velocity = nullptr;
    float* total_mass = nullptr;
    float* bounding_radius = nullptr;
    uint32_t* particle_count = nullptr;  // Number of particles in composite
    
    // Member tracking (variable length per composite)
    uint32_t* member_start = nullptr;  // Index into member_particles
    uint32_t* member_count = nullptr;
    uint32_t* member_particles = nullptr;  // Flattened array of all members
    
    size_t capacity = 0;
    size_t count = 0;
    size_t member_capacity = 0;
    size_t member_total = 0;
    
    void allocate(size_t max_composites, size_t max_total_members) {
        deallocate();
        capacity = max_composites;
        member_capacity = max_total_members;
        
        auto alloc_aligned = [](size_t size) {
            void* ptr = nullptr;
            if (posix_memalign(&ptr, 64, size) != 0) {
                throw std::bad_alloc();
            }
            std::memset(ptr, 0, size);
            return ptr;
        };
        
        center_x = (float*)alloc_aligned(capacity * sizeof(float));
        center_y = (float*)alloc_aligned(capacity * sizeof(float));
        center_of_mass_x = (float*)alloc_aligned(capacity * sizeof(float));
        center_of_mass_y = (float*)alloc_aligned(capacity * sizeof(float));
        velocity_x = (float*)alloc_aligned(capacity * sizeof(float));
        velocity_y = (float*)alloc_aligned(capacity * sizeof(float));
        angular_velocity = (float*)alloc_aligned(capacity * sizeof(float));
        total_mass = (float*)alloc_aligned(capacity * sizeof(float));
        bounding_radius = (float*)alloc_aligned(capacity * sizeof(float));
        particle_count = (uint32_t*)alloc_aligned(capacity * sizeof(uint32_t));
        
        member_start = (uint32_t*)alloc_aligned(capacity * sizeof(uint32_t));
        member_count = (uint32_t*)alloc_aligned(capacity * sizeof(uint32_t));
        member_particles = (uint32_t*)alloc_aligned(member_capacity * sizeof(uint32_t));
    }
    
    void deallocate() {
        free(center_x); center_x = nullptr;
        free(center_y); center_y = nullptr;
        free(center_of_mass_x); center_of_mass_x = nullptr;
        free(center_of_mass_y); center_of_mass_y = nullptr;
        free(velocity_x); velocity_x = nullptr;
        free(velocity_y); velocity_y = nullptr;
        free(angular_velocity); angular_velocity = nullptr;
        free(total_mass); total_mass = nullptr;
        free(bounding_radius); bounding_radius = nullptr;
        free(particle_count); particle_count = nullptr;
        free(member_start); member_start = nullptr;
        free(member_count); member_count = nullptr;
        free(member_particles); member_particles = nullptr;
        capacity = 0;
        count = 0;
        member_capacity = 0;
        member_total = 0;
    }
    
    void clear() {
        count = 0;
        member_total = 0;
    }
};

// Field data structures
struct RadiationField {
    float* intensity = nullptr;
    float* direction_x = nullptr;
    float* direction_y = nullptr;
    size_t grid_size = 0;
    
    void allocate(size_t size) {
        grid_size = size;
        intensity = (float*)aligned_alloc(64, size * size * sizeof(float));
        direction_x = (float*)aligned_alloc(64, size * size * sizeof(float));
        direction_y = (float*)aligned_alloc(64, size * size * sizeof(float));
    }
    
    void deallocate() {
        free(intensity);
        free(direction_x);
        free(direction_y);
    }
};

struct ThermalField {
    float* temperature = nullptr;
    float* heat_flow = nullptr;
    size_t grid_size = 0;
    
    void allocate(size_t size) {
        grid_size = size;
        temperature = (float*)aligned_alloc(64, size * size * sizeof(float));
        heat_flow = (float*)aligned_alloc(64, size * size * sizeof(float));
    }
    
    void deallocate() {
        free(temperature);
        free(heat_flow);
    }
};

struct GravityField {
    // For PM solver
    float* density = nullptr;
    float* potential = nullptr;
    float* force_x = nullptr;
    float* force_y = nullptr;
    size_t grid_size = 0;
    
    void allocate(size_t size) {
        grid_size = size;
        size_t total = size * size;
        density = (float*)aligned_alloc(64, total * sizeof(float));
        potential = (float*)aligned_alloc(64, total * sizeof(float));
        force_x = (float*)aligned_alloc(64, total * sizeof(float));
        force_y = (float*)aligned_alloc(64, total * sizeof(float));
    }
    
    void deallocate() {
        free(density);
        free(potential);
        free(force_x);
        free(force_y);
    }
};

} // namespace digistar
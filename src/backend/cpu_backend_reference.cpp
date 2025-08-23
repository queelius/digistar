#include "cpu_backend_reference.h"
#include <cmath>
#include <cstring>
#include <algorithm>
#include <unordered_map>

namespace digistar {

CpuBackendReference::~CpuBackendReference() {
    shutdown();
}

void CpuBackendReference::initialize(const SimulationConfig& cfg) {
    config = cfg;
    
    // Allocate temporary storage for Velocity Verlet
    if (cfg.max_particles > 0) {
        old_force_x = (float*)aligned_alloc(64, cfg.max_particles * sizeof(float));
        old_force_y = (float*)aligned_alloc(64, cfg.max_particles * sizeof(float));
    }
    
    // Initialize FFTW for PM gravity if needed
    if (config.pm_grid_size > 0) {
        size_t grid_total = config.pm_grid_size * config.pm_grid_size;
        fft_workspace = (fftwf_complex*)fftwf_malloc(sizeof(fftwf_complex) * grid_total);
        
        // Create FFTW plans (using FFTW_MEASURE for best performance)
        fft_forward = fftwf_plan_dft_2d(config.pm_grid_size, config.pm_grid_size,
                                        fft_workspace, fft_workspace,
                                        FFTW_FORWARD, FFTW_MEASURE);
        fft_inverse = fftwf_plan_dft_2d(config.pm_grid_size, config.pm_grid_size,
                                        fft_workspace, fft_workspace,
                                        FFTW_BACKWARD, FFTW_MEASURE);
    }
}

void CpuBackendReference::shutdown() {
    if (old_force_x) {
        free(old_force_x);
        old_force_x = nullptr;
    }
    if (old_force_y) {
        free(old_force_y);
        old_force_y = nullptr;
    }
    
    if (fft_forward) {
        fftwf_destroy_plan(fft_forward);
        fft_forward = nullptr;
    }
    if (fft_inverse) {
        fftwf_destroy_plan(fft_inverse);
        fft_inverse = nullptr;
    }
    if (fft_workspace) {
        fftwf_free(fft_workspace);
        fft_workspace = nullptr;
    }
}

void CpuBackendReference::step(SimulationState& state, const PhysicsConfig& physics_config, float dt) {
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // 1. Update spatial indices
    auto t1 = std::chrono::high_resolution_clock::now();
    
    // Update contact index (finest resolution)
    if (state.contact_index && (physics_config.enabled_systems & PhysicsConfig::CONTACTS)) {
        state.contact_index->clear();
        for (size_t i = 0; i < state.particles.count; i++) {
            state.contact_index->insert(i, state.particles.pos_x[i], state.particles.pos_y[i]);
        }
    }
    
    // Update spring index (medium resolution)
    if (state.spring_index && (physics_config.enabled_systems & PhysicsConfig::SPRING_FIELD)) {
        state.spring_index->clear();
        for (size_t i = 0; i < state.particles.count; i++) {
            state.spring_index->insert(i, state.particles.pos_x[i], state.particles.pos_y[i]);
        }
    }
    
    auto t2 = std::chrono::high_resolution_clock::now();
    stats.spatial_index_time = std::chrono::duration<float, std::milli>(t2 - t1).count();
    
    // 2. Detect contacts
    if (physics_config.enabled_systems & PhysicsConfig::CONTACTS) {
        state.contacts.clear();
        detectContacts(state.particles, state.contacts, *state.contact_index);
    }
    
    auto t3 = std::chrono::high_resolution_clock::now();
    stats.collision_detection_time = std::chrono::duration<float, std::milli>(t3 - t2).count();
    
    // 3. Update composites (connected components via springs)
    if (physics_config.enabled_systems & PhysicsConfig::SPRINGS) {
        updateComposites(state.particles, state.springs, state.composites);
    }
    
    auto t4 = std::chrono::high_resolution_clock::now();
    stats.composite_detection_time = std::chrono::duration<float, std::milli>(t4 - t3).count();
    
    // 4. Store old forces if using Velocity Verlet
    if (physics_config.default_integrator == PhysicsConfig::VELOCITY_VERLET) {
        std::memcpy(old_force_x, state.particles.force_x, state.particles.count * sizeof(float));
        std::memcpy(old_force_y, state.particles.force_y, state.particles.count * sizeof(float));
    }
    
    // 5. Clear forces for new calculation
    state.particles.clear_forces();
    
    // 6. Compute all forces and interactions
    
    // Gravity
    if (physics_config.enabled_systems & PhysicsConfig::GRAVITY) {
        switch (physics_config.gravity_mode) {
            case PhysicsConfig::DIRECT_N2:
                computeGravityDirect(state.particles);
                break;
            case PhysicsConfig::PARTICLE_MESH:
                computeGravityPM(state.particles, state.gravity);
                break;
            case PhysicsConfig::BARNES_HUT:
                computeGravityBarnesHut(state.particles);
                break;
        }
    }
    
    // Contact forces
    if (physics_config.enabled_systems & PhysicsConfig::CONTACTS) {
        computeContacts(state.particles, state.contacts);
    }
    
    // Spring forces
    if (physics_config.enabled_systems & PhysicsConfig::SPRINGS) {
        computeSprings(state.particles, state.springs);
        checkSpringBreaking(state.springs, state.particles);
    }
    
    // Virtual spring field (formation of new springs)
    if (physics_config.enabled_systems & PhysicsConfig::SPRING_FIELD) {
        formNewSprings(state.particles, state.springs, *state.spring_index);
    }
    
    // Thermal conduction through springs
    if (physics_config.enabled_systems & PhysicsConfig::THERMAL) {
        computeThermal(state.particles, state.springs);
    }
    
    // Radiation
    if (physics_config.enabled_systems & PhysicsConfig::RADIATION) {
        computeRadiation(state.particles, state.radiation, *state.radiation_index);
    }
    
    auto t5 = std::chrono::high_resolution_clock::now();
    stats.update_time = std::chrono::duration<float, std::milli>(t5 - t4).count();
    
    // 7. Integrate positions and velocities
    switch (physics_config.default_integrator) {
        case PhysicsConfig::VELOCITY_VERLET:
            integrateVelocityVerlet(state.particles, dt);
            break;
        case PhysicsConfig::SEMI_IMPLICIT:
            integrateSemiImplicit(state.particles, dt);
            break;
        default:
            integrateSemiImplicit(state.particles, dt);
            break;
    }
    
    auto t6 = std::chrono::high_resolution_clock::now();
    stats.integrate_time = std::chrono::duration<float, std::milli>(t6 - t5).count();
    
    // 8. Update statistics
    stats.active_particles = state.particles.count;
    stats.active_springs = state.springs.count;
    stats.active_contacts = state.contacts.count;
    stats.active_composites = state.composites.count;
    
    // Calculate total energy (for monitoring)
    stats.total_energy = 0;
    stats.max_velocity = 0;
    for (size_t i = 0; i < state.particles.count; i++) {
        float v2 = state.particles.vel_x[i] * state.particles.vel_x[i] + 
                   state.particles.vel_y[i] * state.particles.vel_y[i];
        stats.total_energy += 0.5f * state.particles.mass[i] * v2;
        stats.max_velocity = std::max(stats.max_velocity, sqrtf(v2));
    }
}

void CpuBackendReference::computeGravityDirect(ParticlePool& particles) {
    const float G = 6.67430e-11f;  // Real units
    const float softening = 0.1f;  // Prevent singularities
    
    // Simple O(N²) implementation - clear and correct
    for (size_t i = 0; i < particles.count; i++) {
        float fx = 0, fy = 0;
        
        for (size_t j = 0; j < particles.count; j++) {
            if (i == j) continue;
            
            float dx = particles.pos_x[j] - particles.pos_x[i];
            float dy = particles.pos_y[j] - particles.pos_y[i];
            
            // Handle toroidal wrapping if enabled
            if (config.use_toroidal) {
                if (dx > config.world_size * 0.5f) dx -= config.world_size;
                if (dx < -config.world_size * 0.5f) dx += config.world_size;
                if (dy > config.world_size * 0.5f) dy -= config.world_size;
                if (dy < -config.world_size * 0.5f) dy += config.world_size;
            }
            
            float r2 = dx * dx + dy * dy + softening * softening;
            float r = sqrtf(r2);
            
            // F = G * m1 * m2 / r² in direction of r
            float force_magnitude = G * particles.mass[j] / r2;
            
            fx += force_magnitude * dx / r;
            fy += force_magnitude * dy / r;
        }
        
        particles.force_x[i] += fx * particles.mass[i];
        particles.force_y[i] += fy * particles.mass[i];
    }
}

void CpuBackendReference::computeGravityPM(ParticlePool& particles, GravityField& field) {
    const size_t grid_size = config.pm_grid_size;
    const float cell_size = config.world_size / grid_size;
    const float G = 6.67430e-11f;
    
    // Step 1: Clear density grid
    std::memset(field.density, 0, grid_size * grid_size * sizeof(float));
    
    // Step 2: Deposit mass onto grid using CIC (Cloud-In-Cell)
    for (size_t i = 0; i < particles.count; i++) {
        // Particle position in grid coordinates
        float x = particles.pos_x[i] / cell_size;
        float y = particles.pos_y[i] / cell_size;
        
        // Find grid cell
        int ix = (int)floorf(x);
        int iy = (int)floorf(y);
        
        // Fractional position within cell
        float fx = x - ix;
        float fy = y - iy;
        
        // Wrap indices for periodic boundaries
        ix = ((ix % (int)grid_size) + grid_size) % grid_size;
        iy = ((iy % (int)grid_size) + grid_size) % grid_size;
        int ix1 = (ix + 1) % grid_size;
        int iy1 = (iy + 1) % grid_size;
        
        // CIC weights (bilinear interpolation)
        float w00 = (1.0f - fx) * (1.0f - fy);
        float w10 = fx * (1.0f - fy);
        float w01 = (1.0f - fx) * fy;
        float w11 = fx * fy;
        
        // Deposit mass
        field.density[iy * grid_size + ix] += w00 * particles.mass[i];
        field.density[iy * grid_size + ix1] += w10 * particles.mass[i];
        field.density[iy1 * grid_size + ix] += w01 * particles.mass[i];
        field.density[iy1 * grid_size + ix1] += w11 * particles.mass[i];
    }
    
    // Step 3: Copy density to FFT workspace
    for (size_t i = 0; i < grid_size * grid_size; i++) {
        fft_workspace[i][0] = field.density[i];
        fft_workspace[i][1] = 0;
    }
    
    // Step 4: Forward FFT
    fftwf_execute(fft_forward);
    
    // Step 5: Apply Green's function in Fourier space
    const float k_scale = 2.0f * M_PI / config.world_size;
    const float softening = 0.1f;
    
    for (size_t ky = 0; ky < grid_size; ky++) {
        for (size_t kx = 0; kx < grid_size; kx++) {
            size_t idx = ky * grid_size + kx;
            
            // Wave numbers (handle aliasing)
            float kx_val = (kx <= grid_size/2) ? kx : (int)kx - (int)grid_size;
            float ky_val = (ky <= grid_size/2) ? ky : (int)ky - (int)grid_size;
            kx_val *= k_scale;
            ky_val *= k_scale;
            
            float k2 = kx_val * kx_val + ky_val * ky_val;
            
            if (k2 > 0) {
                // Green's function for Poisson equation: G(k) = -4πG / k²
                float green = -4.0f * M_PI * G / (k2 + softening * softening * k_scale * k_scale);
                
                // Multiply by Green's function
                float real = fft_workspace[idx][0];
                float imag = fft_workspace[idx][1];
                fft_workspace[idx][0] = real * green;
                fft_workspace[idx][1] = imag * green;
            } else {
                // Zero frequency component (average density)
                fft_workspace[idx][0] = 0;
                fft_workspace[idx][1] = 0;
            }
        }
    }
    
    // Step 6: Inverse FFT to get potential
    fftwf_execute(fft_inverse);
    
    // Step 7: Normalize and compute forces via finite differences
    float norm = 1.0f / (grid_size * grid_size);
    
    for (size_t iy = 0; iy < grid_size; iy++) {
        for (size_t ix = 0; ix < grid_size; ix++) {
            size_t idx = iy * grid_size + ix;
            
            // Store normalized potential
            field.potential[idx] = fft_workspace[idx][0] * norm;
            
            // Compute force via finite differences
            size_t ix_plus = (ix + 1) % grid_size;
            size_t ix_minus = (ix + grid_size - 1) % grid_size;
            size_t iy_plus = (iy + 1) % grid_size;
            size_t iy_minus = (iy + grid_size - 1) % grid_size;
            
            float dphi_dx = (fft_workspace[iy * grid_size + ix_plus][0] - 
                            fft_workspace[iy * grid_size + ix_minus][0]) * norm / (2.0f * cell_size);
            float dphi_dy = (fft_workspace[iy_plus * grid_size + ix][0] - 
                            fft_workspace[iy_minus * grid_size + ix][0]) * norm / (2.0f * cell_size);
            
            field.force_x[idx] = -dphi_dx;
            field.force_y[idx] = -dphi_dy;
        }
    }
    
    // Step 8: Interpolate forces back to particles (CIC)
    for (size_t i = 0; i < particles.count; i++) {
        float x = particles.pos_x[i] / cell_size;
        float y = particles.pos_y[i] / cell_size;
        
        int ix = (int)floorf(x);
        int iy = (int)floorf(y);
        float fx = x - ix;
        float fy = y - iy;
        
        ix = ((ix % (int)grid_size) + grid_size) % grid_size;
        iy = ((iy % (int)grid_size) + grid_size) % grid_size;
        int ix1 = (ix + 1) % grid_size;
        int iy1 = (iy + 1) % grid_size;
        
        // Interpolate force
        float force_x = 
            field.force_x[iy * grid_size + ix] * (1-fx) * (1-fy) +
            field.force_x[iy * grid_size + ix1] * fx * (1-fy) +
            field.force_x[iy1 * grid_size + ix] * (1-fx) * fy +
            field.force_x[iy1 * grid_size + ix1] * fx * fy;
            
        float force_y = 
            field.force_y[iy * grid_size + ix] * (1-fx) * (1-fy) +
            field.force_y[iy * grid_size + ix1] * fx * (1-fy) +
            field.force_y[iy1 * grid_size + ix] * (1-fx) * fy +
            field.force_y[iy1 * grid_size + ix1] * fx * fy;
        
        particles.force_x[i] += force_x * particles.mass[i];
        particles.force_y[i] += force_y * particles.mass[i];
    }
}

void CpuBackendReference::computeGravityBarnesHut(ParticlePool& particles) {
    // TODO: Implement Barnes-Hut tree algorithm
    // For now, fall back to direct
    computeGravityDirect(particles);
}

void CpuBackendReference::computeContacts(ParticlePool& particles, ContactPool& contacts) {
    const float k_contact = 1000.0f;  // Contact stiffness
    const float damping = 10.0f;      // Contact damping
    
    // Process all detected contacts
    for (size_t c = 0; c < contacts.count; c++) {
        uint32_t i = contacts.particle1[c];
        uint32_t j = contacts.particle2[c];
        
        // Hertzian contact model: F = k * δ^(3/2)
        float overlap = contacts.overlap[c];
        float force_magnitude = k_contact * powf(overlap, 1.5f);
        
        // Velocity-dependent damping
        float vx_rel = particles.vel_x[j] - particles.vel_x[i];
        float vy_rel = particles.vel_y[j] - particles.vel_y[i];
        
        // Project relative velocity onto contact normal
        float v_normal = vx_rel * contacts.normal_x[c] + vy_rel * contacts.normal_y[c];
        
        // Add damping force
        force_magnitude += damping * v_normal * sqrtf(overlap);
        
        // Apply equal and opposite forces
        float fx = force_magnitude * contacts.normal_x[c];
        float fy = force_magnitude * contacts.normal_y[c];
        
        particles.force_x[i] -= fx;
        particles.force_y[i] -= fy;
        particles.force_x[j] += fx;
        particles.force_y[j] += fy;
        
        // Generate heat from collision
        if (v_normal > 0) {  // Only if compressing
            float heat_generated = 0.01f * damping * v_normal * v_normal * overlap;
            particles.temp_internal[i] += heat_generated / (2.0f * particles.mass[i]);
            particles.temp_internal[j] += heat_generated / (2.0f * particles.mass[j]);
        }
    }
}

void CpuBackendReference::computeSprings(ParticlePool& particles, SpringPool& springs) {
    // Process all active springs
    for (size_t s = 0; s < springs.count; s++) {
        if (springs.is_broken[s]) continue;
        
        uint32_t i = springs.particle1[s];
        uint32_t j = springs.particle2[s];
        
        // Vector from particle i to particle j
        float dx = particles.pos_x[j] - particles.pos_x[i];
        float dy = particles.pos_y[j] - particles.pos_y[i];
        
        // Handle toroidal wrapping
        if (config.use_toroidal) {
            if (dx > config.world_size * 0.5f) dx -= config.world_size;
            if (dx < -config.world_size * 0.5f) dx += config.world_size;
            if (dy > config.world_size * 0.5f) dy -= config.world_size;
            if (dy < -config.world_size * 0.5f) dy += config.world_size;
        }
        
        float distance = sqrtf(dx * dx + dy * dy);
        if (distance < 1e-6f) continue;  // Avoid division by zero
        
        // Calculate strain
        float strain = (distance - springs.rest_length[s]) / springs.rest_length[s];
        springs.current_strain[s] = strain;
        
        // Hooke's law: F = -k * Δx
        float spring_force = springs.stiffness[s] * (distance - springs.rest_length[s]);
        
        // Damping force
        float vx_rel = particles.vel_x[j] - particles.vel_x[i];
        float vy_rel = particles.vel_y[j] - particles.vel_y[i];
        float v_along_spring = (vx_rel * dx + vy_rel * dy) / distance;
        float damping_force = springs.damping[s] * v_along_spring;
        
        // Total force along spring
        float total_force = (spring_force + damping_force) / distance;
        
        // Apply forces
        float fx = total_force * dx;
        float fy = total_force * dy;
        
        particles.force_x[i] += fx;
        particles.force_y[i] += fy;
        particles.force_x[j] -= fx;
        particles.force_y[j] -= fy;
        
        // Accumulate damage if yielding
        if (fabsf(strain) > 0.2f) {  // Plastic deformation threshold
            springs.damage[s] += fabsf(strain) * 0.01f;
        }
    }
}

void CpuBackendReference::checkSpringBreaking(SpringPool& springs, ParticlePool& particles) {
    for (size_t s = 0; s < springs.count; s++) {
        if (springs.is_broken[s]) continue;
        
        // Check strain limit
        if (fabsf(springs.current_strain[s]) > springs.break_strain[s]) {
            springs.is_broken[s] = 1;
            
            // Release elastic energy as heat
            float energy = 0.5f * springs.stiffness[s] * 
                          springs.current_strain[s] * springs.current_strain[s] * 
                          springs.rest_length[s] * springs.rest_length[s];
            
            uint32_t i = springs.particle1[s];
            uint32_t j = springs.particle2[s];
            
            particles.temp_internal[i] += energy / (2.0f * particles.mass[i]);
            particles.temp_internal[j] += energy / (2.0f * particles.mass[j]);
        }
        
        // Check accumulated damage
        if (springs.damage[s] > 1.0f) {
            springs.is_broken[s] = 1;
        }
    }
}

void CpuBackendReference::formNewSprings(ParticlePool& particles, SpringPool& springs, SpatialIndex& index) {
    const float formation_distance = 5.0f;  // Maximum distance for spring formation
    const float max_relative_velocity = 1.0f;  // Maximum relative velocity
    
    // Check each particle's neighborhood
    for (size_t i = 0; i < particles.count; i++) {
        auto neighbors = index.query_radius(particles.pos_x[i], particles.pos_y[i], 
                                           formation_distance);
        
        for (uint32_t j : neighbors) {
            if (j <= i) continue;  // Avoid duplicates
            
            // Check if spring already exists
            bool exists = false;
            for (size_t s = 0; s < springs.count; s++) {
                if ((springs.particle1[s] == i && springs.particle2[s] == j) ||
                    (springs.particle1[s] == j && springs.particle2[s] == i)) {
                    exists = true;
                    break;
                }
            }
            if (exists) continue;
            
            // Check distance
            float dx = particles.pos_x[j] - particles.pos_x[i];
            float dy = particles.pos_y[j] - particles.pos_y[i];
            float dist = sqrtf(dx * dx + dy * dy);
            
            if (dist > formation_distance) continue;
            
            // Check relative velocity
            float vx_rel = particles.vel_x[j] - particles.vel_x[i];
            float vy_rel = particles.vel_y[j] - particles.vel_y[i];
            float v_rel = sqrtf(vx_rel * vx_rel + vy_rel * vy_rel);
            
            if (v_rel > max_relative_velocity) continue;
            
            // Check material compatibility
            if (particles.material_type[i] == particles.material_type[j]) {
                // Form spring with material-appropriate properties
                float stiffness = 100.0f;
                float damping = 1.0f;
                
                if (particles.material_type[i] == MATERIAL_METAL) {
                    stiffness = 500.0f;
                    damping = 5.0f;
                } else if (particles.material_type[i] == MATERIAL_ORGANIC) {
                    stiffness = 50.0f;
                    damping = 2.0f;
                }
                
                springs.add_spring(i, j, dist, stiffness, damping);
            }
        }
    }
}

void CpuBackendReference::computeThermal(ParticlePool& particles, SpringPool& springs) {
    const float dt = 0.01f;  // Small timestep for thermal diffusion
    
    // Heat conduction through springs (Fourier's law)
    for (size_t s = 0; s < springs.count; s++) {
        if (springs.is_broken[s]) continue;
        
        uint32_t i = springs.particle1[s];
        uint32_t j = springs.particle2[s];
        
        float temp_diff = particles.temp_internal[j] - particles.temp_internal[i];
        
        // Q = k * A * ΔT / L
        float heat_flow = springs.thermal_conductivity[s] * temp_diff * dt / springs.rest_length[s];
        
        // Update temperatures
        particles.temp_internal[i] += heat_flow / particles.mass[i];
        particles.temp_internal[j] -= heat_flow / particles.mass[j];
    }
}

void CpuBackendReference::computeRadiation(ParticlePool& particles, RadiationField& field, SpatialIndex& index) {
    // Simple radiation model - hot particles emit, all particles absorb
    const float stefan_boltzmann = 5.67e-8f;
    const float emissivity = 0.8f;
    
    for (size_t i = 0; i < particles.count; i++) {
        if (particles.temp_internal[i] > 300.0f) {  // Only radiate if hot
            // Stefan-Boltzmann law: P = ε * σ * A * T^4
            // In 2D, use T^3 instead
            float power = emissivity * stefan_boltzmann * 
                         powf(particles.temp_internal[i], 3.0f) * 
                         particles.radius[i] * 2.0f * M_PI;
            
            // Cool the emitting particle
            particles.temp_internal[i] -= power * 0.01f / particles.mass[i];
            
            // Heat nearby particles (simplified - not physically accurate)
            auto neighbors = index.query_radius(particles.pos_x[i], particles.pos_y[i], 100.0f);
            
            for (uint32_t j : neighbors) {
                if (j == i) continue;
                
                float dx = particles.pos_x[j] - particles.pos_x[i];
                float dy = particles.pos_y[j] - particles.pos_y[i];
                float dist2 = dx * dx + dy * dy;
                
                if (dist2 > 1.0f) {
                    // Intensity falls off with 1/r in 2D
                    float received_power = power / (2.0f * M_PI * sqrtf(dist2));
                    particles.temp_internal[j] += received_power * 0.01f / particles.mass[j];
                }
            }
        }
    }
}

void CpuBackendReference::detectContacts(ParticlePool& particles, ContactPool& contacts, SpatialIndex& index) {
    // Use spatial index to find potentially colliding pairs
    const auto& grid = static_cast<const SparseGrid&>(index);
    
    for (const auto& [cell_hash, particle_ids] : grid.get_cells()) {
        // Check all pairs within the same cell
        for (size_t i = 0; i < particle_ids.size(); i++) {
            for (size_t j = i + 1; j < particle_ids.size(); j++) {
                uint32_t pi = particle_ids[i];
                uint32_t pj = particle_ids[j];
                
                float dx = particles.pos_x[pj] - particles.pos_x[pi];
                float dy = particles.pos_y[pj] - particles.pos_y[pi];
                
                // Handle toroidal wrapping
                if (config.use_toroidal) {
                    if (dx > config.world_size * 0.5f) dx -= config.world_size;
                    if (dx < -config.world_size * 0.5f) dx += config.world_size;
                    if (dy > config.world_size * 0.5f) dy -= config.world_size;
                    if (dy < -config.world_size * 0.5f) dy += config.world_size;
                }
                
                float dist2 = dx * dx + dy * dy;
                float min_dist = particles.radius[pi] + particles.radius[pj];
                
                if (dist2 < min_dist * min_dist && dist2 > 1e-12f) {
                    float dist = sqrtf(dist2);
                    float overlap = min_dist - dist;
                    
                    // Normal vector from i to j
                    float nx = dx / dist;
                    float ny = dy / dist;
                    
                    // Contact point (midpoint of overlap)
                    float cx = particles.pos_x[pi] + nx * (particles.radius[pi] - overlap * 0.5f);
                    float cy = particles.pos_y[pi] + ny * (particles.radius[pi] - overlap * 0.5f);
                    
                    contacts.add_contact(pi, pj, overlap, nx, ny, cx, cy);
                }
            }
        }
        
        // Also check with neighboring cells
        auto neighbor_cells = index.get_neighbor_cells(
            particles.pos_x[particle_ids[0]], 
            particles.pos_y[particle_ids[0]]
        );
        
        for (uint64_t neighbor_hash : neighbor_cells) {
            if (neighbor_hash <= cell_hash) continue;  // Avoid checking pairs twice
            
            auto neighbor_particles = index.query_cell(
                particles.pos_x[particle_ids[0]], 
                particles.pos_y[particle_ids[0]]
            );
            
            for (uint32_t pi : particle_ids) {
                for (uint32_t pj : neighbor_particles) {
                    // Same contact detection logic as above
                    float dx = particles.pos_x[pj] - particles.pos_x[pi];
                    float dy = particles.pos_y[pj] - particles.pos_y[pi];
                    
                    if (config.use_toroidal) {
                        if (dx > config.world_size * 0.5f) dx -= config.world_size;
                        if (dx < -config.world_size * 0.5f) dx += config.world_size;
                        if (dy > config.world_size * 0.5f) dy -= config.world_size;
                        if (dy < -config.world_size * 0.5f) dy += config.world_size;
                    }
                    
                    float dist2 = dx * dx + dy * dy;
                    float min_dist = particles.radius[pi] + particles.radius[pj];
                    
                    if (dist2 < min_dist * min_dist && dist2 > 1e-12f) {
                        float dist = sqrtf(dist2);
                        float overlap = min_dist - dist;
                        float nx = dx / dist;
                        float ny = dy / dist;
                        float cx = particles.pos_x[pi] + nx * (particles.radius[pi] - overlap * 0.5f);
                        float cy = particles.pos_y[pi] + ny * (particles.radius[pi] - overlap * 0.5f);
                        
                        contacts.add_contact(pi, pj, overlap, nx, ny, cx, cy);
                    }
                }
            }
        }
    }
}

void CpuBackendReference::updateComposites(ParticlePool& particles, SpringPool& springs, 
                                          CompositePool& composites) {
    // Use Union-Find to detect connected components
    UnionFind uf;
    uf.reset(particles.count);
    
    // Unite particles connected by unbroken springs
    for (size_t s = 0; s < springs.count; s++) {
        if (!springs.is_broken[s]) {
            uf.unite(springs.particle1[s], springs.particle2[s]);
        }
    }
    
    // Group particles by their root
    std::unordered_map<uint32_t, std::vector<uint32_t>> groups;
    for (size_t i = 0; i < particles.count; i++) {
        uint32_t root = uf.find(i);
        groups[root].push_back(i);
        particles.composite_id[i] = root;  // Store composite ID in particle
    }
    
    // Clear and rebuild composite pool
    composites.clear();
    
    for (const auto& [root, members] : groups) {
        if (members.size() <= 1) continue;  // Skip singletons
        
        // Calculate composite properties
        float com_x = 0, com_y = 0;
        float vel_x = 0, vel_y = 0;
        float total_mass = 0;
        float max_radius = 0;
        
        for (uint32_t id : members) {
            float m = particles.mass[id];
            com_x += particles.pos_x[id] * m;
            com_y += particles.pos_y[id] * m;
            vel_x += particles.vel_x[id] * m;
            vel_y += particles.vel_y[id] * m;
            total_mass += m;
        }
        
        com_x /= total_mass;
        com_y /= total_mass;
        vel_x /= total_mass;
        vel_y /= total_mass;
        
        // Calculate bounding radius
        for (uint32_t id : members) {
            float dx = particles.pos_x[id] - com_x;
            float dy = particles.pos_y[id] - com_y;
            float r = sqrtf(dx * dx + dy * dy) + particles.radius[id];
            max_radius = std::max(max_radius, r);
        }
        
        // Add composite to pool
        if (composites.count < composites.capacity) {
            size_t idx = composites.count++;
            composites.center_of_mass_x[idx] = com_x;
            composites.center_of_mass_y[idx] = com_y;
            composites.velocity_x[idx] = vel_x;
            composites.velocity_y[idx] = vel_y;
            composites.total_mass[idx] = total_mass;
            composites.bounding_radius[idx] = max_radius;
            composites.member_count[idx] = members.size();
            
            // Copy member IDs
            composites.member_start[idx] = composites.member_total;
            for (uint32_t id : members) {
                if (composites.member_total < composites.member_capacity) {
                    composites.member_particles[composites.member_total++] = id;
                }
            }
        }
    }
}

void CpuBackendReference::integrateSemiImplicit(ParticlePool& particles, float dt) {
    for (size_t i = 0; i < particles.count; i++) {
        // Update velocity first (makes it semi-implicit)
        particles.vel_x[i] += (particles.force_x[i] / particles.mass[i]) * dt;
        particles.vel_y[i] += (particles.force_y[i] / particles.mass[i]) * dt;
        
        // Clamp velocity to prevent numerical explosions
        float v2 = particles.vel_x[i] * particles.vel_x[i] + 
                   particles.vel_y[i] * particles.vel_y[i];
        if (v2 > Constants::MAX_VELOCITY * Constants::MAX_VELOCITY) {
            float scale = Constants::MAX_VELOCITY / sqrtf(v2);
            particles.vel_x[i] *= scale;
            particles.vel_y[i] *= scale;
        }
        
        // Update position using new velocity
        particles.pos_x[i] += particles.vel_x[i] * dt;
        particles.pos_y[i] += particles.vel_y[i] * dt;
        
        // Handle boundary conditions
        if (config.use_toroidal) {
            // Wrap positions
            while (particles.pos_x[i] < 0) particles.pos_x[i] += config.world_size;
            while (particles.pos_x[i] >= config.world_size) particles.pos_x[i] -= config.world_size;
            while (particles.pos_y[i] < 0) particles.pos_y[i] += config.world_size;
            while (particles.pos_y[i] >= config.world_size) particles.pos_y[i] -= config.world_size;
        } else {
            // Reflective boundaries
            if (particles.pos_x[i] < particles.radius[i]) {
                particles.pos_x[i] = particles.radius[i];
                particles.vel_x[i] = -particles.vel_x[i] * 0.8f;  // Some energy loss
            }
            if (particles.pos_x[i] > config.world_size - particles.radius[i]) {
                particles.pos_x[i] = config.world_size - particles.radius[i];
                particles.vel_x[i] = -particles.vel_x[i] * 0.8f;
            }
            if (particles.pos_y[i] < particles.radius[i]) {
                particles.pos_y[i] = particles.radius[i];
                particles.vel_y[i] = -particles.vel_y[i] * 0.8f;
            }
            if (particles.pos_y[i] > config.world_size - particles.radius[i]) {
                particles.pos_y[i] = config.world_size - particles.radius[i];
                particles.vel_y[i] = -particles.vel_y[i] * 0.8f;
            }
        }
    }
}

void CpuBackendReference::integrateVelocityVerlet(ParticlePool& particles, float dt) {
    // Velocity Verlet is a symplectic integrator that conserves energy better
    
    for (size_t i = 0; i < particles.count; i++) {
        // x(t+dt) = x(t) + v(t)*dt + 0.5*a(t)*dt²
        float ax_old = old_force_x[i] / particles.mass[i];
        float ay_old = old_force_y[i] / particles.mass[i];
        
        particles.pos_x[i] += particles.vel_x[i] * dt + 0.5f * ax_old * dt * dt;
        particles.pos_y[i] += particles.vel_y[i] * dt + 0.5f * ay_old * dt * dt;
        
        // v(t+dt) = v(t) + 0.5*(a(t) + a(t+dt))*dt
        float ax_new = particles.force_x[i] / particles.mass[i];
        float ay_new = particles.force_y[i] / particles.mass[i];
        
        particles.vel_x[i] += 0.5f * (ax_old + ax_new) * dt;
        particles.vel_y[i] += 0.5f * (ay_old + ay_new) * dt;
        
        // Handle boundaries
        if (config.use_toroidal) {
            while (particles.pos_x[i] < 0) particles.pos_x[i] += config.world_size;
            while (particles.pos_x[i] >= config.world_size) particles.pos_x[i] -= config.world_size;
            while (particles.pos_y[i] < 0) particles.pos_y[i] += config.world_size;
            while (particles.pos_y[i] >= config.world_size) particles.pos_y[i] -= config.world_size;
        }
    }
}

void CpuBackendReference::integrateLeapfrog(ParticlePool& particles, float dt) {
    // Leapfrog is another symplectic integrator
    for (size_t i = 0; i < particles.count; i++) {
        // v(t+dt/2) = v(t-dt/2) + a(t)*dt
        particles.vel_x[i] += (particles.force_x[i] / particles.mass[i]) * dt;
        particles.vel_y[i] += (particles.force_y[i] / particles.mass[i]) * dt;
        
        // x(t+dt) = x(t) + v(t+dt/2)*dt
        particles.pos_x[i] += particles.vel_x[i] * dt;
        particles.pos_y[i] += particles.vel_y[i] * dt;
        
        // Handle boundaries
        if (config.use_toroidal) {
            while (particles.pos_x[i] < 0) particles.pos_x[i] += config.world_size;
            while (particles.pos_x[i] >= config.world_size) particles.pos_x[i] -= config.world_size;
            while (particles.pos_y[i] < 0) particles.pos_y[i] += config.world_size;
            while (particles.pos_y[i] >= config.world_size) particles.pos_y[i] -= config.world_size;
        }
    }
}

void CpuBackendReference::integrateForwardEuler(ParticlePool& particles, float dt) {
    // Simple forward Euler - included for comparison (not recommended for production)
    for (size_t i = 0; i < particles.count; i++) {
        // x(t+dt) = x(t) + v(t)*dt
        particles.pos_x[i] += particles.vel_x[i] * dt;
        particles.pos_y[i] += particles.vel_y[i] * dt;
        
        // v(t+dt) = v(t) + a(t)*dt
        particles.vel_x[i] += (particles.force_x[i] / particles.mass[i]) * dt;
        particles.vel_y[i] += (particles.force_y[i] / particles.mass[i]) * dt;
        
        // Handle boundaries
        if (config.use_toroidal) {
            while (particles.pos_x[i] < 0) particles.pos_x[i] += config.world_size;
            while (particles.pos_x[i] >= config.world_size) particles.pos_x[i] -= config.world_size;
            while (particles.pos_y[i] < 0) particles.pos_y[i] += config.world_size;
            while (particles.pos_y[i] >= config.world_size) particles.pos_y[i] -= config.world_size;
        }
    }
}

} // namespace digistar
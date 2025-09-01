#include "cpu_backend_openmp.h"
#include <omp.h>
#include <cmath>
#include <cstring>
#include <algorithm>

namespace digistar {

CpuBackendOpenMP::~CpuBackendOpenMP() {
    shutdown();
}

void CpuBackendOpenMP::initialize(const SimulationConfig& cfg) {
    config = cfg;
    
    // Set OpenMP threads
    if (config.num_threads > 0) {
        omp_set_num_threads(config.num_threads);
    }
    
    // Initialize FFTW for PM gravity if needed
    if (config.pm_grid_size > 0) {
        size_t grid_total = config.pm_grid_size * config.pm_grid_size;
        fft_workspace = (fftwf_complex*)fftwf_malloc(sizeof(fftwf_complex) * grid_total);
        
        // Create FFTW plans
        fft_forward = fftwf_plan_dft_2d(config.pm_grid_size, config.pm_grid_size,
                                        fft_workspace, fft_workspace,
                                        FFTW_FORWARD, FFTW_ESTIMATE);
        fft_inverse = fftwf_plan_dft_2d(config.pm_grid_size, config.pm_grid_size,
                                        fft_workspace, fft_workspace,
                                        FFTW_BACKWARD, FFTW_ESTIMATE);
    }
}

void CpuBackendOpenMP::shutdown() {
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

void CpuBackendOpenMP::step(SimulationState& state, const PhysicsConfig& physics_config, float dt) {
    last_step_start = std::chrono::high_resolution_clock::now();
    
    // Update spatial indices (incremental)
    auto t1 = std::chrono::high_resolution_clock::now();
    
    #pragma omp parallel sections
    {
        #pragma omp section
        if (state.contact_index) {
            for (size_t i = 0; i < state.particles.count; i++) {
                uint32_t idx = state.particles.active_indices[i];
                // Only update if particle moved significantly
                // (implementation would track previous positions)
            }
        }
        
        #pragma omp section
        if (state.spring_index) {
            // Similar incremental update
        }
    }
    
    auto t2 = std::chrono::high_resolution_clock::now();
    stats.spatial_index_time = std::chrono::duration<float, std::milli>(t2 - t1).count();
    
    // Detect collisions
    if (physics_config.enabled_systems & PhysicsConfig::CONTACTS) {
        state.contacts.clear();
        detectContacts(state.particles, state.contacts, *state.contact_index);
    }
    
    auto t3 = std::chrono::high_resolution_clock::now();
    stats.collision_detection_time = std::chrono::duration<float, std::milli>(t3 - t2).count();
    
    // Detect composites
    if (physics_config.enabled_systems & PhysicsConfig::SPRINGS) {
        updateComposites(state.particles, state.springs, state.composites);
    }
    
    auto t4 = std::chrono::high_resolution_clock::now();
    stats.composite_detection_time = std::chrono::duration<float, std::milli>(t4 - t3).count();
    
    // Clear forces
    state.particles.clear_forces();
    
    // Compute all forces (kernel fusion)
    if (physics_config.enabled_systems & PhysicsConfig::GRAVITY) {
        if (physics_config.gravity_mode == PhysicsConfig::PARTICLE_MESH) {
            computeGravityPM(state.particles, state.gravity);
        } else {
            computeGravityDirect(state.particles);
        }
    }
    
    if (physics_config.enabled_systems & PhysicsConfig::CONTACTS) {
        computeContacts(state.particles, state.contacts, *state.contact_index);
    }
    
    if (physics_config.enabled_systems & PhysicsConfig::SPRINGS) {
        computeSprings(state.particles, state.springs);
    }
    
    if (physics_config.enabled_systems & PhysicsConfig::SPRING_FIELD) {
        computeSpringField(state.particles, state.springs, *state.spring_index);
    }
    
    if (physics_config.enabled_systems & PhysicsConfig::RADIATION) {
        computeRadiation(state.particles, state.radiation, *state.radiation_index);
    }
    
    if (physics_config.enabled_systems & PhysicsConfig::THERMAL) {
        computeThermal(state.particles, state.springs, state.thermal);
    }
    
    auto t5 = std::chrono::high_resolution_clock::now();
    stats.update_time = std::chrono::duration<float, std::milli>(t5 - t4).count();
    
    // Integrate positions and velocities
    switch (physics_config.default_integrator) {
        case PhysicsConfig::IntegratorType::VELOCITY_VERLET:
            integrateVelocityVerlet(state.particles, dt);
            break;
        case PhysicsConfig::IntegratorType::SEMI_IMPLICIT:
            integrateSemiImplicit(state.particles, dt);
            break;
        default:
            integrateSemiImplicit(state.particles, dt);
            break;
    }
    
    auto t6 = std::chrono::high_resolution_clock::now();
    stats.integrate_time = std::chrono::duration<float, std::milli>(t6 - t5).count();
    
    // Update statistics
    stats.active_particles = state.particles.count;
    stats.active_springs = state.springs.count;
    stats.active_contacts = state.contacts.count;
    stats.active_composites = state.composites.count;
}

void CpuBackendOpenMP::computeGravityDirect(ParticlePool& particles) {
    const float G = Constants::G;
    const float softening = 1.0f;
    
    #pragma omp parallel for schedule(dynamic, 64)
    for (size_t i = 0; i < particles.count; i++) {
        float fx = 0, fy = 0;
        
        for (size_t j = 0; j < particles.count; j++) {
            if (i == j) continue;
            
            float dx = particles.pos_x[j] - particles.pos_x[i];
            float dy = particles.pos_y[j] - particles.pos_y[i];
            
            // Handle toroidal wrapping
            if (config.use_toroidal) {
                if (dx > config.world_size * 0.5f) dx -= config.world_size;
                if (dx < -config.world_size * 0.5f) dx += config.world_size;
                if (dy > config.world_size * 0.5f) dy -= config.world_size;
                if (dy < -config.world_size * 0.5f) dy += config.world_size;
            }
            
            float r2 = dx * dx + dy * dy + softening * softening;
            float r = sqrtf(r2);
            float f = G * particles.mass[i] * particles.mass[j] / (r2 * r);
            
            fx += f * dx;
            fy += f * dy;
        }
        
        particles.force_x[i] += fx;
        particles.force_y[i] += fy;
    }
}

void CpuBackendOpenMP::computeGravityPM(ParticlePool& particles, GravityField& field) {
    const size_t grid_size = config.pm_grid_size;
    const float cell_size = config.world_size / grid_size;
    const float G = Constants::G;
    
    // Clear density field
    std::memset(field.density, 0, grid_size * grid_size * sizeof(float));
    
    // Deposit particles onto grid (CIC)
    #pragma omp parallel for
    for (size_t i = 0; i < particles.count; i++) {
        float x = particles.pos_x[i] / cell_size;
        float y = particles.pos_y[i] / cell_size;
        
        int ix = (int)x;
        int iy = (int)y;
        float fx = x - ix;
        float fy = y - iy;
        
        // Wrap indices
        ix = (ix % grid_size + grid_size) % grid_size;
        iy = (iy % grid_size + grid_size) % grid_size;
        int ix1 = (ix + 1) % grid_size;
        int iy1 = (iy + 1) % grid_size;
        
        // CIC weights
        float w00 = (1 - fx) * (1 - fy) * particles.mass[i];
        float w10 = fx * (1 - fy) * particles.mass[i];
        float w01 = (1 - fx) * fy * particles.mass[i];
        float w11 = fx * fy * particles.mass[i];
        
        // Atomic adds for thread safety
        #pragma omp atomic
        field.density[iy * grid_size + ix] += w00;
        #pragma omp atomic
        field.density[iy * grid_size + ix1] += w10;
        #pragma omp atomic
        field.density[iy1 * grid_size + ix] += w01;
        #pragma omp atomic
        field.density[iy1 * grid_size + ix1] += w11;
    }
    
    // Copy density to FFT workspace
    for (size_t i = 0; i < grid_size * grid_size; i++) {
        fft_workspace[i][0] = field.density[i];
        fft_workspace[i][1] = 0;
    }
    
    // Forward FFT
    fftwf_execute(fft_forward);
    
    // Apply Green's function in Fourier space
    const float k_scale = 2.0f * M_PI / config.world_size;
    const float softening = 1.0f;
    
    #pragma omp parallel for
    for (size_t ky = 0; ky < grid_size; ky++) {
        for (size_t kx = 0; kx < grid_size; kx++) {
            size_t idx = ky * grid_size + kx;
            
            // Wave numbers (accounting for Nyquist)
            float kx_val = (kx <= grid_size/2) ? kx : kx - grid_size;
            float ky_val = (ky <= grid_size/2) ? ky : ky - grid_size;
            kx_val *= k_scale;
            ky_val *= k_scale;
            
            float k2 = kx_val * kx_val + ky_val * ky_val;
            
            if (k2 > 0) {
                float green = -4.0f * M_PI * G / (k2 + softening * softening);
                fft_workspace[idx][0] *= green;
                fft_workspace[idx][1] *= green;
                
                // Store force components (ik multiplication for gradient)
                field.force_x[idx] = -kx_val * fft_workspace[idx][1];  // Imaginary part
                field.force_y[idx] = -ky_val * fft_workspace[idx][1];
            }
        }
    }
    
    // Inverse FFT for potential
    fftwf_execute(fft_inverse);
    
    // Normalize and store potential
    float norm = 1.0f / (grid_size * grid_size);
    for (size_t i = 0; i < grid_size * grid_size; i++) {
        field.potential[i] = fft_workspace[i][0] * norm;
    }
    
    // Interpolate forces back to particles (CIC)
    #pragma omp parallel for
    for (size_t i = 0; i < particles.count; i++) {
        float x = particles.pos_x[i] / cell_size;
        float y = particles.pos_y[i] / cell_size;
        
        int ix = (int)x;
        int iy = (int)y;
        float fx = x - ix;
        float fy = y - iy;
        
        ix = (ix % grid_size + grid_size) % grid_size;
        iy = (iy % grid_size + grid_size) % grid_size;
        int ix1 = (ix + 1) % grid_size;
        int iy1 = (iy + 1) % grid_size;
        
        // CIC interpolation of forces
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

void CpuBackendOpenMP::computeContacts(ParticlePool& particles, ContactPool& contacts, SpatialIndex& index) {
    const float k_contact = config.contact_stiffness;
    const float damping = config.contact_damping;
    
    #pragma omp parallel for
    for (size_t c = 0; c < contacts.count; c++) {
        uint32_t i = contacts.particle1[c];
        uint32_t j = contacts.particle2[c];
        
        // Hertzian contact model: F = k * δ^1.5
        float force_mag = k_contact * powf(contacts.overlap[c], 1.5f);
        
        // Velocity damping
        float vx_rel = particles.vel_x[j] - particles.vel_x[i];
        float vy_rel = particles.vel_y[j] - particles.vel_y[i];
        float v_normal = vx_rel * contacts.normal_x[c] + vy_rel * contacts.normal_y[c];
        
        force_mag += damping * v_normal;
        
        // Apply forces
        float fx = force_mag * contacts.normal_x[c];
        float fy = force_mag * contacts.normal_y[c];
        
        #pragma omp atomic
        particles.force_x[i] -= fx;
        #pragma omp atomic
        particles.force_y[i] -= fy;
        #pragma omp atomic
        particles.force_x[j] += fx;
        #pragma omp atomic
        particles.force_y[j] += fy;
    }
}

void CpuBackendOpenMP::computeSprings(ParticlePool& particles, SpringPool& springs) {
    #pragma omp parallel for
    for (size_t s = 0; s < springs.count; s++) {
        if (springs.is_broken[s]) continue;
        
        uint32_t i = springs.particle1[s];
        uint32_t j = springs.particle2[s];
        
        float dx = particles.pos_x[j] - particles.pos_x[i];
        float dy = particles.pos_y[j] - particles.pos_y[i];
        
        // Handle toroidal wrapping
        if (config.use_toroidal) {
            if (dx > config.world_size * 0.5f) dx -= config.world_size;
            if (dx < -config.world_size * 0.5f) dx += config.world_size;
            if (dy > config.world_size * 0.5f) dy -= config.world_size;
            if (dy < -config.world_size * 0.5f) dy += config.world_size;
        }
        
        float dist = sqrtf(dx * dx + dy * dy);
        float strain = (dist - springs.rest_length[s]) / springs.rest_length[s];
        
        // Check for breaking
        if (fabsf(strain) > springs.break_strain[s]) {
            springs.is_broken[s] = 1;
            // Energy release as heat
            float energy = 0.5f * springs.stiffness[s] * strain * strain;
            particles.temp_internal[i] += energy / (2 * particles.mass[i]);
            particles.temp_internal[j] += energy / (2 * particles.mass[j]);
            continue;
        }
        
        springs.current_strain[s] = strain;
        
        // Spring force
        float spring_force = springs.stiffness[s] * (dist - springs.rest_length[s]);
        
        // Damping
        float vx_rel = particles.vel_x[j] - particles.vel_x[i];
        float vy_rel = particles.vel_y[j] - particles.vel_y[i];
        float v_along = (vx_rel * dx + vy_rel * dy) / dist;
        float damping_force = springs.damping[s] * v_along;
        
        float total_force = (spring_force + damping_force) / dist;
        float fx = total_force * dx;
        float fy = total_force * dy;
        
        #pragma omp atomic
        particles.force_x[i] += fx;
        #pragma omp atomic
        particles.force_y[i] += fy;
        #pragma omp atomic
        particles.force_x[j] -= fx;
        #pragma omp atomic
        particles.force_y[j] -= fy;
    }
}

void CpuBackendOpenMP::integrateSemiImplicit(ParticlePool& particles, float dt) {
    #pragma omp parallel for
    for (size_t i = 0; i < particles.count; i++) {
        // Update velocity first
        particles.vel_x[i] += (particles.force_x[i] / particles.mass[i]) * dt;
        particles.vel_y[i] += (particles.force_y[i] / particles.mass[i]) * dt;
        
        // Clamp velocity
        float v_mag = sqrtf(particles.vel_x[i] * particles.vel_x[i] + 
                           particles.vel_y[i] * particles.vel_y[i]);
        if (v_mag > Constants::MAX_VELOCITY) {
            float scale = Constants::MAX_VELOCITY / v_mag;
            particles.vel_x[i] *= scale;
            particles.vel_y[i] *= scale;
        }
        
        // Update position with new velocity
        particles.pos_x[i] += particles.vel_x[i] * dt;
        particles.pos_y[i] += particles.vel_y[i] * dt;
        
        // Wrap position for toroidal space
        if (config.use_toroidal) {
            while (particles.pos_x[i] < 0) particles.pos_x[i] += config.world_size;
            while (particles.pos_x[i] >= config.world_size) particles.pos_x[i] -= config.world_size;
            while (particles.pos_y[i] < 0) particles.pos_y[i] += config.world_size;
            while (particles.pos_y[i] >= config.world_size) particles.pos_y[i] -= config.world_size;
        }
    }
}

// Stub implementations for now
void CpuBackendOpenMP::computeSpringField(ParticlePool& particles, SpringPool& springs, SpatialIndex& index) {
    // Virtual spring formation - particles form springs when close with low relative velocity
}

void CpuBackendOpenMP::computeRadiation(ParticlePool& particles, RadiationField& field, SpatialIndex& index) {
    // Radiation pressure and thermal radiation
}

void CpuBackendOpenMP::computeThermal(ParticlePool& particles, SpringPool& springs, ThermalField& field) {
    // Heat conduction through springs
}

void CpuBackendOpenMP::detectContacts(ParticlePool& particles, ContactPool& contacts, SpatialIndex& index) {
    // Use spatial index to find colliding particles
    const auto& grid = static_cast<const SparseGrid&>(index);
    
    for (const auto& [cell_hash, particle_ids] : grid.get_cells()) {
        // Check particles within same cell
        for (size_t i = 0; i < particle_ids.size(); i++) {
            for (size_t j = i + 1; j < particle_ids.size(); j++) {
                uint32_t pi = particle_ids[i];
                uint32_t pj = particle_ids[j];
                
                float dx = particles.pos_x[pj] - particles.pos_x[pi];
                float dy = particles.pos_y[pj] - particles.pos_y[pi];
                float dist2 = dx * dx + dy * dy;
                float min_dist = particles.radius[pi] + particles.radius[pj];
                
                if (dist2 < min_dist * min_dist) {
                    float dist = sqrtf(dist2);
                    float overlap = min_dist - dist;
                    
                    contacts.add_contact(pi, pj, overlap,
                                        dx/dist, dy/dist,
                                        particles.pos_x[pi] + dx * particles.radius[pi]/min_dist,
                                        particles.pos_y[pi] + dy * particles.radius[pi]/min_dist);
                }
            }
        }
    }
}

void CpuBackendOpenMP::updateComposites(ParticlePool& particles, SpringPool& springs, CompositePool& composites) {
    // Use Union-Find to detect connected components
    UnionFind uf;
    uf.reset(particles.count);
    
    for (size_t s = 0; s < springs.count; s++) {
        if (!springs.is_broken[s]) {
            uf.unite(springs.particle1[s], springs.particle2[s]);
        }
    }
    
    // Build composites from connected components
    composites.clear();
    std::unordered_map<uint32_t, std::vector<uint32_t>> groups;
    
    for (size_t i = 0; i < particles.count; i++) {
        uint32_t root = uf.find(i);
        groups[root].push_back(i);
    }
    
    for (const auto& [root, members] : groups) {
        if (members.size() > 1) {
            // Calculate composite properties
            float com_x = 0, com_y = 0, total_mass = 0;
            
            for (uint32_t id : members) {
                com_x += particles.pos_x[id] * particles.mass[id];
                com_y += particles.pos_y[id] * particles.mass[id];
                total_mass += particles.mass[id];
            }
            
            com_x /= total_mass;
            com_y /= total_mass;
            
            // Add to composite pool
            // (would need to implement add_composite method)
        }
    }
}

void CpuBackendOpenMP::integrateVelocityVerlet(ParticlePool& particles, float dt) {
    // Velocity Verlet requires storing old forces
    // For now, fall back to semi-implicit
    integrateSemiImplicit(particles, dt);
}


} // namespace digistar
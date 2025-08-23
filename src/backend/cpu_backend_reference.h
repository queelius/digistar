#pragma once

#include "backend_interface.h"
#include "../physics/spatial_index.h"
#include <chrono>
#include <fftw3.h>

namespace digistar {

// Reference CPU implementation - single-threaded, clear, and deterministic
// This serves as the ground truth for validating optimized backends
class CpuBackendReference : public IBackend {
private:
    SimulationConfig config;
    SimulationStats stats;
    
    // Timing
    std::chrono::high_resolution_clock::time_point last_step_start;
    
    // FFTW plans for PM gravity (if enabled)
    fftwf_plan fft_forward = nullptr;
    fftwf_plan fft_inverse = nullptr;
    fftwf_complex* fft_workspace = nullptr;
    
    // Temporary storage for Velocity Verlet
    float* old_force_x = nullptr;
    float* old_force_y = nullptr;
    
    // Internal methods for force computation - simple and clear
    void computeGravityDirect(ParticlePool& particles);
    void computeGravityPM(ParticlePool& particles, GravityField& field);
    void computeGravityBarnesHut(ParticlePool& particles);  // Future
    
    void computeContacts(ParticlePool& particles, ContactPool& contacts);
    void computeSprings(ParticlePool& particles, SpringPool& springs);
    void computeSpringField(ParticlePool& particles, SpringPool& springs, SpatialIndex& index);
    void computeRadiation(ParticlePool& particles, RadiationField& field, SpatialIndex& index);
    void computeThermal(ParticlePool& particles, SpringPool& springs);
    
    // Integration methods
    void integrateVelocityVerlet(ParticlePool& particles, float dt);
    void integrateSemiImplicit(ParticlePool& particles, float dt);
    void integrateLeapfrog(ParticlePool& particles, float dt);
    void integrateForwardEuler(ParticlePool& particles, float dt);  // For comparison
    
    // Helper methods
    void detectContacts(ParticlePool& particles, ContactPool& contacts, SpatialIndex& index);
    void updateComposites(ParticlePool& particles, SpringPool& springs, CompositePool& composites);
    void checkSpringBreaking(SpringPool& springs, ParticlePool& particles);
    void formNewSprings(ParticlePool& particles, SpringPool& springs, SpatialIndex& index);
    
public:
    CpuBackendReference() = default;
    ~CpuBackendReference() override;
    
    void initialize(const SimulationConfig& config) override;
    void shutdown() override;
    
    void step(SimulationState& state, const PhysicsConfig& config, float dt) override;
    
    SimulationStats getStats() const override { return stats; }
    std::string getName() const override { return "CPU Reference (Single-threaded)"; }
    
    uint32_t getSupportedSystems() const override {
        // Supports everything for testing
        return PhysicsConfig::GRAVITY | 
               PhysicsConfig::CONTACTS | 
               PhysicsConfig::SPRINGS |
               PhysicsConfig::SPRING_FIELD |
               PhysicsConfig::RADIATION |
               PhysicsConfig::THERMAL |
               PhysicsConfig::FUSION |
               PhysicsConfig::FISSION;
    }
    
    size_t getMaxParticles() const override { 
        return 100'000;  // Reasonable for single-threaded
    }
};

} // namespace digistar
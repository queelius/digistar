#pragma once

#include "backend_interface.h"
#include "../physics/spatial_index.h"
#include <chrono>
#include <fftw3.h>

namespace digistar {

class CpuBackendOpenMP : public IBackend {
private:
    SimulationConfig config;
    SimulationStats stats;
    
    // Timing
    std::chrono::high_resolution_clock::time_point last_step_start;
    
    // FFTW plans for PM gravity (if enabled)
    fftwf_plan fft_forward = nullptr;
    fftwf_plan fft_inverse = nullptr;
    fftwf_complex* fft_workspace = nullptr;
    
    // Internal methods for force computation
    void computeGravityDirect(ParticlePool& particles);
    void computeGravityPM(ParticlePool& particles, GravityField& field);
    void computeContacts(ParticlePool& particles, ContactPool& contacts, SpatialIndex& index);
    void computeSprings(ParticlePool& particles, SpringPool& springs);
    void computeSpringField(ParticlePool& particles, SpringPool& springs, SpatialIndex& index);
    void computeRadiation(ParticlePool& particles, RadiationField& field, SpatialIndex& index);
    void computeThermal(ParticlePool& particles, SpringPool& springs, ThermalField& field);
    
    // Integration methods
    void integrateVelocityVerlet(ParticlePool& particles, float dt);
    void integrateSemiImplicit(ParticlePool& particles, float dt);
    
    // Helper methods
    void detectContacts(ParticlePool& particles, ContactPool& contacts, SpatialIndex& index);
    void updateComposites(ParticlePool& particles, SpringPool& springs, CompositePool& composites);
    void handleFusionFission(ParticlePool& particles, const PhysicsConfig& config);
    
public:
    CpuBackendOpenMP() = default;
    ~CpuBackendOpenMP() override;
    
    void initialize(const SimulationConfig& config) override;
    void shutdown() override;
    
    void step(SimulationState& state, const PhysicsConfig& config, float dt) override;
    
    SimulationStats getStats() const override { return stats; }
    std::string getName() const override { return "CPU Backend (OpenMP)"; }
    
    uint32_t getSupportedSystems() const override {
        return PhysicsConfig::GRAVITY | 
               PhysicsConfig::CONTACTS | 
               PhysicsConfig::SPRINGS |
               PhysicsConfig::SPRING_FIELD |
               PhysicsConfig::RADIATION |
               PhysicsConfig::THERMAL;
    }
    
    size_t getMaxParticles() const override { 
        return 2'000'000;  // Realistic CPU limit
    }
};

} // namespace digistar
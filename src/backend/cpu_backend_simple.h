#pragma once

#include "backend_interface.h"
#include <chrono>

namespace digistar {

// Simple single-threaded CPU backend for examples and testing
// This is a minimal implementation of IBackend that supports basic physics
class CpuBackendSimple : public IBackend {
private:
    SimulationConfig config;
    SimulationStats stats;
    
    // Timing
    std::chrono::high_resolution_clock::time_point last_step_start;
    
    // Internal methods for basic physics
    void computeGravityDirect(ParticlePool& particles, float gravity_constant);
    void computeContacts(ParticlePool& particles, ContactPool& contacts, float stiffness, float damping);
    void integrateSemiImplicit(ParticlePool& particles, float dt);
    
public:
    CpuBackendSimple() = default;
    ~CpuBackendSimple() override = default;
    
    void initialize(const SimulationConfig& config) override;
    void shutdown() override;
    
    void step(SimulationState& state, const PhysicsConfig& config, float dt) override;
    
    SimulationStats getStats() const override { return stats; }
    std::string getName() const override { return "CPU Simple (Single-threaded)"; }
    
    uint32_t getSupportedSystems() const override {
        // Only supports basic gravity and contacts
        return PhysicsConfig::GRAVITY | PhysicsConfig::CONTACTS;
    }
    
    size_t getMaxParticles() const override { 
        return 10'000;  // Reasonable for simple single-threaded
    }
};

} // namespace digistar
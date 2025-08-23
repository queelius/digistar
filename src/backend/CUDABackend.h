#pragma once

#include "ISimulationBackend.h"

// Forward declaration
class CUDABackend : public ISimulationBackend {
public:
    CUDABackend();
    ~CUDABackend();
    
    void initialize(size_t num_particles, const SimulationParams& params) override;
    void setParticles(const std::vector<Particle>& particles) override;
    void getParticles(std::vector<Particle>& particles) override;
    void computeForces() override;
    void integrate(float dt) override;
    
    size_t getMaxParticles() const override;
    std::string getBackendName() const override;
    bool isGPU() const override;
    size_t getMemoryUsage() const override;
    void cleanup() override;
    
private:
    class Impl;  // Hide CUDA details from header
    std::unique_ptr<Impl> pImpl;
};
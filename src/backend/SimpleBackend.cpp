// Improved Simple CPU Backend with fixed Barnes-Hut
#include "ISimulationBackend.h"
#include "../algorithms/BarnesHutRobust.h"
#include "../algorithms/ParticleMeshCustom.h"
#include <omp.h>
#include <cmath>
#include <iostream>
#include <memory>

class SimpleBackend : public SimulationBackendBase {
private:
    std::unique_ptr<BarnesHutRobust> barnes_hut;
    std::unique_ptr<ParticleMeshCustom> particle_mesh;
    
public:
    SimpleBackend() : barnes_hut(nullptr), particle_mesh(nullptr) {}
    
    void initialize(size_t num_particles, const SimulationParams& p) override {
        SimulationBackendBase::initialize(num_particles, p);
        
        // Initialize algorithms as needed
        if (current_algorithm == ForceAlgorithm::BARNES_HUT) {
            barnes_hut = std::make_unique<BarnesHutRobust>(p.theta);
            barnes_hut->setSoftening(p.softening);
        } else if (current_algorithm == ForceAlgorithm::PARTICLE_MESH) {
            particle_mesh = std::make_unique<ParticleMeshCustom>(
                p.grid_size, p.box_size, p.gravity_constant, p.softening);
        }
    }
    
    void setAlgorithm(ForceAlgorithm algo) override {
        SimulationBackendBase::setAlgorithm(algo);
        
        // Initialize algorithms as needed
        if (algo == ForceAlgorithm::BARNES_HUT && !barnes_hut) {
            barnes_hut = std::make_unique<BarnesHutRobust>(params.theta);
            barnes_hut->setSoftening(params.softening);
        } else if (algo == ForceAlgorithm::PARTICLE_MESH && !particle_mesh) {
            particle_mesh = std::make_unique<ParticleMeshCustom>(
                params.grid_size, params.box_size, params.gravity_constant, params.softening);
        }
    }
    
    size_t getMaxParticles() const override { 
        switch (current_algorithm) {
            case ForceAlgorithm::BRUTE_FORCE:
                return 10000;    // O(n²) - good for <10k
            case ForceAlgorithm::BARNES_HUT:
                return 1000000;  // O(n log n) - good for 10k-1M
            case ForceAlgorithm::PARTICLE_MESH:
                return 10000000; // O(n) - good for >100k
            default:
                return 10000;
        }
    }
    
    std::string getBackendName() const override { 
        std::string algo_name;
        switch (current_algorithm) {
            case ForceAlgorithm::BRUTE_FORCE: algo_name = "Brute"; break;
            case ForceAlgorithm::BARNES_HUT: algo_name = "Barnes-Hut"; break;
            case ForceAlgorithm::PARTICLE_MESH: algo_name = "PM"; break;
            case ForceAlgorithm::HYBRID: algo_name = "Hybrid"; break;
        }
        return "CPU v4 (" + std::to_string(omp_get_max_threads()) + 
               " threads, " + algo_name + ")";
    }
    
    bool isGPU() const override { 
        return false; 
    }
    
    size_t getMemoryUsage() const override {
        size_t mem = SimulationBackendBase::getMemoryUsage();
        // Barnes-Hut tree roughly 2x particle count in nodes
        if (barnes_hut) {
            mem += particles.size() * sizeof(void*) * 4; // Rough estimate
        }
        if (particle_mesh) {
            mem += particle_mesh->getMemoryUsage();
        }
        return mem;
    }
    
protected:
    // Algorithm implementations
    void computeBruteForce() override {
        size_t n = particles.size();
        const float G = params.gravity_constant;
        const float soft2 = params.softening * params.softening;
        
        // Clear forces
        for (auto& f : forces) {
            f.x = 0;
            f.y = 0;
        }
        
        #pragma omp parallel for schedule(dynamic, 32)
        for (size_t i = 0; i < n; i++) {
            float fx = 0, fy = 0;
            
            for (size_t j = 0; j < n; j++) {
                if (i == j) continue;
                
                float dx = particles[j].pos.x - particles[i].pos.x;
                float dy = particles[j].pos.y - particles[i].pos.y;
                float r2 = dx*dx + dy*dy + soft2;
                float r = sqrt(r2);
                float a = G * particles[j].mass / (r2 * r);
                
                // Note: acceleration, not force!
                fx += a * dx;
                fy += a * dy;
            }
            
            forces[i].x = fx;
            forces[i].y = fy;
        }
    }
    
    void computeBarnesHut() override {
        if (!barnes_hut) {
            barnes_hut = std::make_unique<BarnesHutRobust>(params.theta);
            barnes_hut->setSoftening(params.softening);
        }
        
        // Build tree and calculate accelerations
        barnes_hut->buildTree(particles);
        barnes_hut->calculateAccelerations(forces, params.gravity_constant);
    }
    
    void computeParticleMesh() override {
        if (!particle_mesh) {
            particle_mesh = std::make_unique<ParticleMeshCustom>(
                params.grid_size, params.box_size, 
                params.gravity_constant, params.softening);
        }
        
        // Use PM algorithm for force calculation
        particle_mesh->calculateForces(particles, forces, params.gravity_constant);
    }
    
    void computeHybrid() override {
        // Hybrid: Use PM for long-range + direct for short-range
        // This is more complex and would need careful implementation
        // For now, default to Barnes-Hut
        computeBarnesHut();
    }
};
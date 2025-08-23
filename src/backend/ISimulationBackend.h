#pragma once

#include <memory>
#include <string>
#include <vector>
#include <functional>

// Basic 2D math types
struct float2 {
    float x, y;
    
    float2() : x(0), y(0) {}
    float2(float x_, float y_) : x(x_), y(y_) {}
    
    float2 operator+(const float2& other) const { return float2(x + other.x, y + other.y); }
    float2 operator-(const float2& other) const { return float2(x - other.x, y - other.y); }
    float2 operator*(float scalar) const { return float2(x * scalar, y * scalar); }
    float2 operator/(float scalar) const { return float2(x / scalar, y / scalar); }
};

// Particle structure
struct Particle {
    float2 pos;
    float2 vel;
    float mass;
    float radius;
};

// Simulation parameters
struct SimulationParams {
    float box_size = 1000.0f;
    float gravity_constant = 1.0f;
    float softening = 0.1f;
    float dt = 0.016f;  // 60 FPS
    int grid_size = 512;  // PM grid resolution
    float theta = 0.5f;   // Barnes-Hut opening angle
};

// Forward declarations for algorithm-specific data structures
// These are placeholders - actual implementations are in spatial/ and algorithms/
class PMGrid {
public:
    PMGrid(int grid_size, float box_size) {}
    ~PMGrid() = default;
    void assignParticles(const std::vector<Particle>& particles) {}
    void solvePotential() {}
    void interpolateForces(std::vector<float2>& forces) {}
    size_t getMemoryUsage() const { return sizeof(*this); }
};

// =============================================================================
// Force Algorithm Interface
// =============================================================================
enum class ForceAlgorithm {
    BRUTE_FORCE,
    BARNES_HUT,
    PARTICLE_MESH,
    HYBRID  // PM for long-range + brute for short-range
};

// =============================================================================
// Backend Type
// =============================================================================
enum class BackendType {
    AUTO,      // Auto-select best available
    CPU,       // Simple CPU with OpenMP
    SSE2,      // SSE2 SIMD (4-wide)
    AVX2,      // AVX2 SIMD (8-wide)
    CUDA       // GPU acceleration
};

// =============================================================================
// Simulation Backend Interface
// =============================================================================
class ISimulationBackend {
public:
    virtual ~ISimulationBackend() = default;
    
    // Initialize backend with particle count
    virtual void initialize(size_t num_particles, const SimulationParams& params) = 0;
    
    // Set the force calculation algorithm
    virtual void setAlgorithm(ForceAlgorithm algo) = 0;
    virtual ForceAlgorithm getAlgorithm() const = 0;
    
    // Upload particle data to backend
    virtual void setParticles(const std::vector<Particle>& particles) = 0;
    
    // Get particles back from backend
    virtual void getParticles(std::vector<Particle>& particles) = 0;
    
    // Compute forces using selected algorithm
    virtual void computeForces() = 0;
    
    // Integrate particle positions
    virtual void integrate(float dt) = 0;
    
    // Combined update step
    virtual void step(float dt) {
        computeForces();
        integrate(dt);
    }
    
    // Performance queries
    virtual size_t getMaxParticles() const = 0;
    virtual std::string getBackendName() const = 0;
    virtual bool isGPU() const = 0;
    virtual bool supportsAlgorithm(ForceAlgorithm algo) const = 0;
    
    // Memory management
    virtual size_t getMemoryUsage() const = 0;
    virtual void cleanup() = 0;
    
protected:
    // Algorithm-specific implementations (to be overridden by backends)
    virtual void computeBruteForce() = 0;
    virtual void computeBarnesHut() = 0;
    virtual void computeParticleMesh() = 0;
    virtual void computeHybrid() = 0;
};

// =============================================================================
// Base Implementation with Common Functionality
// =============================================================================
class SimulationBackendBase : public ISimulationBackend {
protected:
    std::vector<Particle> particles;
    std::vector<float2> forces;
    SimulationParams params;
    ForceAlgorithm current_algorithm = ForceAlgorithm::BRUTE_FORCE;
    
    // Algorithm-specific data structures
    void* quadtree_impl = nullptr;  // Actual type is in BarnesHut.h
    std::unique_ptr<PMGrid> pm_grid;
    
public:
    void initialize(size_t num_particles, const SimulationParams& p) override {
        params = p;
        particles.resize(num_particles);
        forces.resize(num_particles);
        
        // Initialize algorithm-specific structures if needed
        if (current_algorithm == ForceAlgorithm::BARNES_HUT) {
            // quadtree = std::make_unique<QuadTree>(params.box_size);
        } else if (current_algorithm == ForceAlgorithm::PARTICLE_MESH) {
            // pm_grid = std::make_unique<PMGrid>(params.grid_size, params.box_size);
        }
    }
    
    void setAlgorithm(ForceAlgorithm algo) override {
        if (!supportsAlgorithm(algo)) {
            // Fall back to brute force
            current_algorithm = ForceAlgorithm::BRUTE_FORCE;
        } else {
            current_algorithm = algo;
        }
    }
    
    ForceAlgorithm getAlgorithm() const override {
        return current_algorithm;
    }
    
    void setParticles(const std::vector<Particle>& p) override {
        particles = p;
    }
    
    void getParticles(std::vector<Particle>& p) override {
        p = particles;
    }
    
    void computeForces() override {
        // Clear forces
        for (auto& f : forces) {
            f.x = f.y = 0;
        }
        
        // Dispatch to algorithm-specific implementation
        switch (current_algorithm) {
            case ForceAlgorithm::BRUTE_FORCE:
                computeBruteForce();
                break;
            case ForceAlgorithm::BARNES_HUT:
                computeBarnesHut();
                break;
            case ForceAlgorithm::PARTICLE_MESH:
                computeParticleMesh();
                break;
            case ForceAlgorithm::HYBRID:
                computeHybrid();
                break;
        }
    }
    
    void integrate(float dt) override {
        size_t n = particles.size();
        
        for (size_t i = 0; i < n; i++) {
            // Update velocity (forces array contains accelerations)
            particles[i].vel.x += forces[i].x * dt;
            particles[i].vel.y += forces[i].y * dt;
            
            // Update position
            particles[i].pos.x += particles[i].vel.x * dt;
            particles[i].pos.y += particles[i].vel.y * dt;
            
            // Periodic boundary conditions
            if (particles[i].pos.x < 0) particles[i].pos.x += params.box_size;
            if (particles[i].pos.x >= params.box_size) particles[i].pos.x -= params.box_size;
            if (particles[i].pos.y < 0) particles[i].pos.y += params.box_size;
            if (particles[i].pos.y >= params.box_size) particles[i].pos.y -= params.box_size;
        }
    }
    
    bool supportsAlgorithm(ForceAlgorithm algo) const override {
        // By default, all algorithms are supported
        // Specific backends can override this
        return true;
    }
    
    size_t getMemoryUsage() const override {
        size_t mem = particles.size() * (sizeof(Particle) + sizeof(float2));
        
        // Add algorithm-specific memory
        if (quadtree_impl) {
            // Memory calculation handled by derived classes
            mem += 1024 * 1024;  // Estimate 1MB for tree
        }
        if (pm_grid) {
            mem += pm_grid->getMemoryUsage();
        }
        
        return mem;
    }
    
    void cleanup() override {
        particles.clear();
        forces.clear();
        // Note: quadtree_impl cleanup handled by derived classes
        quadtree_impl = nullptr;
        pm_grid.reset();
    }
    
protected:
    // Default implementations (can be overridden for optimization)
    void computeBarnesHut() override {
        // Default: fall back to brute force
        // Backends that support Barnes-Hut will override this
        computeBruteForce();
    }
    
    void computeParticleMesh() override {
        // Default: fall back to brute force
        // Backends that support PM will override this
        computeBruteForce();
    }
    
    void computeHybrid() override {
        // Default: just use brute force
        // Advanced backends will override this
        computeBruteForce();
    }
};

// =============================================================================
// Backend Factory
// =============================================================================
class BackendFactory {
public:
    static std::unique_ptr<ISimulationBackend> create(
        BackendType type = BackendType::AUTO,
        ForceAlgorithm algorithm = ForceAlgorithm::BRUTE_FORCE,
        size_t target_particles = 0
    );
    
    static bool hasCUDA();
    static bool hasAVX2();
    static int getNumCPUCores();
    
    // Get best backend for given requirements
    static BackendType recommendBackend(
        size_t num_particles,
        ForceAlgorithm algorithm
    );
};
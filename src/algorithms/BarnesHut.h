#pragma once

#include "../spatial/QuadTree.h"
#include "../backend/ISimulationBackend.h"
#include <omp.h>

// Barnes-Hut algorithm implementation that can be used by any backend
// This is a general implementation that CPU backends can use directly
// GPU backends will need their own parallel tree construction/traversal

// Adapter to make our Particle work with QuadTree
struct ParticleForTree {
    Vec2 pos;
    float mass;
    float radius;
    size_t index;  // Index in original particle array
    
    ParticleForTree() : pos(0, 0), mass(1), radius(0.5), index(0) {}
    
    ParticleForTree(const Particle& p, size_t idx) 
        : pos(p.pos.x, p.pos.y), mass(p.mass), radius(p.radius), index(idx) {}
};

class BarnesHutAlgorithm {
private:
    std::unique_ptr<QuadTree<ParticleForTree>> tree;
    BoundingBox bounds;
    float theta;  // Opening angle parameter (0.5 is typical)
    
public:
    BarnesHutAlgorithm(float box_size, float theta_param = 0.5f) 
        : bounds(Vec2(0, 0), Vec2(box_size, box_size)),
          theta(theta_param) {}
    
    // Build the quadtree from particles
    void buildTree(const std::vector<Particle>& particles) {
        tree = std::make_unique<QuadTree<ParticleForTree>>(bounds);
        
        // Insert all particles into tree with their indices
        for (size_t i = 0; i < particles.size(); i++) {
            tree->insert(ParticleForTree(particles[i], i));
        }
    }
    
    // Calculate forces using Barnes-Hut approximation
    // This version can be called by any CPU backend
    void calculateForces(
        const std::vector<Particle>& particles,
        std::vector<float2>& forces,
        float gravity_constant,
        float softening,
        int num_threads = 0  // 0 = use all available
    ) {
        if (!tree) {
            buildTree(particles);
        }
        
        size_t n = particles.size();
        forces.resize(n);
        
        // Set thread count if specified
        int old_threads = omp_get_max_threads();
        if (num_threads > 0) {
            omp_set_num_threads(num_threads);
        }
        
        // Calculate forces in parallel
        #pragma omp parallel for schedule(dynamic, 32)
        for (size_t i = 0; i < n; i++) {
            ParticleForTree p_tree(particles[i], i);
            Vec2 force = tree->calculateForce(
                p_tree, 
                theta, 
                gravity_constant, 
                softening
            );
            forces[i].x = force.x;
            forces[i].y = force.y;
        }
        
        // Restore thread count
        if (num_threads > 0) {
            omp_set_num_threads(old_threads);
        }
    }
    
    // Version for backends that process particles in chunks
    void calculateForcesChunk(
        const std::vector<Particle>& particles,
        std::vector<float2>& forces,
        size_t start_idx,
        size_t end_idx,
        float gravity_constant,
        float softening
    ) {
        if (!tree) {
            buildTree(particles);
        }
        
        for (size_t i = start_idx; i < end_idx && i < particles.size(); i++) {
            ParticleForTree p_tree(particles[i], i);
            Vec2 force = tree->calculateForce(
                p_tree, 
                theta, 
                gravity_constant, 
                softening
            );
            forces[i].x = force.x;
            forces[i].y = force.y;
        }
    }
    
    // SIMD-friendly version that processes multiple particles
    // Returns forces for 4 or 8 particles at once for SSE2/AVX2
    template<int WIDTH>
    void calculateForcesSIMD(
        const Particle* particles,
        float2* forces,
        size_t start_idx,
        float gravity_constant,
        float softening
    ) {
        if (!tree) return;
        
        // Process WIDTH particles
        for (int i = 0; i < WIDTH; i++) {
            ParticleForTree p_tree(particles[start_idx + i], start_idx + i);
            Vec2 force = tree->calculateForce(
                p_tree, 
                theta, 
                gravity_constant, 
                softening
            );
            forces[start_idx + i].x = force.x;
            forces[start_idx + i].y = force.y;
        }
    }
    
    // Update parameters
    void setTheta(float new_theta) { theta = new_theta; }
    float getTheta() const { return theta; }
    
    // Tree statistics
    size_t getNodeCount() const { 
        return tree ? tree->getNodeCount() : 0; 
    }
    
    size_t getTreeHeight() const { 
        return tree ? tree->getHeight() : 0; 
    }
    
    size_t getMemoryUsage() const {
        return tree ? 
            sizeof(*this) + tree->getNodeCount() * sizeof(typename QuadTree<Particle>::Node) : 
            sizeof(*this);
    }
    
    // Clear the tree (forces rebuild on next calculation)
    void clear() {
        tree.reset();
    }
};
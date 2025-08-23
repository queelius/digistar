#pragma once

#include "../backend/ISimulationBackend.h"
#include <vector>
#include <memory>
#include <cmath>
#include <algorithm>
#include <iostream>
#include <omp.h>

// Robust Barnes-Hut implementation
// Key principles:
// 1. Proper tree construction with accurate center of mass
// 2. Correct force calculation (acceleration, not force)
// 3. Handling of edge cases (particles at same position, etc.)
class BarnesHutRobust {
private:
    struct BHNode {
        // Spatial bounds
        float x_min, x_max, y_min, y_max;
        
        // Center of mass and total mass
        float cm_x, cm_y;
        float total_mass;
        
        // Children (quadrants: NW=0, NE=1, SW=2, SE=3)
        std::unique_ptr<BHNode> children[4];
        
        // For leaf nodes: the particle (if exactly one)
        int particle_index;  // -1 if internal node or empty
        
        BHNode(float xmin, float xmax, float ymin, float ymax)
            : x_min(xmin), x_max(xmax), y_min(ymin), y_max(ymax),
              cm_x(0), cm_y(0), total_mass(0), particle_index(-1) {
            for (int i = 0; i < 4; i++) children[i] = nullptr;
        }
        
        bool isLeaf() const {
            return children[0] == nullptr;
        }
        
        float width() const { return x_max - x_min; }
        float height() const { return y_max - y_min; }
        float size() const { return std::max(width(), height()); }
        
        int getQuadrant(float x, float y) const {
            float mid_x = (x_min + x_max) * 0.5f;
            float mid_y = (y_min + y_max) * 0.5f;
            
            if (x < mid_x) {
                return (y < mid_y) ? 2 : 0;  // SW : NW
            } else {
                return (y < mid_y) ? 3 : 1;  // SE : NE
            }
        }
        
        void subdivide() {
            float mid_x = (x_min + x_max) * 0.5f;
            float mid_y = (y_min + y_max) * 0.5f;
            
            children[0] = std::make_unique<BHNode>(x_min, mid_x, mid_y, y_max);  // NW
            children[1] = std::make_unique<BHNode>(mid_x, x_max, mid_y, y_max);  // NE
            children[2] = std::make_unique<BHNode>(x_min, mid_x, y_min, mid_y);  // SW
            children[3] = std::make_unique<BHNode>(mid_x, x_max, y_min, mid_y);  // SE
        }
    };
    
    std::unique_ptr<BHNode> root;
    const std::vector<Particle>* particles_ptr;
    float theta;  // Opening angle (0.5 typical, smaller = more accurate)
    float epsilon2;  // Softening squared
    
public:
    BarnesHutRobust(float theta_ = 0.5f) 
        : root(nullptr), particles_ptr(nullptr), theta(theta_), epsilon2(0.0001f) {}
    
    void setSoftening(float epsilon) {
        epsilon2 = epsilon * epsilon;
    }
    
    void buildTree(const std::vector<Particle>& particles) {
        particles_ptr = &particles;
        
        if (particles.empty()) return;
        
        // Find bounding box
        float x_min = particles[0].pos.x, x_max = particles[0].pos.x;
        float y_min = particles[0].pos.y, y_max = particles[0].pos.y;
        
        for (const auto& p : particles) {
            x_min = std::min(x_min, p.pos.x);
            x_max = std::max(x_max, p.pos.x);
            y_min = std::min(y_min, p.pos.y);
            y_max = std::max(y_max, p.pos.y);
        }
        
        // Add small margin to avoid particles exactly on boundaries
        float margin = (x_max - x_min) * 0.001f + 0.001f;
        x_min -= margin;
        x_max += margin;
        y_min -= margin;
        y_max += margin;
        
        // Create root node
        root = std::make_unique<BHNode>(x_min, x_max, y_min, y_max);
        
        // Insert all particles
        for (size_t i = 0; i < particles.size(); i++) {
            insertParticle(root.get(), i);
        }
        
        // Calculate centers of mass
        computeCenterOfMass(root.get());
    }
    
    void calculateAccelerations(std::vector<float2>& accelerations, float G) {
        if (!root || !particles_ptr) return;
        
        const auto& particles = *particles_ptr;
        accelerations.resize(particles.size());
        
        // Calculate acceleration for each particle
        #pragma omp parallel for schedule(dynamic, 64)
        for (size_t i = 0; i < particles.size(); i++) {
            accelerations[i].x = 0;
            accelerations[i].y = 0;
            
            // Calculate force from tree
            calculateAcceleration(root.get(), particles[i], i, 
                                accelerations[i], G);
        }
    }
    
    // Get tree statistics for debugging
    void printStats() const {
        if (!root) {
            std::cout << "Tree is empty\n";
            return;
        }
        
        int depth = getDepth(root.get());
        int nodes = countNodes(root.get());
        int leaves = countLeaves(root.get());
        
        std::cout << "Barnes-Hut Tree Stats:\n";
        std::cout << "  Depth: " << depth << "\n";
        std::cout << "  Total nodes: " << nodes << "\n";
        std::cout << "  Leaf nodes: " << leaves << "\n";
        std::cout << "  Root mass: " << root->total_mass << "\n";
        std::cout << "  Root CM: (" << root->cm_x << ", " << root->cm_y << ")\n";
    }
    
private:
    void insertParticle(BHNode* node, int index) {
        const auto& particles = *particles_ptr;
        const Particle& p = particles[index];
        
        // Check if particle is within bounds (should always be true)
        if (p.pos.x < node->x_min || p.pos.x > node->x_max ||
            p.pos.y < node->y_min || p.pos.y > node->y_max) {
            // Particle outside bounds - shouldn't happen with proper setup
            return;
        }
        
        // If node is empty leaf, just store particle
        if (node->isLeaf() && node->particle_index == -1) {
            node->particle_index = index;
            return;
        }
        
        // If node contains exactly one particle, need to subdivide
        if (node->isLeaf() && node->particle_index >= 0) {
            int old_index = node->particle_index;
            node->particle_index = -1;
            
            // Create children
            node->subdivide();
            
            // Reinsert old particle
            const Particle& old_p = particles[old_index];
            int old_quad = node->getQuadrant(old_p.pos.x, old_p.pos.y);
            insertParticle(node->children[old_quad].get(), old_index);
        }
        
        // Insert new particle into appropriate child
        int quad = node->getQuadrant(p.pos.x, p.pos.y);
        insertParticle(node->children[quad].get(), index);
    }
    
    void computeCenterOfMass(BHNode* node) {
        if (!node) return;
        
        const auto& particles = *particles_ptr;
        
        if (node->isLeaf()) {
            if (node->particle_index >= 0) {
                // Single particle
                const Particle& p = particles[node->particle_index];
                node->cm_x = p.pos.x;
                node->cm_y = p.pos.y;
                node->total_mass = p.mass;
            } else {
                // Empty node
                node->total_mass = 0;
            }
        } else {
            // Internal node - compute from children
            node->cm_x = 0;
            node->cm_y = 0;
            node->total_mass = 0;
            
            for (int i = 0; i < 4; i++) {
                if (node->children[i]) {
                    computeCenterOfMass(node->children[i].get());
                    
                    BHNode* child = node->children[i].get();
                    if (child->total_mass > 0) {
                        node->cm_x += child->cm_x * child->total_mass;
                        node->cm_y += child->cm_y * child->total_mass;
                        node->total_mass += child->total_mass;
                    }
                }
            }
            
            if (node->total_mass > 0) {
                node->cm_x /= node->total_mass;
                node->cm_y /= node->total_mass;
            }
        }
    }
    
    void calculateAcceleration(BHNode* node, const Particle& p, int p_index,
                              float2& accel, float G) {
        if (!node || node->total_mass == 0) return;
        
        // Skip self-interaction for leaf nodes
        if (node->isLeaf() && node->particle_index == p_index) {
            return;
        }
        
        // Calculate distance to center of mass
        float dx = node->cm_x - p.pos.x;
        float dy = node->cm_y - p.pos.y;
        float r2 = dx*dx + dy*dy;
        
        // Prevent singularity
        if (r2 < 1e-10f) return;
        
        // Barnes-Hut criterion: s/d < theta
        // where s is node size and d is distance
        float node_size = node->size();
        
        if (node->isLeaf() || (node_size * node_size / r2 < theta * theta)) {
            // Treat node as single body
            r2 += epsilon2;  // Add softening
            float r = sqrtf(r2);
            float a = G * node->total_mass / (r2 * r);  // acceleration magnitude
            
            // Add to acceleration (not force!)
            accel.x += a * dx;
            accel.y += a * dy;
        } else {
            // Node too close/large, recurse into children
            for (int i = 0; i < 4; i++) {
                if (node->children[i]) {
                    calculateAcceleration(node->children[i].get(), p, p_index, 
                                        accel, G);
                }
            }
        }
    }
    
    // Helper functions for statistics
    int getDepth(const BHNode* node) const {
        if (!node || node->isLeaf()) return 1;
        
        int max_depth = 0;
        for (int i = 0; i < 4; i++) {
            if (node->children[i]) {
                max_depth = std::max(max_depth, getDepth(node->children[i].get()));
            }
        }
        return max_depth + 1;
    }
    
    int countNodes(const BHNode* node) const {
        if (!node) return 0;
        
        int count = 1;
        if (!node->isLeaf()) {
            for (int i = 0; i < 4; i++) {
                count += countNodes(node->children[i].get());
            }
        }
        return count;
    }
    
    int countLeaves(const BHNode* node) const {
        if (!node) return 0;
        if (node->isLeaf()) return 1;
        
        int count = 0;
        for (int i = 0; i < 4; i++) {
            count += countLeaves(node->children[i].get());
        }
        return count;
    }
};
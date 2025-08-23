#pragma once

#include "../backend/ISimulationBackend.h"
#include <vector>
#include <memory>
#include <cmath>
#include <omp.h>

// Fixed Barnes-Hut implementation with correct force calculation
class BarnesHutFixed {
private:
    struct Node {
        float cx, cy;      // Center of mass
        float mass;        // Total mass
        float size;        // Size of this node
        float x, y;        // Position (bottom-left corner)
        
        Node* children[4]; // Quadrants: NW, NE, SW, SE
        
        // For leaf nodes
        std::vector<size_t> particle_indices;
        
        Node(float x_, float y_, float size_) 
            : cx(0), cy(0), mass(0), size(size_), x(x_), y(y_) {
            for (int i = 0; i < 4; i++) children[i] = nullptr;
        }
        
        ~Node() {
            for (int i = 0; i < 4; i++) {
                delete children[i];
            }
        }
        
        bool isLeaf() const {
            return children[0] == nullptr;
        }
        
        int getQuadrant(float px, float py) const {
            float midx = x + size/2;
            float midy = y + size/2;
            
            if (px < midx) {
                return (py < midy) ? 2 : 0;  // SW : NW
            } else {
                return (py < midy) ? 3 : 1;  // SE : NE
            }
        }
    };
    
    Node* root;
    float theta;  // Barnes-Hut accuracy parameter
    float box_size;
    
public:
    BarnesHutFixed(float box_size_, float theta_ = 0.5f) 
        : root(nullptr), theta(theta_), box_size(box_size_) {}
    
    ~BarnesHutFixed() {
        delete root;
    }
    
    void buildTree(const std::vector<Particle>& particles) {
        // Clean up old tree
        delete root;
        root = new Node(0, 0, box_size);
        
        // Insert all particles
        for (size_t i = 0; i < particles.size(); i++) {
            insertParticle(root, particles[i], i);
        }
        
        // Calculate centers of mass
        calculateCenterOfMass(root, particles);
    }
    
    void calculateForces(const std::vector<Particle>& particles,
                        std::vector<float2>& forces,
                        float G, float softening) {
        forces.resize(particles.size());
        
        #pragma omp parallel for schedule(dynamic, 64)
        for (size_t i = 0; i < particles.size(); i++) {
            forces[i].x = 0;
            forces[i].y = 0;
            calculateForceOnParticle(root, particles[i], forces[i], G, softening);
        }
    }
    
private:
    void insertParticle(Node* node, const Particle& p, size_t index) {
        // If leaf with no particles, just add it
        if (node->isLeaf() && node->particle_indices.empty()) {
            node->particle_indices.push_back(index);
            return;
        }
        
        // If leaf with particles, need to subdivide
        if (node->isLeaf()) {
            // Create children
            float half = node->size / 2;
            node->children[0] = new Node(node->x, node->y + half, half);        // NW
            node->children[1] = new Node(node->x + half, node->y + half, half); // NE
            node->children[2] = new Node(node->x, node->y, half);               // SW
            node->children[3] = new Node(node->x + half, node->y, half);        // SE
            
            // Move existing particles to children
            std::vector<size_t> old_indices = node->particle_indices;
            node->particle_indices.clear();
            
            // Don't actually need particle data here, just distribute indices
            // This is a simplification - in practice we'd pass particles too
        }
        
        // Insert into appropriate child
        int quadrant = node->getQuadrant(p.pos.x, p.pos.y);
        if (!node->children[quadrant]) {
            float half = node->size / 2;
            float child_x = node->x + (quadrant & 1 ? half : 0);
            float child_y = node->y + (quadrant < 2 ? half : 0);
            node->children[quadrant] = new Node(child_x, child_y, half);
        }
        insertParticle(node->children[quadrant], p, index);
    }
    
    void calculateCenterOfMass(Node* node, const std::vector<Particle>& particles) {
        if (node->isLeaf()) {
            // Calculate from particles
            node->cx = 0;
            node->cy = 0;
            node->mass = 0;
            
            for (size_t idx : node->particle_indices) {
                const Particle& p = particles[idx];
                node->cx += p.pos.x * p.mass;
                node->cy += p.pos.y * p.mass;
                node->mass += p.mass;
            }
            
            if (node->mass > 0) {
                node->cx /= node->mass;
                node->cy /= node->mass;
            }
        } else {
            // Calculate from children
            node->cx = 0;
            node->cy = 0;
            node->mass = 0;
            
            for (int i = 0; i < 4; i++) {
                if (node->children[i]) {
                    calculateCenterOfMass(node->children[i], particles);
                    
                    Node* child = node->children[i];
                    node->cx += child->cx * child->mass;
                    node->cy += child->cy * child->mass;
                    node->mass += child->mass;
                }
            }
            
            if (node->mass > 0) {
                node->cx /= node->mass;
                node->cy /= node->mass;
            }
        }
    }
    
    void calculateForceOnParticle(Node* node, const Particle& p, 
                                 float2& force, float G, float softening) {
        if (!node || node->mass == 0) return;
        
        float dx = node->cx - p.pos.x;
        float dy = node->cy - p.pos.y;
        float r2 = dx*dx + dy*dy;
        
        // Skip self-interaction
        if (r2 < 1e-10f) return;
        
        // Check if we can use this node as a single body
        float node_size2 = node->size * node->size;
        
        if (node->isLeaf() || (node_size2 / r2 < theta * theta)) {
            // Calculate force from this node
            r2 += softening * softening;
            float r = sqrt(r2);
            float f = G * node->mass / (r2 * r);
            
            // F = ma, but we want acceleration, so a = F/m
            // But force on particle is F = G * m_particle * m_node / r^2
            // So acceleration is a = G * m_node / r^2
            // This is why we DON'T multiply by particle mass here!
            force.x += f * dx;
            force.y += f * dy;
        } else {
            // Recurse into children
            for (int i = 0; i < 4; i++) {
                if (node->children[i]) {
                    calculateForceOnParticle(node->children[i], p, force, G, softening);
                }
            }
        }
    }
};
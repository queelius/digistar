#pragma once

#include <vector>
#include <string>
#include "../../backend/ISimulationBackend.h"

// Base interface for all AI agents
class IAgent {
public:
    struct Observation {
        std::vector<Particle> visible_particles;  // Particles in sensor range
        float2 position;
        float2 velocity;
        float energy;
        float sensor_range;
        float time;
    };
    
    struct Action {
        float2 thrust;      // Acceleration to apply
        float rotation;     // Rotation rate
        bool fire_weapon;   // Fire projectile
        bool activate_shield;
        float2 target_position;  // For navigation
    };
    
    virtual ~IAgent() = default;
    
    // Core agent interface
    virtual void initialize(size_t agent_id, const SimulationParams& params) = 0;
    virtual Action think(const Observation& obs) = 0;
    virtual void learn(float reward) = 0;
    virtual std::string getName() const = 0;
    virtual std::string getType() const = 0;
    
    // Optional debug interface
    virtual void setDebugMode(bool debug) {}
    virtual std::string getDebugInfo() const { return ""; }
};

// Simple reactive agent using subsumption architecture
class ReactiveAgent : public IAgent {
private:
    size_t id;
    float aggression;
    float exploration;
    
    // Subsumption layers (higher priority overrides lower)
    Action avoidCollision(const Observation& obs) {
        Action act = {0};
        
        // Find nearest particle
        float min_dist = 1000000;
        Particle nearest;
        for (const auto& p : obs.visible_particles) {
            float dx = p.pos.x - obs.position.x;
            float dy = p.pos.y - obs.position.y;
            float dist = sqrt(dx*dx + dy*dy);
            if (dist < min_dist && dist > 0) {
                min_dist = dist;
                nearest = p;
            }
        }
        
        // Avoid if too close
        if (min_dist < 5.0f) {
            float dx = obs.position.x - nearest.pos.x;
            float dy = obs.position.y - nearest.pos.y;
            float mag = sqrt(dx*dx + dy*dy);
            act.thrust.x = (dx/mag) * 10.0f;
            act.thrust.y = (dy/mag) * 10.0f;
        }
        
        return act;
    }
    
    Action seekResource(const Observation& obs) {
        Action act = {0};
        
        // Find resource particles (small mass)
        for (const auto& p : obs.visible_particles) {
            if (p.mass < 1.0f) {
                float dx = p.pos.x - obs.position.x;
                float dy = p.pos.y - obs.position.y;
                act.thrust.x = dx * 0.1f;
                act.thrust.y = dy * 0.1f;
                break;
            }
        }
        
        return act;
    }
    
    Action explore(const Observation& obs) {
        Action act = {0};
        
        // Random walk
        act.thrust.x = (rand() / (float)RAND_MAX - 0.5f) * exploration;
        act.thrust.y = (rand() / (float)RAND_MAX - 0.5f) * exploration;
        
        return act;
    }
    
public:
    ReactiveAgent() : id(0), aggression(0.5f), exploration(1.0f) {}
    
    void initialize(size_t agent_id, const SimulationParams& params) override {
        id = agent_id;
        aggression = rand() / (float)RAND_MAX;
        exploration = rand() / (float)RAND_MAX * 2.0f;
    }
    
    Action think(const Observation& obs) override {
        // Subsumption architecture - higher priority first
        Action act = avoidCollision(obs);
        
        // If no collision avoidance needed
        if (act.thrust.x == 0 && act.thrust.y == 0) {
            act = seekResource(obs);
        }
        
        // If no resources found
        if (act.thrust.x == 0 && act.thrust.y == 0) {
            act = explore(obs);
        }
        
        return act;
    }
    
    void learn(float reward) override {
        // Simple adaptation
        if (reward > 0) {
            exploration *= 0.95f;  // Exploit more
        } else {
            exploration *= 1.05f;  // Explore more
        }
        exploration = std::min(2.0f, std::max(0.1f, exploration));
    }
    
    std::string getName() const override { 
        return "ReactiveAgent_" + std::to_string(id); 
    }
    
    std::string getType() const override { 
        return "Subsumption"; 
    }
};
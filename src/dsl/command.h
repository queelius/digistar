#pragma once

#include <memory>
#include <queue>
#include <vector>
#include <functional>
#include <mutex>
#include <atomic>
#include "../core/vec2.h"

namespace digistar {

// Forward declarations
class Simulation;
class ParticlePool;

namespace dsl {

// Base command interface
class Command {
public:
    virtual ~Command() = default;
    
    // Execute the command on the simulation
    virtual void execute(Simulation& sim) = 0;
    
    // Get command description for debugging
    virtual std::string toString() const = 0;
    
    // Get provisional ID if this command creates an entity
    virtual int getProvisionalId() const { return -1; }
};

using CommandPtr = std::unique_ptr<Command>;

// ============ Particle Commands ============

class CreateParticleCommand : public Command {
private:
    double mass;
    Vec2 position;
    Vec2 velocity;
    double temperature;
    int provisional_id;  // ID assigned before creation
    
public:
    CreateParticleCommand(double m, Vec2 pos, Vec2 vel, double temp, int prov_id)
        : mass(m), position(pos), velocity(vel), temperature(temp), provisional_id(prov_id) {}
    
    void execute(Simulation& sim) override;
    std::string toString() const override;
    int getProvisionalId() const override { return provisional_id; }
};

class SetVelocityCommand : public Command {
private:
    int particle_id;
    Vec2 velocity;
    
public:
    SetVelocityCommand(int id, Vec2 vel) : particle_id(id), velocity(vel) {}
    
    void execute(Simulation& sim) override;
    std::string toString() const override;
};

class ApplyForceCommand : public Command {
private:
    int particle_id;
    Vec2 force;
    
public:
    ApplyForceCommand(int id, Vec2 f) : particle_id(id), force(f) {}
    
    void execute(Simulation& sim) override;
    std::string toString() const override;
};

class DestroyParticleCommand : public Command {
private:
    int particle_id;
    
public:
    explicit DestroyParticleCommand(int id) : particle_id(id) {}
    
    void execute(Simulation& sim) override;
    std::string toString() const override;
};

// ============ Spring Commands ============

class CreateSpringCommand : public Command {
private:
    int particle1_id;
    int particle2_id;
    double stiffness;
    double damping;
    double equilibrium_distance;
    int provisional_id;
    
public:
    CreateSpringCommand(int p1, int p2, double k, double d, double eq, int prov_id)
        : particle1_id(p1), particle2_id(p2), stiffness(k), 
          damping(d), equilibrium_distance(eq), provisional_id(prov_id) {}
    
    void execute(Simulation& sim) override;
    std::string toString() const override;
    int getProvisionalId() const override { return provisional_id; }
};

class BreakSpringCommand : public Command {
private:
    int spring_id;
    
public:
    explicit BreakSpringCommand(int id) : spring_id(id) {}
    
    void execute(Simulation& sim) override;
    std::string toString() const override;
};

// ============ Batch Commands ============

class CreateCloudCommand : public Command {
private:
    Vec2 center;
    double radius;
    int count;
    double mass_min, mass_max;
    double temp;
    std::vector<int> provisional_ids;
    
public:
    CreateCloudCommand(Vec2 c, double r, int n, double m_min, double m_max, double t)
        : center(c), radius(r), count(n), mass_min(m_min), mass_max(m_max), temp(t) {
        // Pre-allocate provisional IDs
        provisional_ids.reserve(n);
        for (int i = 0; i < n; i++) {
            provisional_ids.push_back(allocateProvisionalId());
        }
    }
    
    void execute(Simulation& sim) override;
    std::string toString() const override;
    const std::vector<int>& getProvisionalIds() const { return provisional_ids; }
    
private:
    static int allocateProvisionalId();
};

// ============ Query Commands ============
// Queries don't mutate state but may need special handling

class QueryRegionCommand : public Command {
private:
    Vec2 center;
    double radius;
    std::function<void(const std::vector<int>&)> callback;
    
public:
    QueryRegionCommand(Vec2 c, double r, std::function<void(const std::vector<int>&)> cb)
        : center(c), radius(r), callback(cb) {}
    
    void execute(Simulation& sim) override;
    std::string toString() const override;
};

// ============ Command Queue ============

class CommandQueue {
private:
    std::queue<CommandPtr> pending_commands;
    std::mutex queue_mutex;
    std::atomic<size_t> command_count{0};
    
    // For provisional ID allocation
    static std::atomic<int> next_provisional_id;
    
public:
    // Thread-safe command submission
    void push(CommandPtr cmd);
    
    // Push multiple commands atomically
    void pushBatch(std::vector<CommandPtr> cmds);
    
    // Apply all pending commands (called between frames)
    void executeAll(Simulation& sim);
    
    // Get number of pending commands
    size_t size() const { return command_count.load(); }
    
    // Clear all pending commands
    void clear();
    
    // Allocate a provisional ID for entity creation
    static int allocateProvisionalId() {
        return next_provisional_id.fetch_add(1);
    }
};

// ============ Command Factory ============
// Helper to create commands with provisional IDs

class CommandFactory {
private:
    CommandQueue& queue;
    
public:
    explicit CommandFactory(CommandQueue& q) : queue(q) {}
    
    // Create particle and return provisional ID
    int createParticle(double mass, Vec2 pos, Vec2 vel, double temp = 300);
    
    // Create spring between particles
    int createSpring(int p1, int p2, double stiffness, double damping = 0.1);
    
    // Batch operations
    std::vector<int> createCloud(Vec2 center, double radius, int count, 
                                 double mass_min = 1.0, double mass_max = 1.0);
    
    // Modifications
    void setVelocity(int particle_id, Vec2 vel);
    void applyForce(int particle_id, Vec2 force);
    void destroyParticle(int particle_id);
    void breakSpring(int spring_id);
    
    // Queries (with callbacks)
    void queryRegion(Vec2 center, double radius, 
                    std::function<void(const std::vector<int>&)> callback);
};

} // namespace dsl
} // namespace digistar
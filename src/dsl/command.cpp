#include "command.h"
#include "../core/simulation.h"
#include <sstream>
#include <random>

namespace digistar {
namespace dsl {

// Static initialization
std::atomic<int> CommandQueue::next_provisional_id{1000000}; // Start at 1M to avoid conflicts

// ============ Particle Commands ============

void CreateParticleCommand::execute(Simulation& sim) {
    // Create particle and map provisional ID to actual ID
    int actual_id = sim.createParticle(mass, position, velocity, temperature);
    sim.mapProvisionalId(provisional_id, actual_id);
}

std::string CreateParticleCommand::toString() const {
    std::stringstream ss;
    ss << "CreateParticle{mass=" << mass 
       << ", pos=" << position 
       << ", vel=" << velocity 
       << ", temp=" << temperature
       << ", prov_id=" << provisional_id << "}";
    return ss.str();
}

void SetVelocityCommand::execute(Simulation& sim) {
    // Resolve provisional ID if needed
    int actual_id = sim.resolveId(particle_id);
    auto* particle = sim.getParticle(actual_id);
    if (particle) {
        particle->velocity = velocity;
    }
}

std::string SetVelocityCommand::toString() const {
    std::stringstream ss;
    ss << "SetVelocity{id=" << particle_id << ", vel=" << velocity << "}";
    return ss.str();
}

void ApplyForceCommand::execute(Simulation& sim) {
    int actual_id = sim.resolveId(particle_id);
    auto* particle = sim.getParticle(actual_id);
    if (particle) {
        // F = ma, so a = F/m
        particle->velocity += force / particle->mass * sim.getDt();
    }
}

std::string ApplyForceCommand::toString() const {
    std::stringstream ss;
    ss << "ApplyForce{id=" << particle_id << ", force=" << force << "}";
    return ss.str();
}

void DestroyParticleCommand::execute(Simulation& sim) {
    int actual_id = sim.resolveId(particle_id);
    sim.destroyParticle(actual_id);
}

std::string DestroyParticleCommand::toString() const {
    return "DestroyParticle{id=" + std::to_string(particle_id) + "}";
}

// ============ Spring Commands ============

void CreateSpringCommand::execute(Simulation& sim) {
    int p1_actual = sim.resolveId(particle1_id);
    int p2_actual = sim.resolveId(particle2_id);
    
    int actual_id = sim.createSpring(p1_actual, p2_actual, 
                                     stiffness, damping, equilibrium_distance);
    sim.mapProvisionalId(provisional_id, actual_id);
}

std::string CreateSpringCommand::toString() const {
    std::stringstream ss;
    ss << "CreateSpring{p1=" << particle1_id 
       << ", p2=" << particle2_id
       << ", k=" << stiffness
       << ", prov_id=" << provisional_id << "}";
    return ss.str();
}

void BreakSpringCommand::execute(Simulation& sim) {
    int actual_id = sim.resolveId(spring_id);
    sim.breakSpring(actual_id);
}

std::string BreakSpringCommand::toString() const {
    return "BreakSpring{id=" + std::to_string(spring_id) + "}";
}

// ============ Batch Commands ============

int CreateCloudCommand::allocateProvisionalId() {
    return CommandQueue::allocateProvisionalId();
}

void CreateCloudCommand::execute(Simulation& sim) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> mass_dist(mass_min, mass_max);
    std::uniform_real_distribution<> pos_dist(-1.0, 1.0);
    
    for (int i = 0; i < count; i++) {
        // Random position in circle (2D)
        Vec2 offset;
        do {
            offset = Vec2(pos_dist(gen), pos_dist(gen));
        } while (offset.lengthSquared() > 1.0);
        
        Vec2 pos = center + offset * radius;
        Vec2 vel(0, 0); // Start at rest
        double mass = mass_dist(gen);
        
        int actual_id = sim.createParticle(mass, pos, vel, temp);
        sim.mapProvisionalId(provisional_ids[i], actual_id);
    }
}

std::string CreateCloudCommand::toString() const {
    std::stringstream ss;
    ss << "CreateCloud{center=" << center 
       << ", radius=" << radius
       << ", count=" << count << "}";
    return ss.str();
}

// ============ Query Commands ============

void QueryRegionCommand::execute(Simulation& sim) {
    std::vector<int> particles_in_region;
    
    // Find all particles within radius of center
    sim.forEachParticle([&](int id, const Particle& p) {
        if ((p.position - center).length() <= radius) {
            particles_in_region.push_back(id);
        }
    });
    
    // Invoke callback with results
    if (callback) {
        callback(particles_in_region);
    }
}

std::string QueryRegionCommand::toString() const {
    std::stringstream ss;
    ss << "QueryRegion{center=" << center << ", radius=" << radius << "}";
    return ss.str();
}

// ============ Command Queue ============

void CommandQueue::push(CommandPtr cmd) {
    std::lock_guard<std::mutex> lock(queue_mutex);
    pending_commands.push(std::move(cmd));
    command_count++;
}

void CommandQueue::pushBatch(std::vector<CommandPtr> cmds) {
    std::lock_guard<std::mutex> lock(queue_mutex);
    for (auto& cmd : cmds) {
        pending_commands.push(std::move(cmd));
    }
    command_count += cmds.size();
}

void CommandQueue::executeAll(Simulation& sim) {
    std::queue<CommandPtr> commands_to_execute;
    
    // Swap queues to minimize lock time
    {
        std::lock_guard<std::mutex> lock(queue_mutex);
        std::swap(commands_to_execute, pending_commands);
        command_count = 0;
    }
    
    // Execute commands without holding lock
    while (!commands_to_execute.empty()) {
        auto& cmd = commands_to_execute.front();
        cmd->execute(sim);
        commands_to_execute.pop();
    }
}

void CommandQueue::clear() {
    std::lock_guard<std::mutex> lock(queue_mutex);
    std::queue<CommandPtr> empty;
    std::swap(pending_commands, empty);
    command_count = 0;
}

// ============ Command Factory ============

int CommandFactory::createParticle(double mass, Vec2 pos, Vec2 vel, double temp) {
    int prov_id = CommandQueue::allocateProvisionalId();
    queue.push(std::make_unique<CreateParticleCommand>(mass, pos, vel, temp, prov_id));
    return prov_id;
}

int CommandFactory::createSpring(int p1, int p2, double stiffness, double damping) {
    int prov_id = CommandQueue::allocateProvisionalId();
    // Calculate equilibrium distance from current positions (will be done in execute)
    queue.push(std::make_unique<CreateSpringCommand>(p1, p2, stiffness, damping, -1, prov_id));
    return prov_id;
}

std::vector<int> CommandFactory::createCloud(Vec2 center, double radius, int count, 
                                            double mass_min, double mass_max) {
    auto cmd = std::make_unique<CreateCloudCommand>(center, radius, count, mass_min, mass_max, 300);
    auto ids = cmd->getProvisionalIds();
    queue.push(std::move(cmd));
    return ids;
}

void CommandFactory::setVelocity(int particle_id, Vec2 vel) {
    queue.push(std::make_unique<SetVelocityCommand>(particle_id, vel));
}

void CommandFactory::applyForce(int particle_id, Vec2 force) {
    queue.push(std::make_unique<ApplyForceCommand>(particle_id, force));
}

void CommandFactory::destroyParticle(int particle_id) {
    queue.push(std::make_unique<DestroyParticleCommand>(particle_id));
}

void CommandFactory::breakSpring(int spring_id) {
    queue.push(std::make_unique<BreakSpringCommand>(spring_id));
}

void CommandFactory::queryRegion(Vec2 center, double radius, 
                                std::function<void(const std::vector<int>&)> callback) {
    queue.push(std::make_unique<QueryRegionCommand>(center, radius, callback));
}

} // namespace dsl
} // namespace digistar
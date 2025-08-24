#include <gtest/gtest.h>
#include "../src/dsl/command.h"
#include "../src/core/simulation.h"
#include <thread>
#include <chrono>

using namespace digistar;
using namespace digistar::dsl;

class CommandQueueTest : public ::testing::Test {
protected:
    Simulation sim;
    CommandQueue queue;
    CommandFactory factory{queue};
};

TEST_F(CommandQueueTest, CreateParticleCommand) {
    // Create particle through command
    int prov_id = factory.createParticle(1.0, Vec2(0, 0), Vec2(1, 0), 300);
    
    EXPECT_EQ(queue.size(), 1);
    EXPECT_EQ(sim.getParticleCount(), 0);  // Not created yet
    
    // Execute commands
    queue.executeAll(sim);
    
    EXPECT_EQ(queue.size(), 0);
    EXPECT_EQ(sim.getParticleCount(), 1);
    
    // Check particle was created
    int actual_id = sim.resolveId(prov_id);
    auto* particle = sim.getParticle(actual_id);
    ASSERT_NE(particle, nullptr);
    EXPECT_DOUBLE_EQ(particle->mass, 1.0);
    EXPECT_EQ(particle->position, Vec2(0, 0));
    EXPECT_EQ(particle->velocity, Vec2(1, 0));
}

TEST_F(CommandQueueTest, BatchCommands) {
    // Create multiple particles
    std::vector<int> ids;
    for (int i = 0; i < 10; i++) {
        ids.push_back(factory.createParticle(
            1.0, Vec2(i, 0), Vec2(0, 0), 300));
    }
    
    EXPECT_EQ(queue.size(), 10);
    EXPECT_EQ(sim.getParticleCount(), 0);
    
    // Execute all at once
    queue.executeAll(sim);
    
    EXPECT_EQ(queue.size(), 0);
    EXPECT_EQ(sim.getParticleCount(), 10);
}

TEST_F(CommandQueueTest, CreateCloud) {
    // Create cloud of particles
    auto prov_ids = factory.createCloud(Vec2(0, 0), 10.0, 100, 0.5, 2.0);
    
    EXPECT_EQ(prov_ids.size(), 100);
    EXPECT_EQ(queue.size(), 1);  // Single cloud command
    
    queue.executeAll(sim);
    
    EXPECT_EQ(sim.getParticleCount(), 100);
    
    // Check all particles are within radius
    sim.forEachParticle([](int id, const Particle& p) {
        EXPECT_LE(p.position.length(), 10.0);
        EXPECT_GE(p.mass, 0.5);
        EXPECT_LE(p.mass, 2.0);
    });
}

TEST_F(CommandQueueTest, ModifyParticle) {
    // Create particle
    int prov_id = factory.createParticle(1.0, Vec2(0, 0), Vec2(0, 0), 300);
    queue.executeAll(sim);
    
    // Modify velocity
    factory.setVelocity(prov_id, Vec2(5, 0));
    queue.executeAll(sim);
    
    int actual_id = sim.resolveId(prov_id);
    auto* particle = sim.getParticle(actual_id);
    ASSERT_NE(particle, nullptr);
    EXPECT_EQ(particle->velocity, Vec2(5, 0));
    
    // Apply force
    factory.applyForce(prov_id, Vec2(10, 0));  // F = ma, a = F/m = 10/1 = 10
    queue.executeAll(sim);
    
    // Velocity should increase by a*dt = 10*0.01 = 0.1
    particle = sim.getParticle(actual_id);
    EXPECT_NEAR(particle->velocity.x, 5.1, 1e-6);
}

TEST_F(CommandQueueTest, CreateAndConnectSprings) {
    // Create two particles
    int p1 = factory.createParticle(1.0, Vec2(0, 0), Vec2(0, 0), 300);
    int p2 = factory.createParticle(1.0, Vec2(10, 0), Vec2(0, 0), 300);
    queue.executeAll(sim);
    
    // Connect with spring
    int spring_id = factory.createSpring(p1, p2, 100.0, 0.1);
    queue.executeAll(sim);
    
    EXPECT_EQ(sim.getSpringCount(), 1);
    
    int actual_spring_id = sim.resolveId(spring_id);
    auto* spring = sim.getSpring(actual_spring_id);
    ASSERT_NE(spring, nullptr);
    EXPECT_DOUBLE_EQ(spring->stiffness, 100.0);
    EXPECT_DOUBLE_EQ(spring->damping, 0.1);
    EXPECT_DOUBLE_EQ(spring->equilibrium_distance, 10.0);  // Auto-calculated
}

TEST_F(CommandQueueTest, ThreadSafety) {
    const int num_threads = 4;
    const int particles_per_thread = 100;
    
    std::vector<std::thread> threads;
    std::vector<std::vector<int>> thread_ids(num_threads);
    
    // Each thread creates particles
    for (int t = 0; t < num_threads; t++) {
        threads.emplace_back([this, t, &thread_ids]() {
            for (int i = 0; i < particles_per_thread; i++) {
                int id = factory.createParticle(
                    1.0, 
                    Vec2(t * 100 + i, 0), 
                    Vec2(0, 0), 
                    300);
                thread_ids[t].push_back(id);
                
                // Small delay to increase chance of contention
                std::this_thread::sleep_for(std::chrono::microseconds(10));
            }
        });
    }
    
    // Wait for all threads
    for (auto& thread : threads) {
        thread.join();
    }
    
    // Should have all commands queued
    EXPECT_EQ(queue.size(), num_threads * particles_per_thread);
    
    // Execute all commands
    queue.executeAll(sim);
    
    // Should have all particles
    EXPECT_EQ(sim.getParticleCount(), num_threads * particles_per_thread);
    
    // Verify all provisional IDs map correctly
    for (const auto& ids : thread_ids) {
        for (int prov_id : ids) {
            int actual_id = sim.resolveId(prov_id);
            EXPECT_NE(sim.getParticle(actual_id), nullptr);
        }
    }
}

TEST_F(CommandQueueTest, QueryCommand) {
    // Create particles in different regions
    for (int i = 0; i < 10; i++) {
        factory.createParticle(1.0, Vec2(i * 5, 0), Vec2(0, 0), 300);
    }
    queue.executeAll(sim);
    
    // Query region
    std::vector<int> found_particles;
    factory.queryRegion(Vec2(10, 0), 8.0, 
        [&found_particles](const std::vector<int>& ids) {
            found_particles = ids;
        });
    queue.executeAll(sim);
    
    // Should find particles at x=5, 10, 15 (within radius 8 of x=10)
    EXPECT_EQ(found_particles.size(), 3);
}

TEST_F(CommandQueueTest, ClearQueue) {
    // Add commands
    for (int i = 0; i < 10; i++) {
        factory.createParticle(1.0, Vec2(i, 0), Vec2(0, 0), 300);
    }
    
    EXPECT_EQ(queue.size(), 10);
    
    // Clear before execution
    queue.clear();
    
    EXPECT_EQ(queue.size(), 0);
    
    // Execute should do nothing
    queue.executeAll(sim);
    EXPECT_EQ(sim.getParticleCount(), 0);
}

TEST_F(CommandQueueTest, DestroyParticleBreaksSprings) {
    // Create triangle of particles with springs
    int p1 = factory.createParticle(1.0, Vec2(0, 0), Vec2(0, 0), 300);
    int p2 = factory.createParticle(1.0, Vec2(10, 0), Vec2(0, 0), 300);
    int p3 = factory.createParticle(1.0, Vec2(5, 5), Vec2(0, 0), 300);
    queue.executeAll(sim);
    
    // Connect with springs
    factory.createSpring(p1, p2, 100.0, 0.1);
    factory.createSpring(p2, p3, 100.0, 0.1);
    factory.createSpring(p3, p1, 100.0, 0.1);
    queue.executeAll(sim);
    
    EXPECT_EQ(sim.getParticleCount(), 3);
    EXPECT_EQ(sim.getSpringCount(), 3);
    
    // Destroy middle particle
    factory.destroyParticle(p2);
    queue.executeAll(sim);
    
    EXPECT_EQ(sim.getParticleCount(), 2);
    EXPECT_EQ(sim.getSpringCount(), 1);  // Only spring between p1 and p3 remains
}
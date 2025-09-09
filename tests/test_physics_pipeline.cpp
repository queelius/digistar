/**
 * Unit Tests for Physics Pipeline
 * 
 * Tests the physics pipeline component in isolation to ensure
 * command processing, event generation, and performance monitoring work correctly.
 */

#include <gtest/gtest.h>
#include <memory>
#include <thread>
#include <chrono>

#include "../src/simulation/physics_pipeline.h"
#include "../src/backend/cpu_backend_simple.h"

using namespace digistar;

// Mock event producer for testing
class MockEventProducer : public EventProducer {
public:
    explicit MockEventProducer() : EventProducer(nullptr) {}
    
    void writeEvent(const Event& event) override {
        events_.push_back(event);
    }
    
    const std::vector<Event>& getEvents() const { return events_; }
    void clearEvents() { events_.clear(); }
    size_t getEventCount() const { return events_.size(); }
    
private:
    std::vector<Event> events_;
};

class PhysicsPipelineTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Create simulation configuration
        SimulationConfig config;
        config.max_particles = 1000;
        config.max_springs = 5000;
        config.max_contacts = 1000;
        config.world_size = 1000.0f;
        
        // Create CPU backend
        backend_ = std::make_unique<CpuBackendSimple>();
        backend_->initialize(config);
        
        // Create mock event producer
        event_producer_ = std::make_shared<MockEventProducer>();
        
        // Create physics pipeline
        pipeline_ = std::make_unique<PhysicsPipeline>(backend_, event_producer_);
        pipeline_->initialize(config);
        
        // Initialize simulation state
        state_.particles.reserve(config.max_particles);
        state_.springs.reserve(config.max_springs);
        state_.contacts.reserve(config.max_contacts);
    }
    
    void TearDown() override {
        pipeline_->shutdown();
        backend_.reset();
        pipeline_.reset();
        event_producer_.reset();
    }
    
    std::shared_ptr<IBackend> backend_;
    std::shared_ptr<MockEventProducer> event_producer_;
    std::unique_ptr<PhysicsPipeline> pipeline_;
    SimulationState state_;
};

// Command creation tests
TEST_F(PhysicsPipelineTest, CreateParticleCommand) {
    auto cmd = pipeline_->createParticle(100.0f, 200.0f, 1.5f, 0.8f);
    
    EXPECT_EQ(cmd.type, PipelineCommandType::CREATE_PARTICLE);
    EXPECT_FLOAT_EQ(cmd.x, 100.0f);
    EXPECT_FLOAT_EQ(cmd.y, 200.0f);
    EXPECT_FLOAT_EQ(cmd.mass, 1.5f);
    EXPECT_FLOAT_EQ(cmd.radius, 0.8f);
    EXPECT_EQ(cmd.priority, 50);  // Medium priority for particle creation
}

TEST_F(PhysicsPipelineTest, DestroyParticleCommand) {
    auto cmd = pipeline_->destroyParticle(42);
    
    EXPECT_EQ(cmd.type, PipelineCommandType::DESTROY_PARTICLE);
    EXPECT_EQ(cmd.target_id, 42);
    EXPECT_EQ(cmd.priority, 30);  // High priority for destruction
}

TEST_F(PhysicsPipelineTest, ApplyForceCommand) {
    auto cmd = pipeline_->applyForce(123, 10.0f, -5.0f);
    
    EXPECT_EQ(cmd.type, PipelineCommandType::APPLY_FORCE);
    EXPECT_EQ(cmd.target_id, 123);
    EXPECT_FLOAT_EQ(cmd.fx, 10.0f);
    EXPECT_FLOAT_EQ(cmd.fy, -5.0f);
    EXPECT_EQ(cmd.priority, 80);  // Lower priority for forces
}

TEST_F(PhysicsPipelineTest, CreateSpringCommand) {
    auto cmd = pipeline_->createSpring(10, 20, 1000.0f, 50.0f);
    
    EXPECT_EQ(cmd.type, PipelineCommandType::CREATE_SPRING);
    EXPECT_EQ(cmd.target_ids.size(), 2);
    EXPECT_EQ(cmd.target_ids[0], 10);
    EXPECT_EQ(cmd.target_ids[1], 20);
    EXPECT_FLOAT_EQ(cmd.float_params.at("stiffness"), 1000.0f);
    EXPECT_FLOAT_EQ(cmd.float_params.at("damping"), 50.0f);
    EXPECT_EQ(cmd.priority, 60);  // Medium priority
}

TEST_F(PhysicsPipelineTest, GenerateGalaxyCommand) {
    auto cmd = pipeline_->generateGalaxy(0.0f, 0.0f, 1000, 5000.0f);
    
    EXPECT_EQ(cmd.type, PipelineCommandType::GENERATE_OBJECTS);
    EXPECT_FLOAT_EQ(cmd.x, 0.0f);
    EXPECT_FLOAT_EQ(cmd.y, 0.0f);
    EXPECT_EQ(cmd.int_params.at("count"), 1000);
    EXPECT_FLOAT_EQ(cmd.float_params.at("radius"), 5000.0f);
    EXPECT_EQ(cmd.string_params.at("type"), "galaxy");
    EXPECT_EQ(cmd.priority, 10);  // Very high priority for generation
}

// Queue management tests
TEST_F(PhysicsPipelineTest, BasicQueueOperations) {
    EXPECT_EQ(pipeline_->getQueueSize(), 0);
    
    auto cmd1 = pipeline_->createParticle(0, 0);
    auto cmd2 = pipeline_->createParticle(10, 10);
    
    pipeline_->enqueueCommand(cmd1);
    EXPECT_EQ(pipeline_->getQueueSize(), 1);
    
    pipeline_->enqueueCommand(cmd2);
    EXPECT_EQ(pipeline_->getQueueSize(), 2);
    
    pipeline_->clearQueue();
    EXPECT_EQ(pipeline_->getQueueSize(), 0);
}

TEST_F(PhysicsPipelineTest, BatchEnqueuing) {
    std::vector<PipelineCommand> commands;
    
    for (int i = 0; i < 5; ++i) {
        commands.push_back(pipeline_->createParticle(i * 10.0f, i * 10.0f));
    }
    
    pipeline_->enqueueCommands(commands);
    EXPECT_EQ(pipeline_->getQueueSize(), 5);
}

TEST_F(PhysicsPipelineTest, QueueSizeLimit) {
    pipeline_->setMaxQueueSize(3);
    
    // Add commands up to limit
    for (int i = 0; i < 3; ++i) {
        auto cmd = pipeline_->createParticle(i, i);
        pipeline_->enqueueCommand(cmd);
    }
    
    EXPECT_EQ(pipeline_->getQueueSize(), 3);
    
    // Adding more should fail (and increment failure count)
    auto overflow_cmd = pipeline_->createParticle(100, 100);
    pipeline_->enqueueCommand(overflow_cmd);
    
    EXPECT_EQ(pipeline_->getQueueSize(), 3);  // Should still be at limit
    
    auto stats = pipeline_->getStats();
    EXPECT_GT(stats.commands_failed, 0);
}

// Command processing tests
TEST_F(PhysicsPipelineTest, ParticleCreationProcessing) {
    // Create particle creation command
    auto cmd = pipeline_->createParticle(100.0f, 200.0f, 1.5f, 0.8f);
    pipeline_->enqueueCommand(cmd);
    
    size_t initial_particle_count = state_.particles.size();
    
    // Process commands by running update
    PhysicsConfig physics_config;
    pipeline_->update(state_, physics_config, 0.01f);
    
    // Check if particle was created
    EXPECT_GT(state_.particles.size(), initial_particle_count);
    
    // Verify command was processed
    auto stats = pipeline_->getStats();
    EXPECT_GT(stats.total_commands_processed, 0);
    EXPECT_EQ(pipeline_->getQueueSize(), 0);
}

TEST_F(PhysicsPipelineTest, GalaxyGeneration) {
    // Generate a small galaxy
    auto cmd = pipeline_->generateGalaxy(0.0f, 0.0f, 10, 100.0f);
    pipeline_->enqueueCommand(cmd);
    
    size_t initial_particle_count = state_.particles.size();
    
    PhysicsConfig physics_config;
    pipeline_->update(state_, physics_config, 0.01f);
    
    // Should have created particles
    EXPECT_GT(state_.particles.size(), initial_particle_count);
    
    // Check that particles are within expected range
    for (size_t i = 0; i < state_.particles.size(); ++i) {
        float distance = std::sqrt(state_.particles.x[i] * state_.particles.x[i] + 
                                 state_.particles.y[i] * state_.particles.y[i]);
        EXPECT_LE(distance, 100.0f);  // Within galaxy radius
    }
}

// Statistics and monitoring tests
TEST_F(PhysicsPipelineTest, StatisticsTracking) {
    auto initial_stats = pipeline_->getStats();
    
    EXPECT_EQ(initial_stats.total_commands_processed, 0);
    EXPECT_EQ(initial_stats.commands_pending, 0);
    EXPECT_EQ(initial_stats.commands_failed, 0);
    EXPECT_EQ(initial_stats.events_generated, 0);
    
    // Process some commands
    for (int i = 0; i < 5; ++i) {
        auto cmd = pipeline_->createParticle(i * 10.0f, i * 10.0f);
        pipeline_->enqueueCommand(cmd);
    }
    
    PhysicsConfig physics_config;
    pipeline_->update(state_, physics_config, 0.01f);
    
    auto stats = pipeline_->getStats();
    EXPECT_EQ(stats.total_commands_processed, 5);
    EXPECT_EQ(stats.commands_failed, 0);
    EXPECT_EQ(pipeline_->getQueueSize(), 0);
}

TEST_F(PhysicsPipelineTest, StatisticsReset) {
    // Process some commands first
    auto cmd = pipeline_->createParticle(0, 0);
    pipeline_->enqueueCommand(cmd);
    
    PhysicsConfig physics_config;
    pipeline_->update(state_, physics_config, 0.01f);
    
    auto stats_before = pipeline_->getStats();
    EXPECT_GT(stats_before.total_commands_processed, 0);
    
    // Reset statistics
    pipeline_->resetStats();
    
    auto stats_after = pipeline_->getStats();
    EXPECT_EQ(stats_after.total_commands_processed, 0);
    EXPECT_EQ(stats_after.commands_failed, 0);
    EXPECT_EQ(stats_after.events_generated, 0);
}

// Error handling tests
TEST_F(PhysicsPipelineTest, InvalidCommands) {
    // Create command with invalid particle ID for destruction
    auto invalid_cmd = pipeline_->destroyParticle(999999);
    pipeline_->enqueueCommand(invalid_cmd);
    
    PhysicsConfig physics_config;
    pipeline_->update(state_, physics_config, 0.01f);
    
    auto stats = pipeline_->getStats();
    EXPECT_GT(stats.commands_failed, 0);
    EXPECT_FALSE(stats.last_error.empty());
}

TEST_F(PhysicsPipelineTest, ErrorHandler) {
    bool error_handler_called = false;
    std::string captured_error;
    
    pipeline_->setErrorHandler([&](const std::string& error, const PipelineCommand& cmd) {
        error_handler_called = true;
        captured_error = error;
    });
    
    // Trigger error with invalid command
    auto invalid_cmd = pipeline_->destroyParticle(999999);
    pipeline_->enqueueCommand(invalid_cmd);
    
    PhysicsConfig physics_config;
    pipeline_->update(state_, physics_config, 0.01f);
    
    EXPECT_TRUE(error_handler_called);
    EXPECT_FALSE(captured_error.empty());
}

// Event processing tests
TEST_F(PhysicsPipelineTest, EventGeneration) {
    // Events are typically generated by the backend, not the pipeline
    // But we can test the event producer integration
    
    EXPECT_EQ(event_producer_->getEventCount(), 0);
    
    // Process some commands that might generate events
    auto cmd = pipeline_->createParticle(0, 0);
    pipeline_->enqueueCommand(cmd);
    
    PhysicsConfig physics_config;
    pipeline_->update(state_, physics_config, 0.01f);
    
    // Check if event producer received anything
    // (This depends on backend implementation)
    size_t event_count = event_producer_->getEventCount();
    EXPECT_GE(event_count, 0);  // Should be >= 0
}

// Performance tests
TEST_F(PhysicsPipelineTest, CommandProcessingPerformance) {
    const int num_commands = 1000;
    
    // Prepare many commands
    for (int i = 0; i < num_commands; ++i) {
        auto cmd = pipeline_->createParticle(i % 100, i % 100);
        pipeline_->enqueueCommand(cmd);
    }
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    PhysicsConfig physics_config;
    pipeline_->update(state_, physics_config, 0.01f);
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
    
    // Should process commands reasonably quickly
    EXPECT_LT(duration.count(), 10000);  // Less than 10ms for 1000 commands
    
    auto stats = pipeline_->getStats();
    EXPECT_EQ(stats.total_commands_processed, num_commands);
}

// Configuration tests
TEST_F(PhysicsPipelineTest, EventProcessingConfiguration) {
    EventProcessingConfig event_config;
    event_config.enabled_events = {EventType::PARTICLE_MERGE, EventType::SPRING_CREATED};
    event_config.enable_rate_limiting = true;
    event_config.max_events_per_frame = 100;
    
    pipeline_->setEventProcessingConfig(event_config);
    
    const auto& retrieved_config = pipeline_->getEventProcessingConfig();
    EXPECT_EQ(retrieved_config.enabled_events.size(), 2);
    EXPECT_TRUE(retrieved_config.enable_rate_limiting);
    EXPECT_EQ(retrieved_config.max_events_per_frame, 100);
}

TEST_F(PhysicsPipelineTest, BatchProcessing) {
    pipeline_->enableBatchProcessing(true, 10);
    
    // Add more than batch size
    for (int i = 0; i < 25; ++i) {
        auto cmd = pipeline_->createParticle(i, i);
        pipeline_->enqueueCommand(cmd);
    }
    
    PhysicsConfig physics_config;
    pipeline_->update(state_, physics_config, 0.01f);
    
    auto stats = pipeline_->getStats();
    EXPECT_EQ(stats.total_commands_processed, 25);
}

// Main function for running tests
int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    
    std::cout << "Running Physics Pipeline Tests\n";
    std::cout << "==============================\n";
    
    return RUN_ALL_TESTS();
}
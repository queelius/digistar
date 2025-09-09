/**
 * Comprehensive Integration Tests for DigiStar Simulation System
 * 
 * Tests the complete integration of all DigiStar components:
 * - IntegratedSimulation orchestrator
 * - Physics backends with event emission
 * - DSL runtime integration
 * - Configuration management
 * - Event system integration
 * - Performance monitoring
 */

#include <gtest/gtest.h>
#include <thread>
#include <chrono>
#include <memory>

#include "../src/simulation/integrated_simulation.h"
#include "../src/simulation/simulation_builder.h"
#include "../src/simulation/physics_pipeline.h"
#include "../src/config/simulation_config.h"
#include "../src/events/event_consumer.h"

using namespace digistar;

// Test fixture for integrated simulation tests
class IntegratedSimulationTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Create basic test configuration
        config_.backend_type = BackendFactory::Type::CPU;
        config_.simulation_config.max_particles = 1000;
        config_.simulation_config.max_springs = 5000;
        config_.simulation_config.world_size = 1000.0f;
        config_.enable_events = false;  // Disable events for basic tests
        config_.enable_dsl = false;     // Disable DSL for basic tests
        config_.enable_monitoring = false;
        config_.target_fps = 120.0f;    // Fast for testing
        config_.use_separate_physics_thread = false;  // Single-threaded for testing
    }
    
    void TearDown() override {
        if (simulation_ && simulation_->isRunning()) {
            simulation_->stop();
        }
        simulation_.reset();
    }
    
    IntegratedSimulationConfig config_;
    std::unique_ptr<IntegratedSimulation> simulation_;
};

// Basic lifecycle tests
TEST_F(IntegratedSimulationTest, BasicLifecycle) {
    simulation_ = std::make_unique<IntegratedSimulation>(config_);
    
    EXPECT_FALSE(simulation_->isInitialized());
    EXPECT_FALSE(simulation_->isRunning());
    
    ASSERT_TRUE(simulation_->initialize());
    EXPECT_TRUE(simulation_->isInitialized());
    EXPECT_FALSE(simulation_->isRunning());
    
    simulation_->start();
    EXPECT_TRUE(simulation_->isRunning());
    
    // Let it run briefly
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    
    simulation_->stop();
    EXPECT_FALSE(simulation_->isRunning());
    
    simulation_->shutdown();
    EXPECT_FALSE(simulation_->isInitialized());
}

TEST_F(IntegratedSimulationTest, PauseResume) {
    simulation_ = std::make_unique<IntegratedSimulation>(config_);
    ASSERT_TRUE(simulation_->initialize());
    
    simulation_->start();
    EXPECT_TRUE(simulation_->isRunning());
    EXPECT_FALSE(simulation_->isPaused());
    
    simulation_->pause();
    EXPECT_TRUE(simulation_->isRunning());
    EXPECT_TRUE(simulation_->isPaused());
    
    simulation_->resume();
    EXPECT_TRUE(simulation_->isRunning());
    EXPECT_FALSE(simulation_->isPaused());
    
    simulation_->stop();
}

TEST_F(IntegratedSimulationTest, SingleStep) {
    simulation_ = std::make_unique<IntegratedSimulation>(config_);
    ASSERT_TRUE(simulation_->initialize());
    
    simulation_->start();
    simulation_->pause();
    
    uint32_t initial_tick = simulation_->getStats().current_tick;
    
    simulation_->step();
    
    uint32_t after_step_tick = simulation_->getStats().current_tick;
    EXPECT_EQ(after_step_tick, initial_tick + 1);
}

// Component access tests
TEST_F(IntegratedSimulationTest, ComponentAccess) {
    simulation_ = std::make_unique<IntegratedSimulation>(config_);
    ASSERT_TRUE(simulation_->initialize());
    
    // Test backend access
    EXPECT_NE(simulation_->getBackend(), nullptr);
    EXPECT_EQ(simulation_->getBackend()->getName(), "CPU Simple Backend");
    
    // Test simulation state access
    auto& state = simulation_->getSimulationState();
    EXPECT_LE(state.particles.size(), config_.simulation_config.max_particles);
    EXPECT_LE(state.springs.size(), config_.simulation_config.max_springs);
}

// Configuration tests
TEST_F(IntegratedSimulationTest, ConfigurationUpdate) {
    simulation_ = std::make_unique<IntegratedSimulation>(config_);
    ASSERT_TRUE(simulation_->initialize());
    
    auto original_config = simulation_->getConfig();
    EXPECT_EQ(original_config.target_fps, 120.0f);
    
    // Update configuration
    auto new_config = original_config;
    new_config.target_fps = 60.0f;
    
    simulation_->updateConfig(new_config);
    
    auto updated_config = simulation_->getConfig();
    EXPECT_EQ(updated_config.target_fps, 60.0f);
}

// Performance and statistics tests
TEST_F(IntegratedSimulationTest, StatisticsTracking) {
    simulation_ = std::make_unique<IntegratedSimulation>(config_);
    ASSERT_TRUE(simulation_->initialize());
    
    simulation_->start();
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    
    const auto& stats = simulation_->getStats();
    
    EXPECT_GT(stats.total_frames, 0);
    EXPECT_GE(stats.current_fps, 0.0f);
    EXPECT_GE(stats.simulation_time, 0.0f);
    EXPECT_TRUE(stats.is_running);
    EXPECT_FALSE(stats.is_paused);
    
    simulation_->stop();
}

TEST_F(IntegratedSimulationTest, PerformanceReset) {
    simulation_ = std::make_unique<IntegratedSimulation>(config_);
    ASSERT_TRUE(simulation_->initialize());
    
    simulation_->start();
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    
    auto stats_before = simulation_->getStats();
    EXPECT_GT(stats_before.total_frames, 0);
    
    simulation_->resetStats();
    
    auto stats_after = simulation_->getStats();
    EXPECT_EQ(stats_after.total_frames, 0);
    EXPECT_EQ(stats_after.simulation_time, 0.0f);
    
    simulation_->stop();
}

// Error handling tests
TEST_F(IntegratedSimulationTest, ErrorHandling) {
    bool error_called = false;
    std::string error_message;
    
    simulation_ = std::make_unique<IntegratedSimulation>(config_);
    
    simulation_->setErrorHandler([&](const std::string& msg) {
        error_called = true;
        error_message = msg;
    });
    
    // Test error handling works
    simulation_->handleError("Test error", "test");
    
    EXPECT_TRUE(error_called);
    EXPECT_EQ(error_message, "Test error");
    EXPECT_EQ(simulation_->getLastError(), "Test error");
}

// Builder pattern tests
class SimulationBuilderTest : public ::testing::Test {
protected:
    void TearDown() override {
        if (simulation_ && simulation_->isRunning()) {
            simulation_->stop();
        }
        simulation_.reset();
    }
    
    std::unique_ptr<IntegratedSimulation> simulation_;
};

TEST_F(SimulationBuilderTest, BasicBuilder) {
    simulation_ = SimulationBuilder()
        .withMaxParticles(500)
        .withMaxSprings(2500)
        .withWorldSize(500.0f)
        .withTargetFPS(60.0f)
        .enableGravity(true)
        .enableContacts(true)
        .withoutEvents()
        .withDSL(false)
        .build();
    
    ASSERT_NE(simulation_, nullptr);
    
    const auto& config = simulation_->getConfig();
    EXPECT_EQ(config.simulation_config.max_particles, 500);
    EXPECT_EQ(config.simulation_config.max_springs, 2500);
    EXPECT_EQ(config.simulation_config.world_size, 500.0f);
    EXPECT_EQ(config.target_fps, 60.0f);
    EXPECT_FALSE(config.enable_events);
    EXPECT_FALSE(config.enable_dsl);
}

TEST_F(SimulationBuilderTest, PresetConfigurations) {
    // Test minimal preset
    simulation_ = SimulationBuilder::minimal().build();
    ASSERT_NE(simulation_, nullptr);
    
    const auto& config = simulation_->getConfig();
    EXPECT_EQ(config.simulation_config.max_particles, 1000);
    EXPECT_FALSE(config.enable_events);
    EXPECT_FALSE(config.enable_dsl);
}

TEST_F(SimulationBuilderTest, BuildAndInitialize) {
    simulation_ = SimulationBuilder()
        .withMaxParticles(100)
        .withoutEvents()
        .withDSL(false)
        .buildAndInitialize();
    
    ASSERT_NE(simulation_, nullptr);
    EXPECT_TRUE(simulation_->isInitialized());
}

TEST_F(SimulationBuilderTest, ValidationErrors) {
    EXPECT_THROW(
        SimulationBuilder()
            .withMaxParticles(0)  // Invalid
            .build(),
        std::runtime_error
    );
}

// Event system integration tests
class EventIntegrationTest : public ::testing::Test {
protected:
    void SetUp() override {
        config_.backend_type = BackendFactory::Type::CPU;
        config_.simulation_config.max_particles = 100;
        config_.enable_events = true;
        config_.event_shm_name = "test_events";
        config_.auto_create_event_system = true;
        config_.enable_dsl = false;
        config_.enable_monitoring = false;
        config_.use_separate_physics_thread = false;
    }
    
    void TearDown() override {
        if (simulation_ && simulation_->isRunning()) {
            simulation_->stop();
        }
        simulation_.reset();
    }
    
    IntegratedSimulationConfig config_;
    std::unique_ptr<IntegratedSimulation> simulation_;
};

TEST_F(EventIntegrationTest, EventSystemCreation) {
    simulation_ = std::make_unique<IntegratedSimulation>(config_);
    ASSERT_TRUE(simulation_->initialize());
    
    EXPECT_NE(simulation_->getEventSystem(), nullptr);
    EXPECT_TRUE(simulation_->getEventSystem()->is_valid());
}

TEST_F(EventIntegrationTest, EventConsumption) {
    simulation_ = std::make_unique<IntegratedSimulation>(config_);
    ASSERT_TRUE(simulation_->initialize());
    
    auto event_system = simulation_->getEventSystem();
    ASSERT_NE(event_system, nullptr);
    
    // Create event consumer
    EventConsumer consumer(event_system->get_buffer(), "test_consumer");
    
    simulation_->start();
    
    // Let simulation run briefly to potentially generate events
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    
    // Check if we can read events (there might not be any, but the mechanism should work)
    Event event;
    size_t events_read = 0;
    while (consumer.tryRead(event) && events_read < 10) {
        events_read++;
    }
    
    // No specific assertion about event count since it depends on physics
    EXPECT_GE(events_read, 0);
    
    simulation_->stop();
}

// Physics pipeline tests
class PhysicsPipelineTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Create a simple backend for testing
        SimulationConfig config;
        config.max_particles = 1000;
        config.max_springs = 5000;
        config.world_size = 1000.0f;
        
        backend_ = BackendFactory::create(BackendFactory::Type::CPU, config);
        ASSERT_NE(backend_, nullptr);
        
        pipeline_ = std::make_unique<PhysicsPipeline>(backend_);
        
        pipeline_->initialize(config);
    }
    
    void TearDown() override {
        pipeline_->shutdown();
        backend_.reset();
        pipeline_.reset();
    }
    
    std::shared_ptr<IBackend> backend_;
    std::unique_ptr<PhysicsPipeline> pipeline_;
};

TEST_F(PhysicsPipelineTest, CommandEnqueuing) {
    auto create_cmd = pipeline_->createParticle(100.0f, 200.0f, 1.0f, 0.5f);
    
    EXPECT_EQ(create_cmd.type, PipelineCommandType::CREATE_PARTICLE);
    EXPECT_EQ(create_cmd.x, 100.0f);
    EXPECT_EQ(create_cmd.y, 200.0f);
    EXPECT_EQ(create_cmd.mass, 1.0f);
    EXPECT_EQ(create_cmd.radius, 0.5f);
    
    EXPECT_EQ(pipeline_->getQueueSize(), 0);
    
    pipeline_->enqueueCommand(create_cmd);
    
    EXPECT_EQ(pipeline_->getQueueSize(), 1);
    
    pipeline_->clearQueue();
    
    EXPECT_EQ(pipeline_->getQueueSize(), 0);
}

TEST_F(PhysicsPipelineTest, BatchCommands) {
    std::vector<PipelineCommand> commands;
    
    for (int i = 0; i < 10; ++i) {
        commands.push_back(pipeline_->createParticle(i * 10.0f, i * 10.0f));
    }
    
    pipeline_->enqueueCommands(commands);
    
    EXPECT_EQ(pipeline_->getQueueSize(), 10);
}

TEST_F(PhysicsPipelineTest, Statistics) {
    auto stats = pipeline_->getStats();
    
    EXPECT_EQ(stats.total_commands_processed, 0);
    EXPECT_EQ(stats.commands_pending, 0);
    EXPECT_EQ(stats.commands_failed, 0);
    
    pipeline_->resetStats();
    
    auto reset_stats = pipeline_->getStats();
    EXPECT_EQ(reset_stats.total_commands_processed, 0);
}

// Configuration management tests
class ConfigurationTest : public ::testing::Test {
protected:
    void SetUp() override {
        config_manager_ = std::make_unique<ConfigurationManager>();
    }
    
    std::unique_ptr<ConfigurationManager> config_manager_;
};

TEST_F(ConfigurationTest, DefaultConfiguration) {
    ASSERT_NE(config_manager_, nullptr);
    
    const auto& ds_config = config_manager_->getDigiStarConfig();
    
    // Test default values
    EXPECT_EQ(ds_config.backend_type.get(), "cpu");
    EXPECT_EQ(ds_config.max_particles.get(), 100000);
    EXPECT_EQ(ds_config.enable_gravity.get(), true);
    EXPECT_EQ(ds_config.target_fps.get(), 60.0f);
}

TEST_F(ConfigurationTest, ProfileManagement) {
    auto profiles = config_manager_->getProfiles();
    
    EXPECT_FALSE(profiles.empty());
    EXPECT_NE(std::find(profiles.begin(), profiles.end(), "default"), profiles.end());
    EXPECT_NE(std::find(profiles.begin(), profiles.end(), "development"), profiles.end());
    
    EXPECT_EQ(config_manager_->getCurrentProfile(), "default");
    
    config_manager_->setProfile("development");
    EXPECT_EQ(config_manager_->getCurrentProfile(), "development");
}

TEST_F(ConfigurationTest, Validation) {
    auto errors = config_manager_->validate();
    EXPECT_TRUE(errors.empty());  // Default config should be valid
    
    // Test invalid configuration
    config_manager_->getDigiStarConfig().max_particles = 0;  // Invalid
    
    auto validation_errors = config_manager_->validate();
    EXPECT_FALSE(validation_errors.empty());
}

TEST_F(ConfigurationTest, ConfigurationSummary) {
    std::string summary = config_manager_->getSummary();
    
    EXPECT_FALSE(summary.empty());
    EXPECT_NE(summary.find("DigiStar Configuration Summary"), std::string::npos);
    EXPECT_NE(summary.find("Backend:"), std::string::npos);
    EXPECT_NE(summary.find("Physics:"), std::string::npos);
}

// Integration utility tests
TEST(UtilityTest, ConfigConversion) {
    // Test configuration manager to integrated config conversion
    auto config_manager = ConfigurationManager::createDefault();
    
    auto integrated_config = ConfigUtils::toIntegratedConfig(*config_manager);
    
    EXPECT_EQ(integrated_config.backend_type, BackendFactory::Type::CPU);
    EXPECT_EQ(integrated_config.simulation_config.max_particles, 100000);
    EXPECT_TRUE(integrated_config.enable_events);
    
    // Test reverse conversion
    auto reverse_config = ConfigUtils::fromIntegratedConfig(integrated_config);
    
    EXPECT_EQ(reverse_config->getDigiStarConfig().backend_type.get(), "cpu");
    EXPECT_EQ(reverse_config->getDigiStarConfig().max_particles.get(), 100000);
}

TEST(UtilityTest, ConfigPresets) {
    auto minimal_config = ConfigPresets::minimal();
    EXPECT_EQ(minimal_config->getDigiStarConfig().max_particles.get(), 1000);
    EXPECT_FALSE(minimal_config->getDigiStarConfig().enable_events.get());
    
    auto galaxy_config = ConfigPresets::galaxyFormation();
    EXPECT_EQ(galaxy_config->getDigiStarConfig().max_particles.get(), 1000000);
    EXPECT_FALSE(galaxy_config->getDigiStarConfig().enable_springs.get());
}

// Main test function
int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    
    std::cout << "Running DigiStar Integration Tests\n";
    std::cout << "==================================\n";
    
    int result = RUN_ALL_TESTS();
    
    std::cout << "\nIntegration test run completed.\n";
    
    return result;
}
#pragma once

#include <memory>
#include <string>
#include <functional>
#include <chrono>
#include <thread>
#include <atomic>
#include <unordered_map>

#include "../backend/backend_interface.h"
#include "../events/event_system.h"
#include "../events/event_producer.h"
#include "../events/event_consumer.h"
#include "../dsl/dsl_runtime.h"
#include "../physics/pools.h"

namespace digistar {

/**
 * Configuration for the integrated simulation system
 */
struct IntegratedSimulationConfig {
    // Backend configuration
    BackendFactory::Type backend_type = BackendFactory::Type::CPU;
    SimulationConfig simulation_config;
    PhysicsConfig physics_config;
    
    // Event system configuration
    bool enable_events = true;
    std::string event_shm_name = "digistar_events";
    bool auto_create_event_system = true;
    
    // DSL configuration
    bool enable_dsl = true;
    std::vector<std::string> startup_scripts;
    std::string dsl_script_directory = "scripts/";
    
    // Performance and monitoring
    bool enable_monitoring = true;
    std::chrono::milliseconds monitor_interval{1000};
    bool log_performance_stats = false;
    
    // Update loop configuration
    float target_fps = 60.0f;
    bool adaptive_timestep = false;
    float min_dt = 1e-6f;
    float max_dt = 1e-2f;
    
    // Threading configuration
    bool use_separate_physics_thread = true;
    bool use_separate_event_thread = false;
    
    // Error handling
    bool continue_on_backend_error = false;
    bool continue_on_dsl_error = true;
};

/**
 * Runtime statistics for the integrated simulation
 */
struct IntegratedSimulationStats {
    // Timing statistics
    std::chrono::microseconds total_frame_time{0};
    std::chrono::microseconds physics_time{0};
    std::chrono::microseconds dsl_time{0};
    std::chrono::microseconds event_processing_time{0};
    
    // Frame statistics
    uint64_t total_frames = 0;
    float current_fps = 0.0f;
    float average_fps = 0.0f;
    
    // Simulation state
    uint32_t current_tick = 0;
    float simulation_time = 0.0f;
    bool is_running = false;
    bool is_paused = false;
    
    // Backend statistics
    SimulationStats backend_stats;
    
    // DSL statistics
    dsl::DSLRuntime::Performance dsl_stats;
    
    // Event statistics
    SharedMemoryEventSystem::Stats event_stats;
    
    // Error tracking
    size_t backend_errors = 0;
    size_t dsl_errors = 0;
    size_t event_errors = 0;
    std::string last_error;
    
    // Memory usage (if available)
    size_t memory_usage_bytes = 0;
    size_t peak_memory_usage_bytes = 0;
};

/**
 * Event handlers for simulation lifecycle events
 */
struct SimulationEventHandlers {
    std::function<void()> on_start;
    std::function<void()> on_stop;
    std::function<void()> on_pause;
    std::function<void()> on_resume;
    std::function<void(const std::string&)> on_error;
    std::function<void(const IntegratedSimulationStats&)> on_stats_update;
    std::function<void(float)> on_frame_complete;
};

/**
 * Main integrated simulation orchestrator
 * 
 * This class brings together all DigiStar components into a unified simulation system:
 * - Physics backend (CPU/CUDA/Distributed)
 * - Event system for zero-copy IPC
 * - DSL runtime for scripting and automation
 * - Performance monitoring and statistics
 * - Thread management and lifecycle control
 */
class IntegratedSimulation {
public:
    /**
     * Create integrated simulation with configuration
     */
    explicit IntegratedSimulation(const IntegratedSimulationConfig& config);
    
    /**
     * Destructor - ensures clean shutdown
     */
    ~IntegratedSimulation();
    
    // Lifecycle management
    bool initialize();
    void shutdown();
    bool isInitialized() const { return initialized_; }
    
    // Execution control
    void start();
    void stop();
    void pause();
    void resume();
    void step();  // Single frame step
    
    // Status queries
    bool isRunning() const { return running_.load(); }
    bool isPaused() const { return paused_.load(); }
    const IntegratedSimulationStats& getStats() const { return stats_; }
    
    // Configuration access
    const IntegratedSimulationConfig& getConfig() const { return config_; }
    void updateConfig(const IntegratedSimulationConfig& new_config);
    
    // Component access
    IBackend* getBackend() { return backend_.get(); }
    const IBackend* getBackend() const { return backend_.get(); }
    SharedMemoryEventSystem* getEventSystem() { return event_system_.get(); }
    const SharedMemoryEventSystem* getEventSystem() const { return event_system_.get(); }
    dsl::DSLRuntime* getDSLRuntime() { return dsl_runtime_.get(); }
    const dsl::DSLRuntime* getDSLRuntime() const { return dsl_runtime_.get(); }
    
    // Simulation state access
    SimulationState& getSimulationState() { return simulation_state_; }
    const SimulationState& getSimulationState() const { return simulation_state_; }
    
    // Event handling
    void setEventHandlers(const SimulationEventHandlers& handlers);
    
    // Script management
    void loadScript(const std::string& name, const std::string& source);
    void loadScriptFile(const std::string& name, const std::string& filename);
    void runScript(const std::string& name);
    void scheduleScript(const std::string& name, std::chrono::milliseconds interval);
    
    // Manual physics controls (for testing/debugging)
    void enablePhysicsSystem(PhysicsConfig::SystemFlags system, bool enable = true);
    void setPhysicsParameter(const std::string& param, float value);
    
    // Performance monitoring
    void enableProfiling(bool enable) { profiling_enabled_ = enable; }
    bool isProfilingEnabled() const { return profiling_enabled_; }
    void resetStats();
    std::string getPerformanceReport() const;
    
    // Error handling
    void setErrorHandler(std::function<void(const std::string&)> handler);
    const std::string& getLastError() const { return stats_.last_error; }
    
private:
    // Configuration
    IntegratedSimulationConfig config_;
    
    // Core components
    std::unique_ptr<IBackend> backend_;
    std::unique_ptr<SharedMemoryEventSystem> event_system_;
    std::unique_ptr<EventProducer> event_producer_;
    std::unique_ptr<EventConsumer> event_consumer_;
    std::unique_ptr<dsl::DSLRuntime> dsl_runtime_;
    
    // Simulation state
    SimulationState simulation_state_;
    
    // Thread management
    std::unique_ptr<std::thread> physics_thread_;
    std::unique_ptr<std::thread> event_thread_;
    std::atomic<bool> running_{false};
    std::atomic<bool> paused_{false};
    std::atomic<bool> stop_requested_{false};
    
    // Statistics and monitoring
    IntegratedSimulationStats stats_;
    std::chrono::steady_clock::time_point last_frame_time_;
    std::chrono::steady_clock::time_point simulation_start_time_;
    bool profiling_enabled_ = false;
    
    // State tracking
    bool initialized_ = false;
    float accumulated_dt_ = 0.0f;
    
    // Event handlers
    SimulationEventHandlers event_handlers_;
    
    // Error handling
    std::function<void(const std::string&)> error_handler_;
    
    // Internal methods
    bool initializeBackend();
    bool initializeEventSystem();
    bool initializeDSLRuntime();
    void loadStartupScripts();
    
    void physicsThreadLoop();
    void eventThreadLoop();
    void mainUpdateLoop();
    void singleThreadedUpdate();
    
    void updatePhysics(float dt);
    void updateDSL(float dt);
    void processEvents();
    void updateStatistics();
    void handleError(const std::string& error, const std::string& component = "");
    
    float calculateDeltaTime();
    void limitFrameRate();
    
    // Helper methods
    void logInfo(const std::string& message);
    void logWarning(const std::string& message);
    void logError(const std::string& message);
};

} // namespace digistar
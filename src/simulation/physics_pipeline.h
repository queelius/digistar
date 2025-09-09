#pragma once

#include <memory>
#include <vector>
#include <functional>
#include <unordered_map>
#include <queue>
#include <mutex>

#include "../backend/backend_interface.h"
#include "../events/event_system.h"
#include "../events/event_producer.h"
#include "../dsl/command.h"

namespace digistar {

/**
 * Command types for physics pipeline operations
 */
enum class PipelineCommandType {
    // Particle operations
    CREATE_PARTICLE,
    DESTROY_PARTICLE,
    MODIFY_PARTICLE,
    APPLY_FORCE,
    
    // Spring operations
    CREATE_SPRING,
    DESTROY_SPRING,
    MODIFY_SPRING,
    
    // Contact operations
    ENABLE_CONTACTS,
    DISABLE_CONTACTS,
    
    // Composite operations
    CREATE_COMPOSITE,
    DESTROY_COMPOSITE,
    
    // Physics system controls
    ENABLE_SYSTEM,
    DISABLE_SYSTEM,
    SET_PARAMETER,
    
    // Procedural generation
    GENERATE_OBJECTS,
    
    // System controls
    RESET_SIMULATION,
    SAVE_STATE,
    LOAD_STATE
};

/**
 * Command for physics pipeline operations
 */
struct PipelineCommand {
    PipelineCommandType type;
    uint32_t priority = 100;  // Lower = higher priority
    
    // Target identification
    uint32_t target_id = 0xFFFFFFFF;
    std::vector<uint32_t> target_ids;
    
    // Parameters
    std::unordered_map<std::string, float> float_params;
    std::unordered_map<std::string, int> int_params;
    std::unordered_map<std::string, std::string> string_params;
    
    // Position/vector data
    float x = 0.0f, y = 0.0f;
    float vx = 0.0f, vy = 0.0f;
    float fx = 0.0f, fy = 0.0f;
    
    // Generic properties
    float mass = 1.0f;
    float radius = 1.0f;
    float temperature = 300.0f;
    float charge = 0.0f;
    
    // Callback for completion notification
    std::function<void(bool success, const std::string& error)> completion_callback;
    
    // Timestamp for ordering
    std::chrono::steady_clock::time_point timestamp = std::chrono::steady_clock::now();
};

/**
 * Event filtering and processing configuration
 */
struct EventProcessingConfig {
    // Event type filters
    std::unordered_set<EventType> enabled_events;
    std::unordered_set<EventType> high_priority_events;
    
    // Spatial filtering
    bool enable_spatial_filtering = false;
    float spatial_filter_radius = 1000.0f;
    float spatial_filter_x = 0.0f;
    float spatial_filter_y = 0.0f;
    
    // Rate limiting
    bool enable_rate_limiting = false;
    size_t max_events_per_frame = 1000;
    
    // Aggregation
    bool enable_event_aggregation = false;
    std::chrono::milliseconds aggregation_window{100};
};

/**
 * Physics pipeline statistics
 */
struct PipelineStats {
    // Command processing
    uint64_t total_commands_processed = 0;
    uint64_t commands_pending = 0;
    uint64_t commands_failed = 0;
    std::chrono::microseconds command_processing_time{0};
    
    // Event generation
    uint64_t events_generated = 0;
    uint64_t events_filtered = 0;
    uint64_t events_aggregated = 0;
    std::chrono::microseconds event_processing_time{0};
    
    // Performance
    float commands_per_second = 0.0f;
    float events_per_second = 0.0f;
    size_t peak_queue_size = 0;
    
    // Error tracking
    std::string last_error;
    std::unordered_map<PipelineCommandType, size_t> failed_commands_by_type;
};

/**
 * Event aggregation for reducing event volume
 */
class EventAggregator {
public:
    struct AggregatedEvent {
        EventType type;
        uint32_t count = 1;
        float sum_magnitude = 0.0f;
        float max_magnitude = 0.0f;
        float center_x = 0.0f;
        float center_y = 0.0f;
        std::chrono::steady_clock::time_point first_time;
        std::chrono::steady_clock::time_point last_time;
        std::vector<uint32_t> participant_ids;
    };
    
    void addEvent(const Event& event);
    std::vector<Event> flushAggregated(std::chrono::milliseconds max_age);
    void clear();
    size_t getPendingCount() const;
    
private:
    std::unordered_map<uint64_t, AggregatedEvent> aggregated_events_;
    std::mutex aggregation_mutex_;
    
    uint64_t makeAggregationKey(EventType type, float x, float y, uint32_t spatial_bucket = 32);
};

/**
 * Physics pipeline that orchestrates physics updates and event emission
 * 
 * This class acts as the bridge between the simulation control layer and the
 * physics backend. It processes commands from the DSL system, coordinates
 * physics updates, and emits events to the event system.
 * 
 * Key responsibilities:
 * - Command queue management and processing
 * - Physics backend coordination
 * - Event generation and filtering
 * - Performance monitoring and statistics
 * - Error handling and recovery
 */
class PhysicsPipeline {
public:
    /**
     * Constructor
     */
    explicit PhysicsPipeline(std::shared_ptr<IBackend> backend,
                            std::shared_ptr<EventProducer> event_producer = nullptr);
    
    /**
     * Destructor
     */
    ~PhysicsPipeline();
    
    // Initialization and configuration
    void initialize(const SimulationConfig& config);
    void shutdown();
    bool isInitialized() const { return initialized_; }
    
    // Event processing configuration
    void setEventProcessingConfig(const EventProcessingConfig& config);
    const EventProcessingConfig& getEventProcessingConfig() const { return event_config_; }
    
    // Command queue management
    void enqueueCommand(const PipelineCommand& command);
    void enqueueCommands(const std::vector<PipelineCommand>& commands);
    size_t getQueueSize() const;
    void clearQueue();
    
    // High-level command builders
    PipelineCommand createParticle(float x, float y, float mass = 1.0f, float radius = 1.0f);
    PipelineCommand destroyParticle(uint32_t particle_id);
    PipelineCommand applyForce(uint32_t particle_id, float fx, float fy);
    PipelineCommand createSpring(uint32_t p1_id, uint32_t p2_id, float stiffness, float damping);
    PipelineCommand generateGalaxy(float x, float y, size_t count, float radius);
    
    // Physics update coordination
    void update(SimulationState& state, const PhysicsConfig& physics_config, float dt);
    
    // Event system integration
    void setEventProducer(std::shared_ptr<EventProducer> producer);
    std::shared_ptr<EventProducer> getEventProducer() const { return event_producer_; }
    
    // Statistics and monitoring
    const PipelineStats& getStats() const { return stats_; }
    void resetStats();
    std::string getStatsReport() const;
    
    // Error handling
    void setErrorHandler(std::function<void(const std::string&, const PipelineCommand&)> handler);
    
    // Advanced features
    void enableBatchProcessing(bool enable, size_t batch_size = 100);
    void enableAsyncProcessing(bool enable);
    void setMaxQueueSize(size_t max_size);
    
private:
    // Core components
    std::shared_ptr<IBackend> backend_;
    std::shared_ptr<EventProducer> event_producer_;
    
    // Configuration
    EventProcessingConfig event_config_;
    bool initialized_ = false;
    
    // Command queue
    std::priority_queue<PipelineCommand, std::vector<PipelineCommand>, 
                       std::function<bool(const PipelineCommand&, const PipelineCommand&)>> command_queue_;
    std::mutex queue_mutex_;
    size_t max_queue_size_ = 10000;
    
    // Event processing
    std::unique_ptr<EventAggregator> event_aggregator_;
    std::vector<Event> pending_events_;
    std::mutex event_mutex_;
    
    // Performance tracking
    PipelineStats stats_;
    std::chrono::steady_clock::time_point last_stats_update_;
    
    // Batch processing
    bool batch_processing_enabled_ = false;
    size_t batch_size_ = 100;
    std::vector<PipelineCommand> command_batch_;
    
    // Async processing
    bool async_processing_enabled_ = false;
    std::unique_ptr<std::thread> processing_thread_;
    std::atomic<bool> processing_active_{false};
    std::condition_variable processing_cv_;
    
    // Error handling
    std::function<void(const std::string&, const PipelineCommand&)> error_handler_;
    
    // Internal methods
    void processCommands(SimulationState& state);
    void processCommand(const PipelineCommand& command, SimulationState& state);
    void processBatch(const std::vector<PipelineCommand>& batch, SimulationState& state);
    
    void generateEvents(const SimulationState& state, float dt);
    void processEvents();
    void emitEvent(const Event& event);
    bool shouldFilterEvent(const Event& event) const;
    
    void updateStatistics();
    void asyncProcessingLoop();
    
    // Command processors
    bool processCreateParticle(const PipelineCommand& command, SimulationState& state);
    bool processDestroyParticle(const PipelineCommand& command, SimulationState& state);
    bool processApplyForce(const PipelineCommand& command, SimulationState& state);
    bool processCreateSpring(const PipelineCommand& command, SimulationState& state);
    bool processGenerateObjects(const PipelineCommand& command, SimulationState& state);
    
    // Helper methods
    void logInfo(const std::string& message);
    void logWarning(const std::string& message);
    void logError(const std::string& message);
    void handleCommandError(const std::string& error, const PipelineCommand& command);
    
    // Priority comparison for command queue
    struct CommandComparator {
        bool operator()(const PipelineCommand& a, const PipelineCommand& b) const {
            if (a.priority != b.priority) {
                return a.priority > b.priority;  // Lower priority value = higher priority
            }
            return a.timestamp > b.timestamp;  // Earlier timestamp = higher priority
        }
    };
};

} // namespace digistar
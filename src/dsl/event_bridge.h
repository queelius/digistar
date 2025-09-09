#pragma once

#include "sexpr.h"
#include "pattern_matcher.h"
#include "bytecode_compiler.h"
#include "bytecode_vm.h"
#include "../events/event_system.h"
#include "../events/event_producer.h"
#include "../events/event_consumer.h"
#include <functional>
#include <queue>
#include <thread>
#include <atomic>

namespace digistar {

// Forward declaration
struct SimulationState;

namespace dsl {

/**
 * Event handler registration
 */
struct EventHandler {
    EventType event_type;
    PatternPtr pattern;                    // Pattern to match against event
    std::shared_ptr<BytecodeChunk> action; // Compiled action to execute
    int priority;                           // Handler priority (higher = first)
    bool active;                           // Is handler active?
    
    EventHandler(EventType type, PatternPtr pat, 
                std::shared_ptr<BytecodeChunk> act, int prio = 0)
        : event_type(type), pattern(pat), action(act), 
          priority(prio), active(true) {}
};

/**
 * Event to S-expression converter
 */
class EventConverter {
public:
    // Convert Event struct to S-expression for pattern matching
    static SExprPtr eventToSExpr(const Event& event);
    
    // Convert S-expression to Event struct for emission
    static Event sExprToEvent(SExprPtr expr);
    
    // Extract specific event data
    static SExprPtr extractCollisionData(const Event& event);
    static SExprPtr extractMergeData(const Event& event);
    static SExprPtr extractSpringData(const Event& event);
    static SExprPtr extractThermalData(const Event& event);
    
    // Create event S-expressions
    static SExprPtr makeCollisionEvent(uint32_t p1, uint32_t p2, 
                                       float energy, float impulse);
    static SExprPtr makeMergeEvent(uint32_t p1, uint32_t p2,
                                   float mass_before, float mass_after);
    static SExprPtr makeSpringEvent(uint32_t p1, uint32_t p2,
                                    float stress, bool broken);
};

/**
 * Event system bridge for DSL integration
 * 
 * Features:
 * - Event pattern matching
 * - Reactive programming primitives
 * - Event emission from DSL
 * - Handler priority and filtering
 * - Async event processing
 */
class EventBridge : public EventConsumer {
private:
    
    // Event handlers organized by type
    std::unordered_map<EventType, std::vector<dsl::EventHandler>> handlers;
    
    // Event queue for processing
    std::queue<Event> event_queue;
    std::mutex queue_mutex;
    std::condition_variable queue_cv;
    
    // VM for executing handlers
    std::unique_ptr<BytecodeVM> vm;
    BytecodeCompiler compiler;
    
    // Processing thread
    std::thread processor_thread;
    std::atomic<bool> should_stop{false};
    
    // Statistics
    struct Stats {
        std::atomic<size_t> events_received{0};
        std::atomic<size_t> events_processed{0};
        std::atomic<size_t> handlers_triggered{0};
        std::atomic<size_t> patterns_matched{0};
        std::atomic<size_t> patterns_failed{0};
    } stats;
    
    // Event producer for emitting events
    std::unique_ptr<EventProducer> producer;
    
    // Processing methods
    void processEventQueue();
    void handleEvent(const Event& event);
    bool matchAndExecute(const Event& event, const dsl::EventHandler& handler);
    
public:
    EventBridge(EventRingBuffer* buffer, SimulationState* state = nullptr);
    ~EventBridge();
    
    // Event handling
    void onEvent(const Event& event);
    
    // Handler registration
    void registerHandler(EventType type, PatternPtr pattern,
                         SExprPtr action, int priority = 0);
    void registerHandler(EventType type, PatternPtr pattern,
                         std::shared_ptr<BytecodeChunk> action, int priority = 0);
    
    // DSL convenience methods
    void onCollision(PatternPtr pattern, SExprPtr action);
    void onMerge(PatternPtr pattern, SExprPtr action);
    void onSpringBreak(PatternPtr pattern, SExprPtr action);
    void onPhaseTransition(PatternPtr pattern, SExprPtr action);
    
    // Event emission from DSL
    void emitEvent(EventType type, const std::unordered_map<std::string, SExprPtr>& data);
    void emitCollision(uint32_t p1, uint32_t p2, float energy);
    void emitCustomEvent(SExprPtr event_expr);
    
    // Handler management
    void enableHandler(EventType type, size_t index);
    void disableHandler(EventType type, size_t index);
    void removeHandler(EventType type, size_t index);
    void clearHandlers(EventType type);
    void clearAllHandlers();
    
    // Reactive programming primitives
    void whenChanged(const std::string& property,
                     std::function<void(SExprPtr old_val, SExprPtr new_val)> callback);
    void debounce(EventType type, std::chrono::milliseconds delay,
                  std::function<void(const Event&)> callback);
    void throttle(EventType type, std::chrono::milliseconds interval,
                  std::function<void(const Event&)> callback);
    
    // Event filtering
    void setEventFilter(std::function<bool(const Event&)> filter);
    void setSpatialFilter(float x, float y, float radius);
    void setPriorityThreshold(int min_priority);
    
    // Statistics
    const Stats& getStats() const { return stats; }
    void resetStats();
    
    // Control
    void start();
    void stop();
    void pause();
    void resume();
    
    // DSL integration helpers
    static void registerEventFunctions(std::shared_ptr<Environment> env,
                                      EventBridge* bridge);
};

/**
 * Event-driven behavior system
 */
class EventDrivenBehavior {
private:
    std::string name;
    std::vector<std::pair<PatternPtr, std::shared_ptr<BytecodeChunk>>> rules;
    std::shared_ptr<Environment> state_env;  // Behavior state
    bool active = true;
    
public:
    EventDrivenBehavior(const std::string& n) 
        : name(n), state_env(std::make_shared<Environment>()) {}
    
    // Add behavior rule
    void addRule(PatternPtr trigger, std::shared_ptr<BytecodeChunk> action) {
        rules.emplace_back(trigger, action);
    }
    
    // Process event
    bool processEvent(const Event& event, BytecodeVM& vm);
    
    // State management
    void setState(const std::string& key, SExprPtr value) {
        state_env->define(key, value);
    }
    
    SExprPtr getState(const std::string& key) const {
        return state_env->lookup(key);
    }
    
    // Control
    void activate() { active = true; }
    void deactivate() { active = false; }
    bool isActive() const { return active; }
};

/**
 * Complex event processing (CEP) engine
 */
class ComplexEventProcessor {
private:
    // Event window for temporal patterns
    struct EventWindow {
        std::deque<std::pair<Event, std::chrono::steady_clock::time_point>> events;
        std::chrono::milliseconds window_size;
        
        void add(const Event& e);
        void removeOld();
        std::vector<Event> getEvents() const;
    };
    
    // Pattern definitions for complex events
    struct ComplexPattern {
        std::string name;
        std::vector<PatternPtr> sequence;  // Sequence of patterns
        std::chrono::milliseconds time_window;
        std::function<void(const std::vector<Event>&)> action;
    };
    
    std::vector<ComplexPattern> patterns;
    std::unordered_map<std::string, EventWindow> windows;
    
public:
    // Register complex event patterns
    void registerSequence(const std::string& name,
                         const std::vector<PatternPtr>& sequence,
                         std::chrono::milliseconds window,
                         std::function<void(const std::vector<Event>&)> action);
    
    void registerAggregate(const std::string& name,
                          PatternPtr pattern,
                          size_t count,
                          std::chrono::milliseconds window,
                          std::function<void(const std::vector<Event>&)> action);
    
    // Process incoming event
    void processEvent(const Event& event);
    
    // Query event history
    std::vector<Event> queryEvents(PatternPtr pattern,
                                   std::chrono::milliseconds lookback);
};

/**
 * Event DSL macros and forms
 */
class EventDSL {
public:
    // Create DSL forms for event handling
    static SExprPtr createOnEventForm(const std::string& event_type,
                                      PatternPtr pattern,
                                      SExprPtr action);
    
    static SExprPtr createWhenChangeForm(const std::string& property,
                                        SExprPtr action);
    
    static SExprPtr createEventStreamForm(const std::vector<std::string>& event_types,
                                         SExprPtr transformer);
    
    // Parse event handling expressions
    static void parseEventHandler(SExprPtr expr,
                                 EventBridge* bridge);
    
    // Register event DSL in environment
    static void registerEventDSL(std::shared_ptr<Environment> env,
                                EventBridge* bridge);
};

} // namespace dsl
} // namespace digistar
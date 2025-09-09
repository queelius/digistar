#include "event_bridge.h"
#include <algorithm>
#include <sstream>

namespace digistar {
namespace dsl {

//=============================================================================
// EventConverter implementation
//=============================================================================

SExprPtr EventConverter::eventToSExpr(const Event& event) {
    std::vector<SExprPtr> elements;
    
    // Add event type as symbol
    std::string type_name;
    switch (event.type) {
        case EventType::HARD_COLLISION: type_name = "collision"; break;
        case EventType::PARTICLE_MERGE: type_name = "merge"; break;
        case EventType::SPRING_BROKEN: type_name = "spring-break"; break;
        case EventType::PHASE_TRANSITION: type_name = "phase-transition"; break;
        case EventType::FUSION_IGNITION: type_name = "fusion"; break;
        default: type_name = "unknown"; break;
    }
    elements.push_back(SExpr::makeSymbol(type_name));
    
    // Add primary and secondary IDs
    elements.push_back(SExpr::makeNumber(event.primary_id));
    if (event.secondary_id != 0xFFFFFFFF) {
        elements.push_back(SExpr::makeNumber(event.secondary_id));
    }
    
    // Add magnitude
    elements.push_back(SExpr::makeNumber(event.magnitude));
    
    // Add position
    elements.push_back(SExpr::makeVector({event.x, event.y}));
    
    // Add type-specific data
    switch (event.type) {
        case EventType::HARD_COLLISION:
            elements.push_back(SExpr::makeList({
                SExpr::makeSymbol("impulse"),
                SExpr::makeNumber(event.data.collision.impulse),
                SExpr::makeSymbol("velocity"),
                SExpr::makeVector({event.data.collision.velocity_x, 
                                  event.data.collision.velocity_y})
            }));
            break;
        case EventType::PARTICLE_MERGE:
            elements.push_back(SExpr::makeList({
                SExpr::makeSymbol("mass-before"),
                SExpr::makeNumber(event.data.merge.mass_before),
                SExpr::makeSymbol("mass-after"),
                SExpr::makeNumber(event.data.merge.mass_after)
            }));
            break;
        default:
            break;
    }
    
    return SExpr::makeList(elements);
}

Event EventConverter::sExprToEvent(SExprPtr expr) {
    Event event;
    event.tick = 0;
    event.timestamp = 0.0f;
    event.flags = EventFlags::FLAG_NONE;
    
    if (!expr || !expr->isList() || expr->length() < 2) {
        event.type = EventType::TICK_COMPLETE;  // Default
        return event;
    }
    
    const auto& list = expr->asList();
    
    // Parse event type
    if (list[0]->isSymbol()) {
        const std::string& type = list[0]->asSymbol();
        if (type == "collision") event.type = EventType::HARD_COLLISION;
        else if (type == "merge") event.type = EventType::PARTICLE_MERGE;
        else if (type == "spring-break") event.type = EventType::SPRING_BROKEN;
        else if (type == "fusion") event.type = EventType::FUSION_IGNITION;
    }
    
    // Parse IDs
    if (list.size() > 1 && list[1]->isNumber()) {
        event.primary_id = static_cast<uint32_t>(list[1]->asNumber());
    }
    if (list.size() > 2 && list[2]->isNumber()) {
        event.secondary_id = static_cast<uint32_t>(list[2]->asNumber());
    } else {
        event.secondary_id = 0xFFFFFFFF;
    }
    
    // Parse magnitude
    if (list.size() > 3 && list[3]->isNumber()) {
        event.magnitude = list[3]->asNumber();
    }
    
    // Parse position
    if (list.size() > 4 && list[4]->isVector()) {
        const auto& pos = list[4]->asVector();
        if (pos.size() >= 2) {
            event.x = pos[0];
            event.y = pos[1];
        }
    }
    
    return event;
}

SExprPtr EventConverter::makeCollisionEvent(uint32_t p1, uint32_t p2, 
                                           float energy, float impulse) {
    return SExpr::makeList({
        SExpr::makeSymbol("collision"),
        SExpr::makeNumber(p1),
        SExpr::makeNumber(p2),
        SExpr::makeNumber(energy),
        SExpr::makeList({
            SExpr::makeSymbol("impulse"),
            SExpr::makeNumber(impulse)
        })
    });
}

//=============================================================================
// EventBridge implementation
//=============================================================================

EventBridge::EventBridge(EventRingBuffer* buffer, SimulationState* state)
    : EventConsumer(buffer, "dsl_bridge") {
    
    vm = std::make_unique<BytecodeVM>(state);
    producer = std::make_unique<EventProducer>(buffer);
    
    // Start processing thread
    processor_thread = std::thread([this]() {
        processEventQueue();
    });
}

EventBridge::~EventBridge() {
    stop();
}

void EventBridge::onEvent(const Event& event) {
    // Add to queue for processing
    {
        std::lock_guard<std::mutex> lock(queue_mutex);
        event_queue.push(event);
    }
    queue_cv.notify_one();
    
    stats.events_received++;
}

void EventBridge::processEventQueue() {
    while (!should_stop) {
        std::unique_lock<std::mutex> lock(queue_mutex);
        queue_cv.wait(lock, [this] { return !event_queue.empty() || should_stop; });
        
        while (!event_queue.empty()) {
            Event event = event_queue.front();
            event_queue.pop();
            lock.unlock();
            
            handleEvent(event);
            stats.events_processed++;
            
            lock.lock();
        }
    }
}

void EventBridge::handleEvent(const Event& event) {
    // Find handlers for this event type
    auto it = handlers.find(event.type);
    if (it == handlers.end()) {
        return;
    }
    
    // Sort handlers by priority
    auto sorted_handlers = it->second;
    std::sort(sorted_handlers.begin(), sorted_handlers.end(),
              [](const dsl::EventHandler& a, const dsl::EventHandler& b) {
                  return a.priority > b.priority;
              });
    
    // Try each handler
    for (const auto& handler : sorted_handlers) {
        if (!handler.active) continue;
        
        if (matchAndExecute(event, handler)) {
            stats.handlers_triggered++;
            break;  // Stop after first successful match (can be configured)
        }
    }
}

bool EventBridge::matchAndExecute(const Event& event, const dsl::EventHandler& handler) {
    // Convert event to S-expression
    auto event_sexpr = EventConverter::eventToSExpr(event);
    
    // Try pattern matching
    auto match_result = handler.pattern->match(event_sexpr);
    
    if (match_result.success) {
        stats.patterns_matched++;
        
        // Set up environment with bindings
        auto env = std::make_shared<Environment>();
        for (const auto& [name, value] : match_result.bindings) {
            env->define(name, value);
        }
        
        // Execute action
        vm->setGlobalEnv(env);
        vm->execute(handler.action);
        
        return true;
    } else {
        stats.patterns_failed++;
        return false;
    }
}

void EventBridge::registerHandler(EventType type, PatternPtr pattern,
                                 SExprPtr action, int priority) {
    auto compiled = compiler.compile(action);
    registerHandler(type, pattern, compiled, priority);
}

void EventBridge::registerHandler(EventType type, PatternPtr pattern,
                                 std::shared_ptr<BytecodeChunk> action, int priority) {
    handlers[type].emplace_back(type, pattern, action, priority);
}

void EventBridge::onCollision(PatternPtr pattern, SExprPtr action) {
    registerHandler(EventType::HARD_COLLISION, pattern, action);
}

void EventBridge::onMerge(PatternPtr pattern, SExprPtr action) {
    registerHandler(EventType::PARTICLE_MERGE, pattern, action);
}

void EventBridge::onSpringBreak(PatternPtr pattern, SExprPtr action) {
    registerHandler(EventType::SPRING_BROKEN, pattern, action);
}

void EventBridge::onPhaseTransition(PatternPtr pattern, SExprPtr action) {
    registerHandler(EventType::PHASE_TRANSITION, pattern, action);
}

void EventBridge::emitEvent(EventType type, 
                           const std::unordered_map<std::string, SExprPtr>& data) {
    Event event;
    event.type = type;
    event.tick = 0;  // Would get from simulation
    event.timestamp = 0.0f;  // Would get from simulation
    event.flags = EventFlags::FLAG_NONE;
    
    // Extract common fields
    auto it = data.find("primary");
    if (it != data.end() && it->second->isNumber()) {
        event.primary_id = static_cast<uint32_t>(it->second->asNumber());
    }
    
    it = data.find("secondary");
    if (it != data.end() && it->second->isNumber()) {
        event.secondary_id = static_cast<uint32_t>(it->second->asNumber());
    } else {
        event.secondary_id = 0xFFFFFFFF;
    }
    
    it = data.find("magnitude");
    if (it != data.end() && it->second->isNumber()) {
        event.magnitude = it->second->asNumber();
    }
    
    it = data.find("position");
    if (it != data.end() && it->second->isVector()) {
        const auto& pos = it->second->asVector();
        if (pos.size() >= 2) {
            event.x = pos[0];
            event.y = pos[1];
        }
    }
    
    // Emit through producer
    producer->emit(event);
}

void EventBridge::emitCollision(uint32_t p1, uint32_t p2, float energy) {
    Event event;
    event.type = EventType::HARD_COLLISION;
    event.primary_id = p1;
    event.secondary_id = p2;
    event.magnitude = energy;
    event.tick = 0;
    event.timestamp = 0.0f;
    event.flags = EventFlags::FLAG_NONE;
    
    producer->emit(event);
}

void EventBridge::clearHandlers(EventType type) {
    handlers[type].clear();
}

void EventBridge::clearAllHandlers() {
    handlers.clear();
}

void EventBridge::resetStats() {
    stats.events_received.store(0);
    stats.events_processed.store(0);
    stats.handlers_triggered.store(0);
    stats.patterns_matched.store(0);
    stats.patterns_failed.store(0);
}

void EventBridge::start() {
    // EventConsumer automatically starts in constructor
}

void EventBridge::stop() {
    should_stop = true;
    queue_cv.notify_all();
    
    if (processor_thread.joinable()) {
        processor_thread.join();
    }
}

void EventBridge::registerEventFunctions(std::shared_ptr<Environment> env,
                                        EventBridge* bridge) {
    // Register event handling functions in DSL environment
    env->define("on-event", SExpr::makeSymbol("on-event"));
    env->define("emit-event", SExpr::makeSymbol("emit-event"));
    env->define("when-change", SExpr::makeSymbol("when-change"));
}

//=============================================================================
// EventDrivenBehavior implementation
//=============================================================================

bool EventDrivenBehavior::processEvent(const Event& event, BytecodeVM& vm) {
    if (!active) return false;
    
    auto event_sexpr = EventConverter::eventToSExpr(event);
    
    for (const auto& [pattern, action] : rules) {
        auto result = pattern->match(event_sexpr);
        if (result.success) {
            // Set up environment with behavior state and match bindings
            auto exec_env = std::make_shared<Environment>(state_env);
            for (const auto& [name, value] : result.bindings) {
                exec_env->define(name, value);
            }
            
            vm.setGlobalEnv(exec_env);
            vm.execute(action);
            return true;
        }
    }
    
    return false;
}

} // namespace dsl
} // namespace digistar
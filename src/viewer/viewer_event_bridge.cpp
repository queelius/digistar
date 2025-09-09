#include "viewer_event_bridge.h"
#include <chrono>
#include <algorithm>

namespace digistar {

ViewerEventBridge::~ViewerEventBridge() {
    shutdown();
}

bool ViewerEventBridge::initialize(std::shared_ptr<GraphicsViewer> viewer_ptr, const Config& cfg) {
    if (!viewer_ptr) {
        printf("ViewerEventBridge: Invalid viewer pointer\n");
        return false;
    }
    
    viewer = viewer_ptr;
    config = cfg;
    
    // Create event consumer
    event_consumer = std::make_unique<EventConsumer>();
    
    // Connect to shared memory event buffer
    if (!event_consumer->connect(config.shm_name)) {
        printf("ViewerEventBridge: Failed to connect to event buffer '%s'\n", config.shm_name.c_str());
        return false;
    }
    
    // Register as consumer
    if (!event_consumer->registerConsumer(config.consumer_name)) {
        printf("ViewerEventBridge: Failed to register as consumer '%s'\n", config.consumer_name.c_str());
        return false;
    }
    
    printf("ViewerEventBridge: Connected to event buffer '%s' as '%s'\n",
           config.shm_name.c_str(), config.consumer_name.c_str());
    
    // Start processing thread if auto-start is enabled
    if (config.auto_start) {
        return start();
    }
    
    return true;
}

void ViewerEventBridge::shutdown() {
    stop();
    
    if (event_consumer) {
        event_consumer->disconnect();
        event_consumer.reset();
    }
    
    viewer.reset();
}

bool ViewerEventBridge::start() {
    if (is_running.load()) {
        return true;  // Already running
    }
    
    if (!event_consumer) {
        printf("ViewerEventBridge: Cannot start - not initialized\n");
        return false;
    }
    
    should_stop.store(false);
    processing_thread = std::make_unique<std::thread>(&ViewerEventBridge::processingLoop, this);
    
    // Wait a moment to ensure thread started
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    
    return is_running.load();
}

void ViewerEventBridge::stop() {
    if (!is_running.load()) {
        return;  // Not running
    }
    
    should_stop.store(true);
    
    if (processing_thread && processing_thread->joinable()) {
        processing_thread->join();
    }
    
    processing_thread.reset();
    is_running.store(false);
}

size_t ViewerEventBridge::processEvents(size_t max_events) {
    if (!event_consumer || !viewer) {
        return 0;
    }
    
    auto start_time = std::chrono::high_resolution_clock::now();
    size_t processed = 0;
    
    Event event;
    while (processed < max_events && event_consumer->readEvent(event)) {
        // Apply filter if set
        if (event_filter && !event_filter(event)) {
            stats.events_filtered.fetch_add(1);
            continue;
        }
        
        // Check if event should be shown
        if (!shouldShowEvent(event)) {
            stats.events_filtered.fetch_add(1);
            continue;
        }
        
        // Route to appropriate handler
        switch (static_cast<uint16_t>(event.type) & 0xFF00) {
            case 0x0000:  // Particle events
                handleParticleEvent(event);
                break;
                
            case 0x0010:  // Spring events  
                handleSpringEvent(event);
                break;
                
            case 0x0020:  // Collision events
                handleCollisionEvent(event);
                break;
                
            case 0x0030:  // Thermal events
                handleThermalEvent(event);
                break;
                
            case 0x0040:  // Composite events
                handleCompositeEvent(event);
                break;
                
            case 0x0200:  // System events
                handleSystemEvent(event);
                break;
                
            default:
                // Unknown event type - ignore
                break;
        }
        
        stats.events_processed.fetch_add(1);
        processed++;
    }
    
    // Update timing stats
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
    stats.processing_time_ms.store(duration.count() / 1000.0f);
    
    return processed;
}

ViewerEventBridge::ProcessingStats ViewerEventBridge::getStats() const {
    ProcessingStats result;
    result.events_processed = stats.events_processed.load();
    result.events_filtered = stats.events_filtered.load();
    result.visual_events_created = stats.visual_events_created.load();
    result.processing_time_ms = stats.processing_time_ms.load();
    result.is_connected = (event_consumer != nullptr) && event_consumer->isConnected();
    result.consumer_status = event_consumer ? event_consumer->getStatusString() : "Not initialized";
    return result;
}

void ViewerEventBridge::resetStats() {
    stats.events_processed.store(0);
    stats.events_filtered.store(0);
    stats.visual_events_created.store(0);
    stats.processing_time_ms.store(0);
}

void ViewerEventBridge::processingLoop() {
    is_running.store(true);
    
    while (!should_stop.load()) {
        // Process a batch of events
        size_t processed = processEvents(50);  // Process up to 50 events per iteration
        
        // If no events processed, sleep briefly to avoid busy waiting
        if (processed == 0) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    }
    
    is_running.store(false);
}

void ViewerEventBridge::handleParticleEvent(const Event& event) {
    VisualEvent visual_event = createVisualEvent(event);
    
    switch (event.type) {
        case EventType::PARTICLE_MERGE:
            visual_event.type = VisualEvent::EXPLOSION;
            visual_event.color = {255, 200, 100, 255};  // Orange
            visual_event.intensity = std::min(1.0f, event.magnitude / 1000.0f);
            break;
            
        case EventType::PARTICLE_FISSION:
            visual_event.type = VisualEvent::EXPLOSION;
            visual_event.color = {255, 100, 100, 255};  // Red
            visual_event.intensity = std::min(1.0f, event.magnitude / 500.0f);
            visual_event.max_radius = 100.0f;
            break;
            
        case EventType::STAR_FORMATION:
            visual_event.type = VisualEvent::EXPLOSION;
            visual_event.color = {255, 255, 200, 255};  // Bright yellow
            visual_event.intensity = 1.0f;
            visual_event.max_radius = 200.0f;
            visual_event.duration = 5.0f;
            break;
            
        case EventType::BLACK_HOLE_FORMATION:
            visual_event.type = VisualEvent::EXPLOSION;
            visual_event.color = {100, 0, 200, 255};    // Purple
            visual_event.intensity = 1.0f;
            visual_event.max_radius = 150.0f;
            visual_event.duration = 3.0f;
            break;
            
        case EventType::BLACK_HOLE_ABSORPTION:
            visual_event.type = VisualEvent::PARTICLE_DEATH;
            visual_event.color = {50, 0, 50, 255};      // Dark purple
            visual_event.intensity = 0.8f;
            break;
            
        default:
            return;  // Don't create visual event for unknown particle events
    }
    
    viewer->addVisualEvent(visual_event);
    stats.visual_events_created.fetch_add(1);
}

void ViewerEventBridge::handleSpringEvent(const Event& event) {
    VisualEvent visual_event = createVisualEvent(event);
    
    switch (event.type) {
        case EventType::SPRING_CREATED:
            visual_event.type = VisualEvent::COMPOSITE_FORMATION;
            visual_event.color = {100, 255, 100, 255};  // Green
            visual_event.intensity = 0.5f;
            visual_event.max_radius = 20.0f;
            visual_event.duration = 1.0f;
            break;
            
        case EventType::SPRING_BROKEN:
            visual_event.type = VisualEvent::SPRING_BREAK;
            visual_event.color = {255, 150, 50, 255};   // Orange
            visual_event.intensity = std::min(1.0f, event.data.spring.stress / 1000.0f);
            visual_event.max_radius = 30.0f;
            break;
            
        case EventType::SPRING_CRITICAL_STRESS:
            visual_event.type = VisualEvent::SPRING_BREAK;
            visual_event.color = {255, 255, 0, 255};    // Yellow warning
            visual_event.intensity = 0.7f;
            visual_event.max_radius = 15.0f;
            visual_event.duration = 0.5f;
            break;
            
        default:
            return;
    }
    
    viewer->addVisualEvent(visual_event);
    stats.visual_events_created.fetch_add(1);
}

void ViewerEventBridge::handleCollisionEvent(const Event& event) {
    VisualEvent visual_event = createVisualEvent(event);
    
    switch (event.type) {
        case EventType::SOFT_CONTACT:
            // Don't show soft contacts - too frequent and not visually interesting
            return;
            
        case EventType::HARD_COLLISION:
            visual_event.type = VisualEvent::COLLISION;
            visual_event.color = {255, 200, 0, 255};    // Bright yellow
            visual_event.intensity = std::min(1.0f, event.data.collision.impulse / 100.0f);
            visual_event.max_radius = 40.0f;
            break;
            
        case EventType::COMPOSITE_COLLISION:
            visual_event.type = VisualEvent::COLLISION;
            visual_event.color = {255, 100, 200, 255};  // Pink
            visual_event.intensity = std::min(1.0f, event.magnitude / 200.0f);
            visual_event.max_radius = 60.0f;
            visual_event.duration = 1.5f;
            break;
            
        default:
            return;
    }
    
    viewer->addVisualEvent(visual_event);
    stats.visual_events_created.fetch_add(1);
}

void ViewerEventBridge::handleThermalEvent(const Event& event) {
    VisualEvent visual_event = createVisualEvent(event);
    
    switch (event.type) {
        case EventType::PHASE_TRANSITION:
            visual_event.type = VisualEvent::EXPLOSION;
            visual_event.color = {100, 200, 255, 255};  // Light blue
            visual_event.intensity = 0.6f;
            visual_event.max_radius = 25.0f;
            break;
            
        case EventType::FUSION_IGNITION:
            visual_event.type = VisualEvent::EXPLOSION;
            visual_event.color = {255, 255, 255, 255};  // Bright white
            visual_event.intensity = 1.0f;
            visual_event.max_radius = 150.0f;
            visual_event.duration = 4.0f;
            break;
            
        case EventType::THERMAL_EXPLOSION:
            visual_event.type = VisualEvent::EXPLOSION;
            visual_event.color = {255, 50, 0, 255};     // Bright red
            visual_event.intensity = 1.0f;
            visual_event.max_radius = 100.0f;
            visual_event.duration = 3.0f;
            break;
            
        default:
            return;
    }
    
    viewer->addVisualEvent(visual_event);
    stats.visual_events_created.fetch_add(1);
}

void ViewerEventBridge::handleCompositeEvent(const Event& event) {
    VisualEvent visual_event = createVisualEvent(event);
    
    switch (event.type) {
        case EventType::COMPOSITE_FORMED:
            visual_event.type = VisualEvent::COMPOSITE_FORMATION;
            visual_event.color = {0, 255, 0, 255};      // Green
            visual_event.intensity = 0.8f;
            visual_event.max_radius = 50.0f;
            visual_event.duration = 2.0f;
            break;
            
        case EventType::COMPOSITE_BROKEN:
            visual_event.type = VisualEvent::EXPLOSION;
            visual_event.color = {255, 100, 0, 255};    // Orange
            visual_event.intensity = std::min(1.0f, event.data.composite.particle_count / 100.0f);
            visual_event.max_radius = 80.0f;
            break;
            
        case EventType::RESONANCE_DETECTED:
            visual_event.type = VisualEvent::COMPOSITE_FORMATION;
            visual_event.color = {200, 0, 200, 255};    // Purple
            visual_event.intensity = 0.6f;
            visual_event.max_radius = 30.0f;
            visual_event.duration = 1.0f;
            break;
            
        case EventType::STRUCTURAL_FAILURE:
            visual_event.type = VisualEvent::EXPLOSION;
            visual_event.color = {200, 0, 0, 255};      // Dark red
            visual_event.intensity = 0.9f;
            visual_event.max_radius = 70.0f;
            break;
            
        default:
            return;
    }
    
    viewer->addVisualEvent(visual_event);
    stats.visual_events_created.fetch_add(1);
}

void ViewerEventBridge::handleSystemEvent(const Event& event) {
    // System events like TICK_COMPLETE and STATS_UPDATE don't typically
    // generate visual events, but they could be used for other purposes
    // like updating UI overlays or triggering system-wide effects
    
    switch (event.type) {
        case EventType::TICK_COMPLETE:
            // Could be used to trigger periodic UI updates
            break;
            
        case EventType::CHECKPOINT:
            // Could show a brief indicator that a checkpoint was saved
            break;
            
        case EventType::STATS_UPDATE:
            // Could trigger refresh of performance overlays
            break;
            
        default:
            break;
    }
}

bool ViewerEventBridge::shouldShowEvent(const Event& event) const {
    if (!viewer) {
        return false;
    }
    
    // Check distance filtering if enabled
    if (config.filter_by_camera) {
        float distance = getDistanceToCamera(event.x, event.y);
        if (distance > config.max_event_distance) {
            return false;
        }
    }
    
    // Check if event is in camera view (with some margin)
    if (!isInCameraView(event.x, event.y, event.radius + 100.0f)) {
        return false;
    }
    
    return true;
}

VisualEvent ViewerEventBridge::createVisualEvent(const Event& event) const {
    VisualEvent visual_event(VisualEvent::EXPLOSION, event.x, event.y);
    
    visual_event.intensity = getEventIntensity(event);
    visual_event.color = getEventColor(event.type);
    visual_event.duration = getEventDuration(event.type);
    visual_event.time_remaining = visual_event.duration;
    
    return visual_event;
}

SDL_Color ViewerEventBridge::getEventColor(EventType type) const {
    switch (type) {
        case EventType::PARTICLE_MERGE:       return {255, 200, 100, 255};
        case EventType::PARTICLE_FISSION:     return {255, 100, 100, 255};
        case EventType::STAR_FORMATION:       return {255, 255, 200, 255};
        case EventType::BLACK_HOLE_FORMATION: return {100, 0, 200, 255};
        case EventType::SPRING_CREATED:       return {100, 255, 100, 255};
        case EventType::SPRING_BROKEN:        return {255, 150, 50, 255};
        case EventType::HARD_COLLISION:       return {255, 200, 0, 255};
        case EventType::FUSION_IGNITION:      return {255, 255, 255, 255};
        case EventType::THERMAL_EXPLOSION:    return {255, 50, 0, 255};
        case EventType::COMPOSITE_FORMED:     return {0, 255, 0, 255};
        case EventType::COMPOSITE_BROKEN:     return {255, 100, 0, 255};
        default:                              return {255, 255, 255, 255};
    }
}

float ViewerEventBridge::getEventIntensity(const Event& event) const {
    // Base intensity on event magnitude, clamped to 0-1 range
    float base_intensity = std::min(1.0f, std::max(0.1f, event.magnitude / 1000.0f));
    
    // Modify based on event type
    switch (event.type) {
        case EventType::STAR_FORMATION:
        case EventType::BLACK_HOLE_FORMATION:
        case EventType::FUSION_IGNITION:
            return 1.0f;  // Always maximum intensity
            
        case EventType::SOFT_CONTACT:
            return 0.1f;  // Very subtle
            
        default:
            return base_intensity;
    }
}

float ViewerEventBridge::getEventDuration(EventType type) const {
    switch (type) {
        case EventType::STAR_FORMATION:       return 5.0f;
        case EventType::BLACK_HOLE_FORMATION: return 3.0f;
        case EventType::FUSION_IGNITION:      return 4.0f;
        case EventType::THERMAL_EXPLOSION:    return 3.0f;
        case EventType::COMPOSITE_FORMED:     return 2.0f;
        case EventType::SPRING_CRITICAL_STRESS: return 0.5f;
        case EventType::SPRING_CREATED:       return 1.0f;
        case EventType::COMPOSITE_COLLISION:  return 1.5f;
        default:                             return config.event_fade_time;
    }
}

float ViewerEventBridge::getDistanceToCamera(float x, float y) const {
    if (!viewer) return 0.0f;
    
    const auto& camera = viewer->getCamera();
    float dx = x - camera.x;
    float dy = y - camera.y;
    return sqrt(dx*dx + dy*dy);
}

bool ViewerEventBridge::isInCameraView(float x, float y, float radius) const {
    if (!viewer) return false;
    
    // Convert to screen coordinates and check if within extended screen bounds
    int screen_x, screen_y;
    viewer->worldToScreen(x, y, screen_x, screen_y);
    
    // Get screen dimensions (would need to be accessible from viewer)
    // For now, assume reasonable screen bounds
    const int margin = 200;  // Pixel margin beyond screen
    return (screen_x >= -margin && screen_x <= 1920 + margin &&
            screen_y >= -margin && screen_y <= 1080 + margin);
}

// Factory implementations

std::unique_ptr<ViewerEventBridge> EventBridgeFactory::createPerformanceBridge() {
    auto bridge = std::make_unique<ViewerEventBridge>();
    
    ViewerEventBridge::Config config;
    config.max_event_distance = 500.0f;    // Shorter distance
    config.event_fade_time = 1.0f;         // Faster fade
    config.filter_by_camera = true;        // Aggressive filtering
    config.max_active_events = 100;        // Lower limit
    
    // Set filter to only show high-energy events
    bridge->setEventFilter(EventFilters::highEnergyOnly);
    
    return bridge;
}

std::unique_ptr<ViewerEventBridge> EventBridgeFactory::createDebugBridge() {
    auto bridge = std::make_unique<ViewerEventBridge>();
    
    ViewerEventBridge::Config config;
    config.max_event_distance = 10000.0f;  // Very large distance
    config.event_fade_time = 5.0f;         // Long fade time
    config.filter_by_camera = false;       // No filtering
    config.max_active_events = 2000;       // High limit
    
    // No event filter - show everything
    
    return bridge;
}

std::unique_ptr<ViewerEventBridge> EventBridgeFactory::createPresentationBridge() {
    auto bridge = std::make_unique<ViewerEventBridge>();
    
    ViewerEventBridge::Config config;
    config.max_event_distance = 1000.0f;
    config.event_fade_time = 3.0f;
    config.filter_by_camera = true;
    config.max_active_events = 500;
    
    // Set filter to only show visually impressive events
    bridge->setEventFilter(EventFilters::ofTypes({
        EventType::STAR_FORMATION,
        EventType::BLACK_HOLE_FORMATION,
        EventType::FUSION_IGNITION,
        EventType::THERMAL_EXPLOSION,
        EventType::COMPOSITE_FORMED,
        EventType::COMPOSITE_BROKEN,
        EventType::HARD_COLLISION
    }));
    
    return bridge;
}

std::unique_ptr<ViewerEventBridge> EventBridgeFactory::fromConfig(const std::string& config_json) {
    // TODO: Parse JSON configuration
    return createPerformanceBridge();
}

// Event filter implementations

bool EventFilters::highEnergyOnly(const Event& event) {
    return event.magnitude > 100.0f || 
           event.type == EventType::STAR_FORMATION ||
           event.type == EventType::BLACK_HOLE_FORMATION ||
           event.type == EventType::FUSION_IGNITION ||
           event.type == EventType::THERMAL_EXPLOSION;
}

std::function<bool(const Event&)> EventFilters::withinDistance(float x, float y, float max_distance) {
    return [x, y, max_distance](const Event& event) -> bool {
        float dx = event.x - x;
        float dy = event.y - y;
        return (dx*dx + dy*dy) <= (max_distance * max_distance);
    };
}

std::function<bool(const Event&)> EventFilters::ofTypes(std::initializer_list<EventType> types) {
    std::vector<EventType> allowed_types(types);
    return [allowed_types](const Event& event) -> bool {
        return std::find(allowed_types.begin(), allowed_types.end(), event.type) != allowed_types.end();
    };
}

std::function<bool(const Event&)> EventFilters::combine(std::vector<std::function<bool(const Event&)>> filters) {
    return [filters](const Event& event) -> bool {
        for (const auto& filter : filters) {
            if (!filter(event)) {
                return false;
            }
        }
        return true;
    };
}

} // namespace digistar
#pragma once

#include "../events/event_system.h"
#include "../events/event_consumer.h"
#include "graphics_viewer.h"
#include <memory>
#include <thread>
#include <atomic>
#include <functional>

namespace digistar {

/**
 * Bridge between the physics event system and graphics viewer
 * 
 * This class consumes physics events and translates them into visual
 * events for the graphics viewer. It runs in a separate thread to
 * avoid blocking the main rendering loop.
 */
class ViewerEventBridge {
public:
    /**
     * Configuration for event processing
     */
    struct Config {
        std::string shm_name = "digistar_events";  ///< Shared memory name for events
        std::string consumer_name = "graphics_viewer";  ///< Name of this consumer
        bool auto_start = true;                    ///< Start processing thread automatically
        float max_event_distance = 1000.0f;       ///< Max distance to show events
        float event_fade_time = 2.0f;             ///< How long events stay visible
        bool filter_by_camera = true;             ///< Only show events near camera
        size_t max_active_events = 1000;          ///< Limit active visual events
    };
    
private:
    std::unique_ptr<EventConsumer> event_consumer;
    std::shared_ptr<GraphicsViewer> viewer;
    Config config;
    
    // Threading
    std::unique_ptr<std::thread> processing_thread;
    std::atomic<bool> should_stop{false};
    std::atomic<bool> is_running{false};
    
    // Statistics
    struct Stats {
        std::atomic<uint64_t> events_processed{0};
        std::atomic<uint64_t> events_filtered{0};
        std::atomic<uint64_t> visual_events_created{0};
        std::atomic<float> processing_time_ms{0};
    } stats;
    
    // Event filters and mappings
    std::function<bool(const Event&)> event_filter;
    
public:
    ViewerEventBridge() = default;
    ~ViewerEventBridge();
    
    // === Lifecycle ===
    
    /**
     * Initialize the event bridge
     * @param viewer_ptr Shared pointer to graphics viewer
     * @param cfg Configuration
     * @return true on success
     */
    bool initialize(std::shared_ptr<GraphicsViewer> viewer_ptr, const Config& cfg = Config{});
    
    /**
     * Shutdown the event bridge
     */
    void shutdown();
    
    /**
     * Start processing events in separate thread
     * @return true on success
     */
    bool start();
    
    /**
     * Stop processing events
     */
    void stop();
    
    /**
     * Check if currently processing events
     */
    bool isRunning() const { return is_running.load(); }
    
    // === Manual Processing ===
    
    /**
     * Process events manually (single-threaded mode)
     * Call this regularly if not using automatic processing
     * @param max_events Maximum events to process per call
     * @return Number of events processed
     */
    size_t processEvents(size_t max_events = 100);
    
    // === Configuration ===
    
    /**
     * Set event filter function
     * @param filter Function that returns true for events to process
     */
    void setEventFilter(std::function<bool(const Event&)> filter) {
        event_filter = filter;
    }
    
    /**
     * Get configuration reference
     */
    Config& getConfig() { return config; }
    const Config& getConfig() const { return config; }
    
    // === Statistics ===
    
    /**
     * Get processing statistics
     */
    struct ProcessingStats {
        uint64_t events_processed;
        uint64_t events_filtered;
        uint64_t visual_events_created;
        float processing_time_ms;
        bool is_connected;
        std::string consumer_status;
    };
    
    ProcessingStats getStats() const;
    
    /**
     * Reset statistics counters
     */
    void resetStats();
    
private:
    // Main processing loop
    void processingLoop();
    
    // Event handlers - convert physics events to visual events
    void handleParticleEvent(const Event& event);
    void handleSpringEvent(const Event& event);
    void handleCollisionEvent(const Event& event);
    void handleThermalEvent(const Event& event);
    void handleCompositeEvent(const Event& event);
    void handleSystemEvent(const Event& event);
    
    // Helper methods
    bool shouldShowEvent(const Event& event) const;
    VisualEvent createVisualEvent(const Event& event) const;
    SDL_Color getEventColor(EventType type) const;
    float getEventIntensity(const Event& event) const;
    float getEventDuration(EventType type) const;
    
    // Distance checking for filtering
    float getDistanceToCamera(float x, float y) const;
    bool isInCameraView(float x, float y, float radius) const;
};

/**
 * Factory for creating pre-configured event bridges
 */
class EventBridgeFactory {
public:
    /**
     * Create bridge optimized for performance (minimal visual events)
     */
    static std::unique_ptr<ViewerEventBridge> createPerformanceBridge();
    
    /**
     * Create bridge for debugging (show all events)
     */
    static std::unique_ptr<ViewerEventBridge> createDebugBridge();
    
    /**
     * Create bridge for presentations (visually impressive events only)
     */
    static std::unique_ptr<ViewerEventBridge> createPresentationBridge();
    
    /**
     * Create bridge from configuration JSON
     */
    static std::unique_ptr<ViewerEventBridge> fromConfig(const std::string& config_json);
};

/**
 * Event filtering utilities
 */
namespace EventFilters {
    /**
     * Filter that only passes high-energy events
     */
    bool highEnergyOnly(const Event& event);
    
    /**
     * Filter that passes events within distance of point
     */
    std::function<bool(const Event&)> withinDistance(float x, float y, float max_distance);
    
    /**
     * Filter that passes events of specific types
     */
    std::function<bool(const Event&)> ofTypes(std::initializer_list<EventType> types);
    
    /**
     * Combine multiple filters with logical AND
     */
    std::function<bool(const Event&)> combine(std::vector<std::function<bool(const Event&)>> filters);
}

} // namespace digistar
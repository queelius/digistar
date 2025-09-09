#pragma once

#include "event_system.h"
#include <functional>
#include <vector>
#include <unordered_set>
#include <bitset>
#include <chrono>

namespace digistar {

/**
 * Event consumer for client applications
 * 
 * This class provides a safe, efficient interface for client applications
 * (renderer, audio, network, etc.) to consume events from the shared ring buffer.
 * It handles consumer registration, event filtering, and provides both polling
 * and callback-based consumption patterns.
 * 
 * Key features:
 * - Automatic consumer registration and cleanup
 * - Event type filtering for efficiency
 * - Spatial filtering for location-based events
 * - Batch processing for high-throughput scenarios
 * - Callback-based event handling
 * - Built-in statistics and health monitoring
 */
class EventConsumer {
public:
    /**
     * Event handler callback type
     */
    using EventHandler = std::function<void(const Event&)>;
    
    /**
     * Constructor - automatically registers this consumer with the buffer
     * 
     * @param buffer Pointer to the shared event ring buffer
     * @param consumer_name Human-readable name for this consumer
     */
    EventConsumer(EventRingBuffer* buffer, const std::string& consumer_name);
    
    /**
     * Destructor - automatically unregisters this consumer
     */
    ~EventConsumer();
    
    // Non-copyable but moveable
    EventConsumer(const EventConsumer&) = delete;
    EventConsumer& operator=(const EventConsumer&) = delete;
    EventConsumer(EventConsumer&&) = default;
    EventConsumer& operator=(EventConsumer&&) = default;
    
    /**
     * Polling interface for synchronous event consumption
     */
    
    /**
     * Poll for the next available event
     * 
     * @param event Reference to store the event if available
     * @return true if an event was retrieved, false if no events available
     */
    bool poll(Event& event);
    
    /**
     * Poll for multiple events (more efficient than individual polls)
     * 
     * @param events Array to store retrieved events
     * @param max_events Maximum number of events to retrieve
     * @return Number of events actually retrieved (0 if none available)
     */
    size_t poll_batch(Event* events, size_t max_events);
    
    /**
     * Check how many events are available to read
     * 
     * @return Number of unread events
     */
    size_t available_events() const;
    
    /**
     * Skip the next N events (useful for catching up when falling behind)
     * 
     * @param count Number of events to skip
     * @return Number of events actually skipped
     */
    size_t skip_events(size_t count);
    
    /**
     * Event filtering interface
     */
    
    /**
     * Enable filtering for specific event types
     * Only events of these types will be returned by poll operations.
     * 
     * @param types Set of event types to include
     */
    void set_event_type_filter(const std::unordered_set<EventType>& types);
    
    /**
     * Add a single event type to the filter
     * 
     * @param type Event type to include
     */
    void add_event_type_filter(EventType type);
    
    /**
     * Remove a single event type from the filter  
     * 
     * @param type Event type to exclude
     */
    void remove_event_type_filter(EventType type);
    
    /**
     * Clear all event type filters (accept all event types)
     */
    void clear_event_type_filter();
    
    /**
     * Set spatial filter - only events within the specified area will be returned
     * 
     * @param center_x Center X coordinate
     * @param center_y Center Y coordinate
     * @param radius Radius of area of interest
     */
    void set_spatial_filter(float center_x, float center_y, float radius);
    
    /**
     * Clear spatial filter (accept events from all locations)
     */
    void clear_spatial_filter();
    
    /**
     * Callback-based interface for asynchronous event handling
     */
    
    /**
     * Set a callback to be invoked for each event
     * This enables push-based event consumption.
     * 
     * @param handler Function to call for each event
     */
    void set_event_handler(EventHandler handler);
    
    /**
     * Process all available events using the registered handler
     * Call this regularly to process events in callback mode.
     * 
     * @param max_events Maximum number of events to process in this call (0 = unlimited)
     * @return Number of events processed
     */
    size_t process_events(size_t max_events = 0);
    
    /**
     * Specialized polling methods for common use cases
     */
    
    /**
     * Poll for events matching a specific type
     * 
     * @param event_type Type of event to look for
     * @param event Reference to store the event if found
     * @return true if a matching event was found
     */
    bool poll_for_type(EventType event_type, Event& event);
    
    /**
     * Poll for events within a spatial area
     * 
     * @param center_x Center X coordinate
     * @param center_y Center Y coordinate  
     * @param radius Search radius
     * @param events Array to store found events
     * @param max_events Maximum number of events to return
     * @return Number of events found
     */
    size_t poll_in_area(float center_x, float center_y, float radius,
                        Event* events, size_t max_events);
    
    /**
     * Statistics and monitoring
     */
    
    /**
     * Consumer statistics
     */
    struct ConsumerStats {
        uint64_t events_consumed;           ///< Total events consumed
        uint64_t events_filtered_type;      ///< Events filtered out by type
        uint64_t events_filtered_spatial;   ///< Events filtered out by location
        uint64_t poll_operations;           ///< Number of poll() calls
        uint64_t batch_operations;          ///< Number of poll_batch() calls
        double average_poll_time_ns;        ///< Average time per poll operation
        double total_processing_time_ms;    ///< Total time spent processing events
        uint64_t events_skipped;            ///< Number of events intentionally skipped
    };
    
    /**
     * Get consumer statistics
     */
    ConsumerStats get_stats() const { return stats_; }
    
    /**
     * Reset statistics counters
     */
    void reset_stats();
    
    /**
     * Get consumer information
     */
    int get_consumer_id() const { return consumer_id_; }
    const std::string& get_consumer_name() const { return consumer_name_; }
    
    /**
     * Check if this consumer is falling behind
     * Returns true if the consumer is significantly behind other consumers.
     */
    bool is_falling_behind() const;
    
    /**
     * Get buffer health from consumer's perspective
     */
    bool is_healthy() const;

private:
    EventRingBuffer* buffer_;              ///< Pointer to shared ring buffer
    int consumer_id_;                      ///< Assigned consumer ID
    std::string consumer_name_;            ///< Human-readable consumer name
    uint64_t local_read_pos_;              ///< Local read position (for efficiency)
    
    // Filtering
    std::bitset<65536> event_type_filter_; ///< Fast lookup for event type filtering
    bool use_type_filter_;                 ///< Whether type filtering is enabled
    bool use_spatial_filter_;              ///< Whether spatial filtering is enabled
    float spatial_center_x_, spatial_center_y_, spatial_radius_; ///< Spatial filter parameters
    
    // Callback handling
    EventHandler event_handler_;           ///< Optional event handler callback
    
    // Statistics
    ConsumerStats stats_;
    std::chrono::high_resolution_clock::time_point last_timing_;
    
    /**
     * Register this consumer with the buffer
     */
    void register_consumer();
    
    /**
     * Unregister this consumer from the buffer
     */
    void unregister_consumer();
    
    /**
     * Check if an event passes the current filters
     */
    bool passes_filters(const Event& event) const;
    
    /**
     * Update timing statistics
     */
    void update_timing(const std::chrono::high_resolution_clock::time_point& start);
    
    /**
     * Internal poll implementation with filtering
     */
    bool poll_internal(Event& event);
};

/**
 * Specialized event consumers for common use cases
 */

/**
 * Audio event consumer
 * 
 * Specialized consumer for audio systems, with built-in filtering for
 * audio-relevant events and spatial audio support.
 */
class AudioEventConsumer : public EventConsumer {
public:
    /**
     * Constructor
     * 
     * @param buffer Pointer to shared event ring buffer
     * @param listener_x Initial listener X position
     * @param listener_y Initial listener Y position
     * @param max_audio_distance Maximum distance for audio events
     */
    AudioEventConsumer(EventRingBuffer* buffer, 
                      float listener_x = 0.0f, float listener_y = 0.0f,
                      float max_audio_distance = 1000.0f);
    
    /**
     * Update listener position for spatial audio
     */
    void set_listener_position(float x, float y);
    
    /**
     * Set maximum distance for audio events
     */
    void set_max_audio_distance(float distance);
    
private:
    float listener_x_, listener_y_;
    float max_audio_distance_;
    
    void setup_audio_filters();
};

/**
 * Network event consumer
 * 
 * Specialized consumer for network systems, with filtering for
 * network-relevant events and automatic serialization support.
 */
class NetworkEventConsumer : public EventConsumer {
public:
    /**
     * Constructor
     * 
     * @param buffer Pointer to shared event ring buffer
     */
    explicit NetworkEventConsumer(EventRingBuffer* buffer);
    
    /**
     * Poll for network-relevant events only
     * These are events marked with FLAG_NETWORK_BROADCAST
     */
    size_t poll_network_events(Event* events, size_t max_events);
    
private:
    void setup_network_filters();
};

/**
 * Visual effects event consumer
 * 
 * Specialized consumer for visual effects systems, with filtering for
 * visually interesting events and camera-relative spatial filtering.
 */
class VFXEventConsumer : public EventConsumer {
public:
    /**
     * Constructor
     * 
     * @param buffer Pointer to shared event ring buffer
     * @param camera_x Initial camera X position
     * @param camera_y Initial camera Y position
     * @param view_distance Maximum view distance for effects
     */
    VFXEventConsumer(EventRingBuffer* buffer,
                     float camera_x = 0.0f, float camera_y = 0.0f,
                     float view_distance = 2000.0f);
    
    /**
     * Update camera position for view culling
     */
    void set_camera_position(float x, float y);
    
    /**
     * Set maximum view distance
     */
    void set_view_distance(float distance);
    
private:
    float camera_x_, camera_y_;
    float view_distance_;
    
    void setup_vfx_filters();
};

} // namespace digistar
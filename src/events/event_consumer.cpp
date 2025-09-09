#include "event_consumer.h"
#include <algorithm>
#include <cmath>
#include <unistd.h>
#include <cstring>
#include <stdexcept>

namespace digistar {

EventConsumer::EventConsumer(EventRingBuffer* buffer, const std::string& consumer_name)
    : buffer_(buffer), consumer_id_(-1), consumer_name_(consumer_name), local_read_pos_(0),
      use_type_filter_(false), use_spatial_filter_(false),
      spatial_center_x_(0.0f), spatial_center_y_(0.0f), spatial_radius_(0.0f),
      stats_{} {
    
    if (!buffer_ || !buffer_->is_valid()) {
        throw std::invalid_argument("Invalid event ring buffer");
    }
    
    register_consumer();
    reset_stats();
}

EventConsumer::~EventConsumer() {
    unregister_consumer();
}

void EventConsumer::register_consumer() {
    uint32_t current_consumers = buffer_->num_consumers.load(std::memory_order_acquire);
    
    if (current_consumers >= EventRingBuffer::MAX_CONSUMERS) {
        throw std::runtime_error("Too many consumers registered");
    }
    
    // Atomic increment and get new consumer ID
    consumer_id_ = static_cast<int>(buffer_->num_consumers.fetch_add(1, std::memory_order_acq_rel));
    
    if (consumer_id_ >= static_cast<int>(EventRingBuffer::MAX_CONSUMERS)) {
        buffer_->num_consumers.fetch_sub(1, std::memory_order_acq_rel);
        throw std::runtime_error("Consumer registration race condition");
    }
    
    // Initialize consumer info
    Consumer& info = buffer_->consumers[consumer_id_];
    info.pid = getpid();
    strncpy(info.name, consumer_name_.c_str(), sizeof(info.name) - 1);
    info.name[sizeof(info.name) - 1] = '\0';
    info.events_read.store(0, std::memory_order_relaxed);
    info.last_read_time.store(
        std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now().time_since_epoch()).count(),
        std::memory_order_relaxed);
    
    // Start reading from current write position (don't read historical events)
    local_read_pos_ = buffer_->write_pos.load(std::memory_order_acquire);
    buffer_->read_pos[consumer_id_].store(local_read_pos_, std::memory_order_release);
}

void EventConsumer::unregister_consumer() {
    if (consumer_id_ >= 0 && consumer_id_ < static_cast<int>(EventRingBuffer::MAX_CONSUMERS)) {
        // Clear consumer info
        Consumer& info = buffer_->consumers[consumer_id_];
        info.pid = 0;
        info.name[0] = '\0';
        
        // Note: We don't decrement num_consumers here as it could create race conditions
        // The producer will handle cleanup of inactive consumers
        consumer_id_ = -1;
    }
}

bool EventConsumer::poll(Event& event) {
    auto start = std::chrono::high_resolution_clock::now();
    bool result = poll_internal(event);
    update_timing(start);
    
    stats_.poll_operations++;
    if (result) {
        stats_.events_consumed++;
    }
    
    return result;
}

bool EventConsumer::poll_internal(Event& event) {
    uint64_t read = local_read_pos_;
    uint64_t write = buffer_->write_pos.load(std::memory_order_acquire);
    
    // Search for next event that passes filters
    while (read < write) {
        uint64_t slot = read & (buffer_->capacity - 1);
        const Event& candidate = buffer_->events[slot];
        
        local_read_pos_ = read + 1;
        read++;
        
        if (passes_filters(candidate)) {
            event = candidate;
            
            // Update shared read position
            buffer_->read_pos[consumer_id_].store(local_read_pos_, std::memory_order_release);
            buffer_->consumers[consumer_id_].events_read.fetch_add(1, std::memory_order_relaxed);
            buffer_->consumers[consumer_id_].last_read_time.store(
                std::chrono::duration_cast<std::chrono::milliseconds>(
                    std::chrono::steady_clock::now().time_since_epoch()).count(),
                std::memory_order_relaxed);
            
            return true;
        }
    }
    
    // No events available or none passed filters
    buffer_->read_pos[consumer_id_].store(local_read_pos_, std::memory_order_release);
    return false;
}

size_t EventConsumer::poll_batch(Event* events, size_t max_events) {
    if (!events || max_events == 0) {
        return 0;
    }
    
    auto start = std::chrono::high_resolution_clock::now();
    
    uint64_t read = local_read_pos_;
    uint64_t write = buffer_->write_pos.load(std::memory_order_acquire);
    size_t events_found = 0;
    
    // Process events while we have space and events available
    while (read < write && events_found < max_events) {
        uint64_t slot = read & (buffer_->capacity - 1);
        const Event& candidate = buffer_->events[slot];
        
        read++;
        
        if (passes_filters(candidate)) {
            events[events_found++] = candidate;
        }
    }
    
    // Update positions
    local_read_pos_ = read;
    buffer_->read_pos[consumer_id_].store(local_read_pos_, std::memory_order_release);
    
    if (events_found > 0) {
        buffer_->consumers[consumer_id_].events_read.fetch_add(events_found, std::memory_order_relaxed);
        buffer_->consumers[consumer_id_].last_read_time.store(
            std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::steady_clock::now().time_since_epoch()).count(),
            std::memory_order_relaxed);
    }
    
    // Update statistics
    stats_.batch_operations++;
    stats_.events_consumed += events_found;
    update_timing(start);
    
    return events_found;
}

size_t EventConsumer::available_events() const {
    uint64_t write = buffer_->write_pos.load(std::memory_order_acquire);
    return static_cast<size_t>(write - local_read_pos_);
}

size_t EventConsumer::skip_events(size_t count) {
    uint64_t write = buffer_->write_pos.load(std::memory_order_acquire);
    uint64_t available = write - local_read_pos_;
    size_t to_skip = std::min(count, static_cast<size_t>(available));
    
    local_read_pos_ += to_skip;
    buffer_->read_pos[consumer_id_].store(local_read_pos_, std::memory_order_release);
    
    stats_.events_skipped += to_skip;
    return to_skip;
}

void EventConsumer::set_event_type_filter(const std::unordered_set<EventType>& types) {
    event_type_filter_.reset();
    for (EventType type : types) {
        event_type_filter_.set(static_cast<size_t>(type));
    }
    use_type_filter_ = !types.empty();
}

void EventConsumer::add_event_type_filter(EventType type) {
    event_type_filter_.set(static_cast<size_t>(type));
    use_type_filter_ = true;
}

void EventConsumer::remove_event_type_filter(EventType type) {
    event_type_filter_.reset(static_cast<size_t>(type));
    use_type_filter_ = event_type_filter_.any();
}

void EventConsumer::clear_event_type_filter() {
    event_type_filter_.reset();
    use_type_filter_ = false;
}

void EventConsumer::set_spatial_filter(float center_x, float center_y, float radius) {
    spatial_center_x_ = center_x;
    spatial_center_y_ = center_y;
    spatial_radius_ = radius;
    use_spatial_filter_ = true;
}

void EventConsumer::clear_spatial_filter() {
    use_spatial_filter_ = false;
}

void EventConsumer::set_event_handler(EventHandler handler) {
    event_handler_ = std::move(handler);
}

size_t EventConsumer::process_events(size_t max_events) {
    if (!event_handler_) {
        return 0;
    }
    
    auto start = std::chrono::high_resolution_clock::now();
    
    const size_t BATCH_SIZE = 100;
    Event events[BATCH_SIZE];
    size_t total_processed = 0;
    
    while (max_events == 0 || total_processed < max_events) {
        size_t batch_limit = (max_events == 0) ? BATCH_SIZE : 
                           std::min(BATCH_SIZE, max_events - total_processed);
        
        size_t batch_count = poll_batch(events, batch_limit);
        if (batch_count == 0) {
            break;  // No more events
        }
        
        // Process events through callback
        for (size_t i = 0; i < batch_count; i++) {
            event_handler_(events[i]);
        }
        
        total_processed += batch_count;
    }
    
    update_timing(start);
    return total_processed;
}

bool EventConsumer::poll_for_type(EventType event_type, Event& event) {
    // Temporarily set type filter
    bool old_use_filter = use_type_filter_;
    auto old_filter = event_type_filter_;
    
    event_type_filter_.reset();
    event_type_filter_.set(static_cast<size_t>(event_type));
    use_type_filter_ = true;
    
    bool result = poll(event);
    
    // Restore old filter
    use_type_filter_ = old_use_filter;
    event_type_filter_ = old_filter;
    
    return result;
}

size_t EventConsumer::poll_in_area(float center_x, float center_y, float radius,
                                  Event* events, size_t max_events) {
    // Temporarily set spatial filter
    bool old_use_spatial = use_spatial_filter_;
    float old_x = spatial_center_x_;
    float old_y = spatial_center_y_;
    float old_radius = spatial_radius_;
    
    set_spatial_filter(center_x, center_y, radius);
    
    size_t result = poll_batch(events, max_events);
    
    // Restore old filter
    if (old_use_spatial) {
        set_spatial_filter(old_x, old_y, old_radius);
    } else {
        clear_spatial_filter();
    }
    
    return result;
}

bool EventConsumer::passes_filters(const Event& event) const {
    // Type filter
    if (use_type_filter_) {
        if (!event_type_filter_.test(static_cast<size_t>(event.type))) {
            const_cast<EventConsumer*>(this)->stats_.events_filtered_type++;
            return false;
        }
    }
    
    // Spatial filter
    if (use_spatial_filter_) {
        float dx = event.x - spatial_center_x_;
        float dy = event.y - spatial_center_y_;
        float distance_sq = dx * dx + dy * dy;
        float radius_sq = spatial_radius_ * spatial_radius_;
        
        if (distance_sq > radius_sq) {
            const_cast<EventConsumer*>(this)->stats_.events_filtered_spatial++;
            return false;
        }
    }
    
    return true;
}

void EventConsumer::update_timing(const std::chrono::high_resolution_clock::time_point& start) {
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
    
    // Update running average
    double new_time = static_cast<double>(duration.count());
    stats_.average_poll_time_ns = stats_.average_poll_time_ns * 0.9 + new_time * 0.1;
    stats_.total_processing_time_ms += new_time / 1e6;  // Convert to milliseconds
}

void EventConsumer::reset_stats() {
    stats_ = {};
    last_timing_ = std::chrono::high_resolution_clock::now();
}

bool EventConsumer::is_falling_behind() const {
    if (!buffer_->is_valid()) {
        return true;
    }
    
    // Check if we're significantly behind the write position
    uint64_t write = buffer_->write_pos.load(std::memory_order_acquire);
    uint64_t behind = write - local_read_pos_;
    
    // Consider falling behind if more than 25% of buffer capacity behind
    return behind > (buffer_->capacity / 4);
}

bool EventConsumer::is_healthy() const {
    return buffer_ && buffer_->is_valid() && !is_falling_behind();
}

// Specialized consumer implementations

AudioEventConsumer::AudioEventConsumer(EventRingBuffer* buffer, 
                                      float listener_x, float listener_y,
                                      float max_audio_distance)
    : EventConsumer(buffer, "AudioConsumer"),
      listener_x_(listener_x), listener_y_(listener_y),
      max_audio_distance_(max_audio_distance) {
    
    setup_audio_filters();
    set_spatial_filter(listener_x, listener_y, max_audio_distance);
}

void AudioEventConsumer::set_listener_position(float x, float y) {
    listener_x_ = x;
    listener_y_ = y;
    set_spatial_filter(x, y, max_audio_distance_);
}

void AudioEventConsumer::set_max_audio_distance(float distance) {
    max_audio_distance_ = distance;
    set_spatial_filter(listener_x_, listener_y_, distance);
}

void AudioEventConsumer::setup_audio_filters() {
    // Filter for audio-relevant events
    std::unordered_set<EventType> audio_types = {
        EventType::SPRING_BROKEN,
        EventType::HARD_COLLISION,
        EventType::SOFT_CONTACT,
        EventType::BLACK_HOLE_FORMATION,
        EventType::STAR_FORMATION,
        EventType::THERMAL_EXPLOSION,
        EventType::STRUCTURAL_FAILURE,
        EventType::TIDAL_DISRUPTION,
        EventType::RESONANCE_DETECTED
    };
    set_event_type_filter(audio_types);
}

NetworkEventConsumer::NetworkEventConsumer(EventRingBuffer* buffer)
    : EventConsumer(buffer, "NetworkConsumer") {
    
    setup_network_filters();
}

size_t NetworkEventConsumer::poll_network_events(Event* events, size_t max_events) {
    if (!events || max_events == 0) {
        return 0;
    }
    
    size_t events_found = 0;
    Event event;
    
    while (events_found < max_events && poll(event)) {
        // Only include events marked for network broadcast
        if (event.flags & FLAG_NETWORK_BROADCAST) {
            events[events_found++] = event;
        }
    }
    
    return events_found;
}

void NetworkEventConsumer::setup_network_filters() {
    // Filter for network-relevant events
    std::unordered_set<EventType> network_types = {
        EventType::PARTICLE_MERGE,
        EventType::BLACK_HOLE_FORMATION,
        EventType::STAR_FORMATION,
        EventType::COMPOSITE_FORMED,
        EventType::COMPOSITE_BROKEN,
        EventType::PLAYER_ACTION,
        EventType::PLAYER_JOIN,
        EventType::PLAYER_LEAVE,
        EventType::TICK_COMPLETE
    };
    set_event_type_filter(network_types);
}

VFXEventConsumer::VFXEventConsumer(EventRingBuffer* buffer,
                                   float camera_x, float camera_y,
                                   float view_distance)
    : EventConsumer(buffer, "VFXConsumer"),
      camera_x_(camera_x), camera_y_(camera_y),
      view_distance_(view_distance) {
    
    setup_vfx_filters();
    set_spatial_filter(camera_x, camera_y, view_distance);
}

void VFXEventConsumer::set_camera_position(float x, float y) {
    camera_x_ = x;
    camera_y_ = y;
    set_spatial_filter(x, y, view_distance_);
}

void VFXEventConsumer::set_view_distance(float distance) {
    view_distance_ = distance;
    set_spatial_filter(camera_x_, camera_y_, distance);
}

void VFXEventConsumer::setup_vfx_filters() {
    // Filter for visually interesting events
    std::unordered_set<EventType> vfx_types = {
        EventType::PARTICLE_MERGE,
        EventType::PARTICLE_FISSION,
        EventType::BLACK_HOLE_FORMATION,
        EventType::BLACK_HOLE_ABSORPTION,
        EventType::STAR_FORMATION,
        EventType::HARD_COLLISION,
        EventType::THERMAL_EXPLOSION,
        EventType::FUSION_IGNITION,
        EventType::STRUCTURAL_FAILURE,
        EventType::TIDAL_DISRUPTION,
        EventType::COMPOSITE_FORMED,
        EventType::COMPOSITE_BROKEN
    };
    set_event_type_filter(vfx_types);
}

} // namespace digistar
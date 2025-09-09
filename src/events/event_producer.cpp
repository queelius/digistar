#include "event_producer.h"
#include <algorithm>
#include <cstring>
#include <stdexcept>

namespace digistar {

EventProducer::EventProducer(EventRingBuffer* buffer)
    : buffer_(buffer), current_tick_(0), current_timestamp_(0.0f), stats_{} {
    
    if (!buffer_ || !buffer_->is_valid()) {
        throw std::invalid_argument("Invalid event ring buffer");
    }
    
    reset_stats();
}

void EventProducer::set_simulation_state(uint32_t tick, float timestamp) {
    current_tick_ = tick;
    current_timestamp_ = timestamp;
}

EventSystemError EventProducer::emit(const Event& event) {
    auto start = std::chrono::high_resolution_clock::now();
    auto result = emit_internal(event);
    update_timing(start);
    
    if (result == EventSystemError::SUCCESS) {
        stats_.events_emitted++;
    } else if (result == EventSystemError::BUFFER_FULL) {
        stats_.events_dropped++;
    }
    
    return result;
}

size_t EventProducer::emit_batch(const Event* events, size_t count) {
    if (!events || count == 0) {
        return 0;
    }
    
    auto start = std::chrono::high_resolution_clock::now();
    
    // Get current positions
    uint64_t write = buffer_->write_pos.load(std::memory_order_relaxed);
    uint64_t min_read = get_min_consumer_pos_cached();
    
    // Calculate available space
    size_t available = buffer_->capacity - (write - min_read);
    size_t to_write = std::min(count, available);
    
    // Write events to buffer
    for (size_t i = 0; i < to_write; i++) {
        uint64_t slot = (write + i) & (buffer_->capacity - 1);
        buffer_->events[slot] = events[i];
    }
    
    // Atomically update write position
    buffer_->write_pos.store(write + to_write, std::memory_order_release);
    buffer_->total_events.fetch_add(to_write, std::memory_order_relaxed);
    
    // Track dropped events
    if (to_write < count) {
        size_t dropped = count - to_write;
        buffer_->dropped_events.fetch_add(dropped, std::memory_order_relaxed);
        stats_.events_dropped += dropped;
    }
    
    // Update statistics
    stats_.events_emitted += to_write;
    stats_.batch_operations++;
    update_timing(start);
    
    return to_write;
}

bool EventProducer::try_emit(const Event& event) {
    return emit_internal(event) == EventSystemError::SUCCESS;
}

EventSystemError EventProducer::emit_internal(const Event& event) {
    // Get minimum consumer position
    uint64_t min_read = get_min_consumer_pos_cached();
    uint64_t write = buffer_->write_pos.load(std::memory_order_relaxed);
    
    // Check if buffer is full
    if (write - min_read >= buffer_->capacity) {
        buffer_->dropped_events.fetch_add(1, std::memory_order_relaxed);
        return EventSystemError::BUFFER_FULL;
    }
    
    // Write event to buffer
    uint64_t slot = write & (buffer_->capacity - 1);
    buffer_->events[slot] = event;
    
    // Atomically publish the event
    buffer_->write_pos.store(write + 1, std::memory_order_release);
    buffer_->total_events.fetch_add(1, std::memory_order_relaxed);
    
    return EventSystemError::SUCCESS;
}

uint64_t EventProducer::get_min_consumer_pos_cached() const {
    return buffer_->get_min_consumer_pos();
}

void EventProducer::update_timing(const std::chrono::high_resolution_clock::time_point& start) {
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
    
    // Update running average (simple exponential moving average)
    double new_time = static_cast<double>(duration.count());
    stats_.average_emit_time_ns = stats_.average_emit_time_ns * 0.9 + new_time * 0.1;
    stats_.total_emit_time_ms += new_time / 1e6;  // Convert to milliseconds
}

EventSystemError EventProducer::emit_spring_break(uint32_t spring_id, uint32_t particle_a, uint32_t particle_b,
                                                 float pos_x, float pos_y, float stress, float break_force) {
    Event event{};
    event.type = EventType::SPRING_BROKEN;
    event.flags = FLAG_NETWORK_BROADCAST | FLAG_SPATIAL_FILTER;
    event.tick = current_tick_;
    event.timestamp = current_timestamp_;
    event.x = pos_x;
    event.y = pos_y;
    event.radius = 0.0f;  // Point event
    event.primary_id = particle_a;
    event.secondary_id = particle_b;
    event.magnitude = stress;
    event.secondary_value = break_force;
    event.data.spring.stress = stress;
    event.data.spring.break_force = break_force;
    
    return emit(event);
}

EventSystemError EventProducer::emit_collision(uint32_t particle_a, uint32_t particle_b,
                                             float pos_x, float pos_y, float impulse, float penetration,
                                             float velocity_x, float velocity_y, bool is_hard) {
    Event event{};
    event.type = is_hard ? EventType::HARD_COLLISION : EventType::SOFT_CONTACT;
    event.flags = FLAG_SPATIAL_FILTER;
    event.tick = current_tick_;
    event.timestamp = current_timestamp_;
    event.x = pos_x;
    event.y = pos_y;
    event.radius = 0.0f;  // Point event
    event.primary_id = particle_a;
    event.secondary_id = particle_b;
    event.magnitude = impulse;
    event.secondary_value = penetration;
    event.data.collision.penetration = penetration;
    event.data.collision.impulse = impulse;
    event.data.collision.velocity_x = velocity_x;
    event.data.collision.velocity_y = velocity_y;
    
    return emit(event);
}

EventSystemError EventProducer::emit_particle_merge(uint32_t surviving_id, uint32_t absorbed_id,
                                                   float pos_x, float pos_y, 
                                                   float mass_before, float mass_after) {
    Event event{};
    event.type = EventType::PARTICLE_MERGE;
    event.flags = FLAG_NETWORK_BROADCAST | FLAG_HIGH_PRIORITY;
    event.tick = current_tick_;
    event.timestamp = current_timestamp_;
    event.x = pos_x;
    event.y = pos_y;
    event.radius = 0.0f;
    event.primary_id = surviving_id;
    event.secondary_id = absorbed_id;
    event.magnitude = mass_after;
    event.secondary_value = mass_before - mass_after;  // Mass difference
    event.data.merge.mass_before = mass_before;
    event.data.merge.mass_after = mass_after;
    
    return emit(event);
}

EventSystemError EventProducer::emit_black_hole_formation(uint32_t particle_id, float pos_x, float pos_y,
                                                         float mass, float event_horizon) {
    Event event{};
    event.type = EventType::BLACK_HOLE_FORMATION;
    event.flags = FLAG_HIGH_PRIORITY | FLAG_NETWORK_BROADCAST;
    event.tick = current_tick_;
    event.timestamp = current_timestamp_;
    event.x = pos_x;
    event.y = pos_y;
    event.radius = event_horizon;
    event.primary_id = particle_id;
    event.secondary_id = 0xFFFFFFFF;  // No secondary participant
    event.magnitude = mass;
    event.secondary_value = event_horizon;
    
    return emit(event);
}

EventSystemError EventProducer::emit_composite_formation(uint32_t composite_id, 
                                                       float center_x, float center_y, float radius,
                                                       uint32_t particle_count, float integrity) {
    Event event{};
    event.type = EventType::COMPOSITE_FORMED;
    event.flags = FLAG_NETWORK_BROADCAST | FLAG_SPATIAL_FILTER;
    event.tick = current_tick_;
    event.timestamp = current_timestamp_;
    event.x = center_x;
    event.y = center_y;
    event.radius = radius;
    event.primary_id = composite_id;
    event.secondary_id = 0xFFFFFFFF;
    event.magnitude = static_cast<float>(particle_count);
    event.secondary_value = integrity;
    event.data.composite.particle_count = particle_count;
    event.data.composite.integrity = integrity;
    
    return emit(event);
}

EventSystemError EventProducer::emit_resonance_detected(uint32_t composite_id,
                                                       float center_x, float center_y, float radius,
                                                       float frequency, float amplitude, float integrity) {
    Event event{};
    event.type = EventType::RESONANCE_DETECTED;
    event.flags = FLAG_SPATIAL_FILTER;
    event.tick = current_tick_;
    event.timestamp = current_timestamp_;
    event.x = center_x;
    event.y = center_y;
    event.radius = radius;
    event.primary_id = composite_id;
    event.secondary_id = 0xFFFFFFFF;
    event.magnitude = frequency;
    event.secondary_value = amplitude;
    event.data.resonance.frequency = frequency;
    event.data.resonance.amplitude = amplitude;
    
    return emit(event);
}

EventSystemError EventProducer::emit_thermal_event(EventType type, uint32_t particle_id,
                                                   float pos_x, float pos_y,
                                                   float temp_before, float temp_after, float pressure) {
    Event event{};
    event.type = type;
    event.flags = FLAG_SPATIAL_FILTER;
    event.tick = current_tick_;
    event.timestamp = current_timestamp_;
    event.x = pos_x;
    event.y = pos_y;
    event.radius = 0.0f;
    event.primary_id = particle_id;
    event.secondary_id = 0xFFFFFFFF;
    event.magnitude = temp_after;
    event.secondary_value = temp_after - temp_before;  // Temperature change
    event.data.thermal.temp_before = temp_before;
    event.data.thermal.temp_after = temp_after;
    event.data.thermal.pressure = pressure;
    
    return emit(event);
}

EventSystemError EventProducer::emit_system_event(EventType type, float magnitude) {
    Event event{};
    event.type = type;
    event.flags = FLAG_NONE;
    event.tick = current_tick_;
    event.timestamp = current_timestamp_;
    event.x = 0.0f;
    event.y = 0.0f;
    event.radius = 0.0f;
    event.primary_id = 0xFFFFFFFF;
    event.secondary_id = 0xFFFFFFFF;
    event.magnitude = magnitude;
    event.secondary_value = 0.0f;
    
    return emit(event);
}

void EventProducer::reset_stats() {
    stats_ = {};
    last_timing_ = std::chrono::high_resolution_clock::now();
}

bool EventProducer::is_healthy() const {
    if (!buffer_ || !buffer_->is_valid()) {
        return false;
    }
    
    // Check if buffer utilization is reasonable (not completely full)
    float utilization = buffer_->get_utilization();
    return utilization < 0.95f;  // Consider unhealthy if >95% full
}

// PhysicsEventEmitter implementation

PhysicsEventEmitter::PhysicsEventEmitter(std::shared_ptr<EventProducer> producer)
    : producer_(std::move(producer)), current_tick_(0), current_timestamp_(0.0f) {
    
    if (!producer_) {
        throw std::invalid_argument("Producer cannot be null");
    }
}

void PhysicsEventEmitter::set_physics_state(uint32_t tick, float timestamp) {
    current_tick_ = tick;
    current_timestamp_ = timestamp;
    producer_->set_simulation_state(tick, timestamp);
}

void PhysicsEventEmitter::emit_tick_complete(const SimulationStats& stats) {
    // Emit system event for tick completion
    producer_->emit_system_event(EventType::TICK_COMPLETE, stats.update_time_ms);
    
    // Could emit additional performance-related events here
    // For example, if performance is degrading significantly
}

} // namespace digistar
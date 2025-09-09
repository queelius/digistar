#pragma once

#include "event_system.h"
#include "../backend/backend_interface.h"
#include <vector>
#include <chrono>

namespace digistar {

/**
 * Event producer for the physics engine
 * 
 * This class provides a high-performance interface for the physics engine to emit
 * events to the shared ring buffer. It's designed to minimize overhead in the
 * critical physics simulation loop while providing comprehensive event generation
 * capabilities.
 * 
 * Key features:
 * - Lock-free event emission
 * - Batch event processing for efficiency  
 * - Automatic event dropping when consumers can't keep up
 * - Built-in timing and statistics
 * - Helper methods for common physics events
 */
class EventProducer {
public:
    /**
     * Constructor
     * 
     * @param buffer Pointer to the shared event ring buffer
     */
    explicit EventProducer(EventRingBuffer* buffer);
    
    /**
     * Destructor
     */
    ~EventProducer() = default;
    
    // Non-copyable but moveable
    EventProducer(const EventProducer&) = delete;
    EventProducer& operator=(const EventProducer&) = delete;
    EventProducer(EventProducer&&) = default;
    EventProducer& operator=(EventProducer&&) = default;
    
    /**
     * Set current simulation state for automatic event metadata
     * 
     * @param tick Current simulation tick
     * @param timestamp Current simulation time in seconds
     */
    void set_simulation_state(uint32_t tick, float timestamp);
    
    /**
     * Emit a single event
     * 
     * @param event The event to emit
     * @return EventSystemError::SUCCESS if emitted, ERROR_BUFFER_FULL if dropped
     */
    EventSystemError emit(const Event& event);
    
    /**
     * Emit multiple events in a batch (more efficient than individual emits)
     * 
     * @param events Array of events to emit
     * @param count Number of events in the array
     * @return Number of events successfully emitted (may be less than count if buffer is full)
     */
    size_t emit_batch(const Event* events, size_t count);
    
    /**
     * Try to emit an event (non-blocking, returns immediately if buffer is full)
     * 
     * @param event The event to emit
     * @return true if emitted, false if buffer is full
     */
    bool try_emit(const Event& event);
    
    /**
     * Helper methods for common physics events
     * These automatically fill in common fields like timestamp, tick, etc.
     */
    
    /**
     * Emit a spring break event
     * 
     * @param spring_id ID of the broken spring
     * @param particle_a ID of first particle
     * @param particle_b ID of second particle
     * @param pos_x Position X coordinate
     * @param pos_y Position Y coordinate  
     * @param stress Final stress that broke the spring
     * @param break_force Spring's break force threshold
     */
    EventSystemError emit_spring_break(uint32_t spring_id, uint32_t particle_a, uint32_t particle_b,
                                     float pos_x, float pos_y, float stress, float break_force);
    
    /**
     * Emit a particle collision event
     * 
     * @param particle_a ID of first particle
     * @param particle_b ID of second particle
     * @param pos_x Collision position X
     * @param pos_y Collision position Y
     * @param impulse Collision impulse magnitude
     * @param penetration Penetration depth
     * @param velocity_x Relative velocity X component
     * @param velocity_y Relative velocity Y component
     * @param is_hard True for inelastic collision, false for elastic
     */
    EventSystemError emit_collision(uint32_t particle_a, uint32_t particle_b,
                                  float pos_x, float pos_y, float impulse, float penetration,
                                  float velocity_x, float velocity_y, bool is_hard = false);
    
    /**
     * Emit a particle merge event
     * 
     * @param surviving_id ID of the particle that survives
     * @param absorbed_id ID of the particle that is absorbed
     * @param pos_x Merge position X
     * @param pos_y Merge position Y
     * @param mass_before Combined mass before merge
     * @param mass_after Mass of surviving particle after merge
     */
    EventSystemError emit_particle_merge(uint32_t surviving_id, uint32_t absorbed_id,
                                       float pos_x, float pos_y, 
                                       float mass_before, float mass_after);
    
    /**
     * Emit a black hole formation event
     * 
     * @param particle_id ID of the particle that collapsed
     * @param pos_x Black hole position X
     * @param pos_y Black hole position Y  
     * @param mass Black hole mass
     * @param event_horizon Event horizon radius
     */
    EventSystemError emit_black_hole_formation(uint32_t particle_id, float pos_x, float pos_y,
                                             float mass, float event_horizon);
    
    /**
     * Emit a composite body formation event
     * 
     * @param composite_id ID of the new composite body
     * @param center_x Center of mass X
     * @param center_y Center of mass Y
     * @param radius Bounding radius
     * @param particle_count Number of particles in composite
     * @param integrity Structural integrity (0.0 - 1.0)
     */
    EventSystemError emit_composite_formation(uint32_t composite_id, 
                                            float center_x, float center_y, float radius,
                                            uint32_t particle_count, float integrity);
    
    /**
     * Emit a resonance detection event
     * 
     * @param composite_id ID of the composite body
     * @param center_x Center of mass X
     * @param center_y Center of mass Y
     * @param radius Affected radius
     * @param frequency Resonance frequency
     * @param amplitude Resonance amplitude
     * @param integrity Current structural integrity
     */
    EventSystemError emit_resonance_detected(uint32_t composite_id,
                                           float center_x, float center_y, float radius,
                                           float frequency, float amplitude, float integrity);
    
    /**
     * Emit a thermal event
     * 
     * @param type Type of thermal event
     * @param particle_id ID of affected particle
     * @param pos_x Position X
     * @param pos_y Position Y
     * @param temp_before Temperature before event
     * @param temp_after Temperature after event
     * @param pressure Current pressure
     */
    EventSystemError emit_thermal_event(EventType type, uint32_t particle_id,
                                       float pos_x, float pos_y,
                                       float temp_before, float temp_after, float pressure);
    
    /**
     * Emit a system event (tick complete, checkpoint, etc.)
     * 
     * @param type Type of system event
     * @param magnitude Optional magnitude/value associated with event
     */
    EventSystemError emit_system_event(EventType type, float magnitude = 0.0f);
    
    /**
     * Performance and monitoring
     */
    
    /**
     * Get producer statistics
     */
    struct ProducerStats {
        uint64_t events_emitted;      ///< Total events successfully emitted
        uint64_t events_dropped;      ///< Total events dropped due to full buffer
        uint64_t batch_operations;    ///< Number of batch emit operations
        double average_emit_time_ns;  ///< Average time per emit operation (nanoseconds)
        double total_emit_time_ms;    ///< Total time spent emitting events (milliseconds)
    };
    
    ProducerStats get_stats() const { return stats_; }
    
    /**
     * Reset statistics counters
     */
    void reset_stats();
    
    /**
     * Check if the underlying buffer is healthy
     */
    bool is_healthy() const;
    
private:
    EventRingBuffer* buffer_;      ///< Pointer to shared ring buffer
    uint32_t current_tick_;        ///< Current simulation tick
    float current_timestamp_;      ///< Current simulation timestamp
    
    // Performance tracking
    ProducerStats stats_;
    std::chrono::high_resolution_clock::time_point last_timing_;
    
    /**
     * Internal emit implementation with timing
     */
    EventSystemError emit_internal(const Event& event);
    
    /**
     * Get minimum consumer position (cached for performance)
     */
    uint64_t get_min_consumer_pos_cached() const;
    
    /**
     * Update timing statistics
     */
    void update_timing(const std::chrono::high_resolution_clock::time_point& start);
};

/**
 * Physics event emitter - higher-level interface for physics backends
 * 
 * This class provides a convenient interface for physics backends to emit events
 * based on physics state changes. It integrates with the backend architecture and
 * provides automatic event generation for common physics scenarios.
 */
class PhysicsEventEmitter {
public:
    /**
     * Constructor
     * 
     * @param producer Event producer instance
     */
    explicit PhysicsEventEmitter(std::shared_ptr<EventProducer> producer);
    
    /**
     * Set the current physics state for event context
     * 
     * @param tick Current simulation tick
     * @param timestamp Current simulation time
     */
    void set_physics_state(uint32_t tick, float timestamp);
    
    /**
     * Physics-specific event emission methods
     * These methods understand physics data structures and automatically
     * extract relevant information for events.
     */
    
    /**
     * Monitor particle interactions and emit relevant events
     * This would typically be called after contact resolution.
     */
    void process_particle_interactions(/* particle and contact data structures */);
    
    /**
     * Monitor spring network and emit spring-related events
     * This would typically be called after spring force calculation.
     */
    void process_spring_network(/* spring data structures */);
    
    /**
     * Monitor composite bodies and emit structural events
     * This would typically be called after composite body analysis.
     */
    void process_composite_bodies(/* composite body data structures */);
    
    /**
     * Emit tick completion event with performance statistics
     * 
     * @param stats Simulation performance statistics
     */
    void emit_tick_complete(const SimulationStats& stats);
    
    /**
     * Get event emission statistics
     */
    EventProducer::ProducerStats get_producer_stats() const {
        return producer_->get_stats();
    }
    
private:
    std::shared_ptr<EventProducer> producer_;
    uint32_t current_tick_;
    float current_timestamp_;
};

} // namespace digistar
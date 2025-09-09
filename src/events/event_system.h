#pragma once

#include <atomic>
#include <cstdint>
#include <string>
#include <memory>
#include <sys/types.h>

namespace digistar {

/**
 * Event types for the DigiStar physics simulation
 * 
 * Each event type is carefully chosen to represent significant physics
 * occurrences that clients (renderers, audio, network) need to know about.
 * Events are designed to be 64-byte cache-aligned for optimal performance.
 */
enum class EventType : uint16_t {
    // Particle Events (0x0001 - 0x000F)
    PARTICLE_MERGE = 0x0001,        ///< Two particles merged into one
    PARTICLE_FISSION = 0x0002,      ///< One particle split into multiple
    STAR_FORMATION = 0x0003,        ///< Fusion ignition occurred
    BLACK_HOLE_FORMATION = 0x0004,  ///< Particle collapsed into black hole
    BLACK_HOLE_ABSORPTION = 0x0005, ///< Black hole absorbed another object
    
    // Spring Events (0x0010 - 0x001F)
    SPRING_CREATED = 0x0010,        ///< New spring connection formed
    SPRING_BROKEN = 0x0011,         ///< Spring connection broke
    SPRING_CRITICAL_STRESS = 0x0012,///< Spring approaching break point
    
    // Collision Events (0x0020 - 0x002F)
    SOFT_CONTACT = 0x0020,          ///< Elastic collision between particles
    HARD_COLLISION = 0x0021,        ///< High-energy inelastic collision
    COMPOSITE_COLLISION = 0x0022,   ///< Collision involving composite body
    
    // Thermal Events (0x0030 - 0x003F)
    PHASE_TRANSITION = 0x0030,      ///< Material phase change
    FUSION_IGNITION = 0x0031,       ///< Nuclear fusion started
    THERMAL_EXPLOSION = 0x0032,     ///< Rapid thermal expansion
    
    // Composite Events (0x0040 - 0x004F)
    COMPOSITE_FORMED = 0x0040,      ///< Spring network formed composite body
    COMPOSITE_BROKEN = 0x0041,      ///< Composite body disintegrated
    RESONANCE_DETECTED = 0x0042,    ///< Harmonic resonance in structure
    STRUCTURAL_FAILURE = 0x0043,    ///< Critical structural damage
    
    // Tidal Events (0x0050 - 0x005F)
    TIDAL_DISRUPTION = 0x0050,      ///< Object torn apart by tidal forces
    ROCHE_LIMIT_BREACH = 0x0051,    ///< Object crossed Roche limit
    
    // Player Events (0x0100 - 0x010F)
    PLAYER_ACTION = 0x0100,         ///< Player performed action
    PLAYER_JOIN = 0x0101,           ///< Player joined simulation
    PLAYER_LEAVE = 0x0102,          ///< Player left simulation
    
    // System Events (0x0200 - 0x020F)
    TICK_COMPLETE = 0x0200,         ///< Simulation tick completed
    CHECKPOINT = 0x0201,            ///< Simulation state checkpoint
    STATS_UPDATE = 0x0202           ///< Performance statistics updated
};

/**
 * Event flags for metadata and routing
 */
enum EventFlags : uint16_t {
    FLAG_NONE = 0x0000,
    FLAG_HIGH_PRIORITY = 0x0001,     ///< Process before other events
    FLAG_NETWORK_BROADCAST = 0x0002, ///< Send to network clients
    FLAG_AUTHORITATIVE = 0x0004,     ///< Server-authoritative event
    FLAG_AGGREGATE = 0x0008,         ///< Can be aggregated with similar events
    FLAG_SPATIAL_FILTER = 0x0010     ///< Subject to spatial filtering
};

/**
 * Fixed-size event structure optimized for lock-free ring buffer
 * 
 * This structure is exactly 64 bytes (one cache line) for optimal performance.
 * The layout is carefully designed to minimize false sharing and maximize
 * cache efficiency.
 */
struct alignas(64) Event {
    // Header (8 bytes)
    EventType type;                  ///< What kind of event occurred
    uint16_t flags;                  ///< Event metadata and routing hints
    uint32_t tick;                   ///< Simulation tick when event occurred
    
    // Timing (4 bytes)
    float timestamp;                 ///< Simulation time in seconds
    
    // Spatial information (12 bytes)
    float x, y;                      ///< Event position in world coordinates
    float radius;                    ///< Affected area radius
    
    // Participants (8 bytes)
    uint32_t primary_id;             ///< Main particle/object involved
    uint32_t secondary_id;           ///< Secondary participant (or 0xFFFFFFFF if none)
    
    // Magnitude (8 bytes)
    float magnitude;                 ///< Primary measurement (force, energy, mass, etc.)
    float secondary_value;           ///< Additional measurement (temperature, velocity, etc.)
    
    // Type-specific data (20 bytes)
    union {
        struct { float mass_before, mass_after; uint8_t reserved[12]; } merge;
        struct { float stress, break_force; uint8_t reserved[12]; } spring;
        struct { float penetration, impulse; float velocity_x, velocity_y; uint8_t reserved[4]; } collision;
        struct { float temp_before, temp_after; float pressure; uint8_t reserved[8]; } thermal;
        struct { uint32_t particle_count; float integrity; uint8_t reserved[12]; } composite;
        struct { float frequency; float amplitude; uint8_t reserved[12]; } resonance;
        uint8_t raw_data[20];        ///< Raw access for custom event types
    } data;
    
    // Padding to reach exactly 64 bytes (4 bytes)
    uint8_t padding[4];
    
    // Ensure exact 64-byte size
    static_assert(sizeof(EventType) == 2, "EventType must be 2 bytes");
    static_assert(sizeof(float) == 4, "float must be 4 bytes");
    static_assert(sizeof(uint32_t) == 4, "uint32_t must be 4 bytes");
};

static_assert(sizeof(Event) == 64, "Event must be exactly 64 bytes for cache alignment");

/**
 * Consumer registration information
 * 
 * Tracks information about processes consuming events from the ring buffer.
 */
struct Consumer {
    pid_t pid;                       ///< Process ID of consumer
    char name[32];                   ///< Human-readable consumer name
    std::atomic<uint64_t> events_read; ///< Total events read by this consumer
    std::atomic<uint64_t> last_read_time; ///< Timestamp of last read (for cleanup)
    
    Consumer() : pid(0), events_read(0), last_read_time(0) {
        name[0] = '\0';
    }
};

/**
 * Lock-free ring buffer for high-performance event distribution
 * 
 * This structure is designed to be placed in shared memory (posix_shm) for
 * zero-copy IPC between the physics engine and client processes. The design
 * supports one producer (physics engine) and up to MAX_CONSUMERS readers.
 * 
 * The ring buffer uses atomic operations for synchronization and is optimized
 * for the common case where consumers are keeping up with the producer.
 * When consumers fall behind, events are dropped rather than blocking the
 * physics simulation.
 */
struct EventRingBuffer {
    static constexpr size_t CAPACITY = 65536;     ///< Must be power of 2 for fast modulo
    static constexpr size_t MAX_CONSUMERS = 8;    ///< Maximum concurrent consumers
    static constexpr uint32_t MAGIC = 0xD161574A; ///< Magic number for validation
    static constexpr uint32_t VERSION = 1;        ///< Structure version
    
    // Producer state (cache line 1)
    alignas(64) std::atomic<uint64_t> write_pos;
    uint8_t producer_padding[56];
    
    // Consumer positions (cache line 2)
    alignas(64) std::atomic<uint64_t> read_pos[MAX_CONSUMERS];
    
    // Metadata (cache line 3)
    alignas(64) uint32_t magic;                   ///< Magic number for validation
    uint32_t version;                             ///< Structure version
    uint32_t capacity;                            ///< Buffer capacity
    uint32_t event_size;                          ///< Size of each event
    std::atomic<uint32_t> num_consumers;          ///< Number of active consumers
    std::atomic<uint64_t> total_events;           ///< Total events produced
    std::atomic<uint64_t> dropped_events;         ///< Events dropped due to full buffer
    uint8_t metadata_padding[32];
    
    // Event data (aligned)
    alignas(64) Event events[CAPACITY];
    
    // Consumer information
    Consumer consumers[MAX_CONSUMERS];
    
    /**
     * Initialize the ring buffer
     */
    EventRingBuffer() 
        : write_pos(0), magic(MAGIC), version(VERSION), 
          capacity(CAPACITY), event_size(sizeof(Event)),
          num_consumers(0), total_events(0), dropped_events(0) {
        
        // Initialize consumer read positions
        for (int i = 0; i < MAX_CONSUMERS; i++) {
            read_pos[i].store(0, std::memory_order_relaxed);
        }
    }
    
    /**
     * Validate buffer integrity
     */
    bool is_valid() const {
        return magic == MAGIC && version == VERSION && 
               capacity == CAPACITY && event_size == sizeof(Event);
    }
    
    /**
     * Get the minimum read position across all consumers
     * This determines how far the producer can advance before overwriting unread events.
     */
    uint64_t get_min_consumer_pos() const {
        uint64_t min_read = UINT64_MAX;
        uint32_t consumers = num_consumers.load(std::memory_order_acquire);
        
        for (uint32_t i = 0; i < consumers && i < MAX_CONSUMERS; i++) {
            uint64_t pos = read_pos[i].load(std::memory_order_acquire);
            min_read = std::min(min_read, pos);
        }
        
        return (min_read == UINT64_MAX) ? write_pos.load(std::memory_order_acquire) : min_read;
    }
    
    /**
     * Check if buffer has space for n events
     */
    bool has_space(size_t n) const {
        uint64_t write = write_pos.load(std::memory_order_relaxed);
        uint64_t min_read = get_min_consumer_pos();
        return (write - min_read + n) <= capacity;
    }
    
    /**
     * Get current buffer utilization (0.0 to 1.0)
     */
    float get_utilization() const {
        uint64_t write = write_pos.load(std::memory_order_relaxed);
        uint64_t min_read = get_min_consumer_pos();
        return static_cast<float>(write - min_read) / capacity;
    }
};

/**
 * Shared memory event system
 * 
 * Provides a high-level interface for creating and managing the event ring buffer
 * in shared memory. This class handles the lifecycle of the shared memory segment
 * and provides factory methods for creating producers and consumers.
 */
class SharedMemoryEventSystem {
public:
    /**
     * Create or attach to a shared memory event system
     * 
     * @param shm_name Name of the shared memory segment
     * @param create_new If true, create a new segment; if false, attach to existing
     */
    explicit SharedMemoryEventSystem(const std::string& shm_name, bool create_new = true);
    
    /**
     * Destructor - cleans up shared memory resources
     */
    ~SharedMemoryEventSystem();
    
    /**
     * Get the underlying ring buffer (for direct access)
     */
    EventRingBuffer* get_buffer() const { return buffer_; }
    
    /**
     * Check if the event system is valid and ready to use
     */
    bool is_valid() const;
    
    /**
     * Get system statistics
     */
    struct Stats {
        uint64_t total_events;
        uint64_t dropped_events;
        uint32_t num_consumers;
        float buffer_utilization;
        size_t buffer_size_bytes;
    };
    
    Stats get_stats() const;

private:
    std::string shm_name_;
    int shm_fd_;
    EventRingBuffer* buffer_;
    size_t buffer_size_;
    bool is_owner_;  ///< True if this instance created the shared memory
    
    void create_shared_memory();
    void attach_shared_memory();
    void cleanup();
};

/**
 * Event system error types
 */
enum class EventSystemError {
    SUCCESS = 0,
    BUFFER_FULL,
    INVALID_CONSUMER,
    NO_EVENTS,
    SHARED_MEMORY_ERROR,
    INVALID_BUFFER,
    TOO_MANY_CONSUMERS
};

/**
 * Convert error code to human-readable string
 */
const char* event_system_error_string(EventSystemError error);

} // namespace digistar
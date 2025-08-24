# Event System Design

## Core Philosophy

Events are the nervous system of DigiStar - they notify all interested parties (renderers, audio, network clients) about significant physics occurrences without tight coupling. Zero-copy shared memory for local clients, efficient serialization for network clients.

## Event Types

### Physics Events

```cpp
enum class EventType : uint16_t {
    // Particle Events
    PARTICLE_MERGE = 0x0001,
    PARTICLE_FISSION = 0x0002,
    STAR_FORMATION = 0x0003,
    BLACK_HOLE_FORMATION = 0x0004,
    BLACK_HOLE_ABSORPTION = 0x0005,
    
    // Spring Events  
    SPRING_CREATED = 0x0010,
    SPRING_BROKEN = 0x0011,
    SPRING_CRITICAL_STRESS = 0x0012,
    
    // Collision Events
    SOFT_CONTACT = 0x0020,
    HARD_COLLISION = 0x0021,
    COMPOSITE_COLLISION = 0x0022,
    
    // Thermal Events
    PHASE_TRANSITION = 0x0030,
    FUSION_IGNITION = 0x0031,
    THERMAL_EXPLOSION = 0x0032,
    
    // Composite Events
    COMPOSITE_FORMED = 0x0040,
    COMPOSITE_BROKEN = 0x0041,
    RESONANCE_DETECTED = 0x0042,
    STRUCTURAL_FAILURE = 0x0043,
    
    // Tidal Events
    TIDAL_DISRUPTION = 0x0050,
    ROCHE_LIMIT_BREACH = 0x0051,
    
    // Player Events (networked)
    PLAYER_ACTION = 0x0100,
    PLAYER_JOIN = 0x0101,
    PLAYER_LEAVE = 0x0102,
    
    // System Events
    TICK_COMPLETE = 0x0200,
    CHECKPOINT = 0x0201,
    STATS_UPDATE = 0x0202
};
```

### Event Structure

```cpp
// Fixed-size event for lock-free ring buffer
struct alignas(64) Event {  // Cache line aligned
    EventType type;
    uint16_t flags;          // Priority, network-relevant, etc.
    uint32_t tick;           // Simulation tick number
    float timestamp;         // Simulation time
    
    // Event location
    float x, y;              // Position in world
    float radius;            // Affected area
    
    // Event participants (indices)
    uint32_t primary_id;     // Main particle/spring/composite
    uint32_t secondary_id;   // Other participant (collision, merge, etc.)
    
    // Event magnitude
    float magnitude;         // Force, energy, mass, etc.
    float secondary_value;   // Temperature, velocity, etc.
    
    // Additional data (type-specific)
    union {
        struct { float mass_before, mass_after; } merge;
        struct { float stress, break_force; } spring;
        struct { float penetration, impulse; } collision;
        struct { float temp_before, temp_after; } thermal;
        struct { uint32_t particle_count; float integrity; } composite;
        uint8_t raw_data[32];
    } data;
    
    // Padding to 64 bytes
    uint8_t padding[8];
};

static_assert(sizeof(Event) == 64, "Event must be exactly 64 bytes");
```

## Lock-Free Ring Buffer

### Shared Memory Layout

```cpp
// In posix_shm segment
struct EventRingBuffer {
    static constexpr size_t CAPACITY = 65536;  // Power of 2
    
    // Cache line 1: Producer
    alignas(64) std::atomic<uint64_t> write_pos;
    uint8_t producer_padding[56];
    
    // Cache line 2: Consumer positions (multiple readers)
    alignas(64) std::atomic<uint64_t> read_pos[8];  // Up to 8 consumers
    
    // Cache line 3: Metadata
    alignas(64) uint32_t magic;  // 0xD161574R
    uint32_t version;
    uint32_t capacity;
    uint32_t event_size;
    std::atomic<uint32_t> num_consumers;
    std::atomic<uint64_t> total_events;
    std::atomic<uint64_t> dropped_events;
    uint8_t metadata_padding[32];
    
    // Event data
    alignas(64) Event events[CAPACITY];
    
    // Consumer registration
    struct Consumer {
        pid_t pid;
        char name[32];
        uint64_t events_read;
        uint64_t last_read_time;
    } consumers[8];
};
```

### Producer (Physics Engine)

```cpp
class EventProducer {
    EventRingBuffer* buffer;
    uint64_t local_write_pos;
    
public:
    void emit(const Event& event) {
        // Get minimum consumer position (slowest reader)
        uint64_t min_read = UINT64_MAX;
        for (int i = 0; i < buffer->num_consumers; i++) {
            uint64_t read = buffer->read_pos[i].load(std::memory_order_acquire);
            min_read = std::min(min_read, read);
        }
        
        uint64_t write = buffer->write_pos.load(std::memory_order_relaxed);
        
        // Check if buffer is full
        if (write - min_read >= buffer->capacity) {
            buffer->dropped_events.fetch_add(1, std::memory_order_relaxed);
            return;  // Drop event
        }
        
        // Write event
        uint64_t slot = write & (buffer->capacity - 1);
        buffer->events[slot] = event;
        
        // Publish
        buffer->write_pos.store(write + 1, std::memory_order_release);
        buffer->total_events.fetch_add(1, std::memory_order_relaxed);
    }
    
    // Batch emit for efficiency
    void emit_batch(const Event* events, size_t count) {
        uint64_t write = buffer->write_pos.load(std::memory_order_relaxed);
        uint64_t min_read = get_min_consumer_pos();
        
        size_t available = buffer->capacity - (write - min_read);
        size_t to_write = std::min(count, available);
        
        for (size_t i = 0; i < to_write; i++) {
            uint64_t slot = (write + i) & (buffer->capacity - 1);
            buffer->events[slot] = events[i];
        }
        
        buffer->write_pos.store(write + to_write, std::memory_order_release);
        buffer->total_events.fetch_add(to_write, std::memory_order_relaxed);
        
        if (to_write < count) {
            buffer->dropped_events.fetch_add(count - to_write, std::memory_order_relaxed);
        }
    }
};
```

### Consumer (Renderer/Audio/Network)

```cpp
class EventConsumer {
    EventRingBuffer* buffer;
    int consumer_id;
    uint64_t local_read_pos;
    
public:
    bool poll(Event& event) {
        uint64_t read = local_read_pos;
        uint64_t write = buffer->write_pos.load(std::memory_order_acquire);
        
        if (read >= write) {
            return false;  // No new events
        }
        
        // Read event
        uint64_t slot = read & (buffer->capacity - 1);
        event = buffer->events[slot];
        
        // Update position
        local_read_pos = read + 1;
        buffer->read_pos[consumer_id].store(read + 1, std::memory_order_release);
        
        return true;
    }
    
    // Batch read for efficiency
    size_t poll_batch(Event* events, size_t max_count) {
        uint64_t read = local_read_pos;
        uint64_t write = buffer->write_pos.load(std::memory_order_acquire);
        
        size_t available = write - read;
        size_t to_read = std::min(available, max_count);
        
        for (size_t i = 0; i < to_read; i++) {
            uint64_t slot = (read + i) & (buffer->capacity - 1);
            events[i] = buffer->events[slot];
        }
        
        local_read_pos = read + to_read;
        buffer->read_pos[consumer_id].store(read + to_read, std::memory_order_release);
        
        return to_read;
    }
};
```

## Event Generation

### Physics Integration

```cpp
class PhysicsEventEmitter {
    EventProducer producer;
    uint32_t current_tick;
    float current_time;
    
    void on_spring_break(const Spring& spring, float stress) {
        Event e;
        e.type = EventType::SPRING_BROKEN;
        e.tick = current_tick;
        e.timestamp = current_time;
        e.x = (particles[spring.i].pos.x + particles[spring.j].pos.x) / 2;
        e.y = (particles[spring.i].pos.y + particles[spring.j].pos.y) / 2;
        e.primary_id = spring.i;
        e.secondary_id = spring.j;
        e.magnitude = stress;
        e.data.spring.stress = stress;
        e.data.spring.break_force = spring.break_force;
        
        producer.emit(e);
    }
    
    void on_black_hole_formation(const Particle& p) {
        Event e;
        e.type = EventType::BLACK_HOLE_FORMATION;
        e.tick = current_tick;
        e.timestamp = current_time;
        e.x = p.pos.x;
        e.y = p.pos.y;
        e.primary_id = p.id;
        e.magnitude = p.mass;
        e.radius = p.event_horizon;
        
        // High priority event
        e.flags = FLAG_HIGH_PRIORITY | FLAG_NETWORK_BROADCAST;
        
        producer.emit(e);
    }
    
    void on_composite_resonance(const CompositeBody& body, float frequency) {
        Event e;
        e.type = EventType::RESONANCE_DETECTED;
        e.tick = current_tick;
        e.timestamp = current_time;
        e.x = body.center_of_mass.x;
        e.y = body.center_of_mass.y;
        e.radius = body.bounding_radius;
        e.magnitude = frequency;
        e.secondary_value = body.structural_integrity;
        e.data.composite.particle_count = body.num_particles;
        e.data.composite.integrity = body.structural_integrity;
        
        producer.emit(e);
    }
};
```

## Client Examples

### Audio Client

```cpp
class AudioClient : public EventConsumer {
    void process_events() {
        Event events[100];
        size_t count = poll_batch(events, 100);
        
        for (size_t i = 0; i < count; i++) {
            const Event& e = events[i];
            
            switch (e.type) {
            case EventType::SPRING_BROKEN:
                // Crack sound based on stress
                play_sound("crack", e.x, e.y, 
                          volume_from_magnitude(e.magnitude));
                break;
                
            case EventType::BLACK_HOLE_FORMATION:
                // Deep rumble with reverb
                play_sound("black_hole_birth", e.x, e.y, 1.0f);
                start_ambient("black_hole_drone", e.primary_id);
                break;
                
            case EventType::RESONANCE_DETECTED:
                // Vibration sound at frequency
                play_tone(e.magnitude, e.x, e.y, 
                         1.0f - e.secondary_value);  // Volume from integrity
                break;
                
            case EventType::STAR_FORMATION:
                // Ignition whoosh
                play_sound("fusion_ignite", e.x, e.y, 
                          volume_from_magnitude(e.magnitude));
                start_ambient("star_hum", e.primary_id);
                break;
            }
        }
    }
};
```

### Visual Effects Client

```cpp
class VFXClient : public EventConsumer {
    void process_events() {
        Event e;
        while (poll(e)) {
            switch (e.type) {
            case EventType::BLACK_HOLE_FORMATION:
                // Gravitational lens effect
                spawn_effect("gravitational_collapse", e.x, e.y);
                add_persistent_effect("event_horizon", e.primary_id, e.radius);
                screen_shake(e.magnitude * 0.001f);
                break;
                
            case EventType::PARTICLE_MERGE:
                // Flash and particle spray
                spawn_effect("impact_flash", e.x, e.y, 
                            scale_from_mass(e.data.merge.mass_after));
                break;
                
            case EventType::TIDAL_DISRUPTION:
                // Stretching effect
                spawn_effect("spaghettification", e.x, e.y);
                create_particle_trail(e.primary_id, e.secondary_id);
                break;
                
            case EventType::STRUCTURAL_FAILURE:
                // Debris particles
                spawn_debris(e.x, e.y, e.data.composite.particle_count);
                screen_shake(0.1f);
                break;
            }
        }
    }
};
```

## Network Protocol

### Event Serialization

```cpp
class NetworkEventSerializer {
    // Compact binary format for network transmission
    struct NetworkEvent {
        uint16_t type;
        uint16_t flags;
        uint32_t tick;
        float x, y;
        uint32_t data[4];  // Type-specific, compressed
    } __attribute__((packed));
    
    NetworkEvent compress(const Event& e) {
        NetworkEvent ne;
        ne.type = static_cast<uint16_t>(e.type);
        ne.flags = e.flags;
        ne.tick = e.tick;
        ne.x = quantize_position(e.x);
        ne.y = quantize_position(e.y);
        
        // Type-specific compression
        switch (e.type) {
        case EventType::SPRING_BROKEN:
            ne.data[0] = e.primary_id;
            ne.data[1] = e.secondary_id;
            ne.data[2] = quantize_float(e.magnitude);
            break;
        // ... other types
        }
        
        return ne;
    }
};
```

### Network Distribution

```cpp
class NetworkEventDistributor {
    std::vector<NetworkClient> clients;
    EventConsumer consumer;
    
    void distribute_events() {
        Event events[1000];
        size_t count = consumer.poll_batch(events, 1000);
        
        // Filter network-relevant events
        std::vector<NetworkEvent> to_send;
        for (size_t i = 0; i < count; i++) {
            if (events[i].flags & FLAG_NETWORK_BROADCAST) {
                to_send.push_back(compress(events[i]));
            }
        }
        
        // Batch send to all clients
        for (auto& client : clients) {
            client.send_events(to_send);
        }
    }
};
```

### Player Actions

```cpp
class PlayerActionHandler {
    EventProducer producer;
    
    void on_player_action(uint32_t player_id, const Action& action) {
        Event e;
        e.type = EventType::PLAYER_ACTION;
        e.tick = current_tick;
        e.timestamp = current_time;
        e.primary_id = player_id;
        e.x = action.target_x;
        e.y = action.target_y;
        e.magnitude = action.force;
        e.flags = FLAG_NETWORK_BROADCAST | FLAG_AUTHORITATIVE;
        
        // Emit locally
        producer.emit(e);
        
        // Also broadcast to network
        broadcast_to_peers(e);
    }
};
```

## Performance Optimizations

### Event Filtering

```cpp
class FilteredEventConsumer {
    EventConsumer base_consumer;
    std::bitset<65536> type_filter;  // Which event types to process
    float area_of_interest_x, area_of_interest_y, area_radius;
    
    bool poll_filtered(Event& event) {
        while (base_consumer.poll(event)) {
            // Type filter
            if (!type_filter[static_cast<uint16_t>(event.type)]) {
                continue;
            }
            
            // Spatial filter
            float dx = event.x - area_of_interest_x;
            float dy = event.y - area_of_interest_y;
            if (dx*dx + dy*dy > area_radius * area_radius) {
                continue;
            }
            
            return true;
        }
        return false;
    }
};
```

### Event Aggregation

```cpp
class EventAggregator {
    // Combine multiple similar events
    struct AggregatedEvent {
        EventType type;
        float x, y;
        float total_magnitude;
        uint32_t count;
        float time_window;
    };
    
    std::unordered_map<uint64_t, AggregatedEvent> aggregates;
    
    void aggregate(const Event& e) {
        uint64_t key = spatial_hash(e.x, e.y) ^ static_cast<uint64_t>(e.type);
        
        auto& agg = aggregates[key];
        agg.type = e.type;
        agg.x = (agg.x * agg.count + e.x) / (agg.count + 1);
        agg.y = (agg.y * agg.count + e.y) / (agg.count + 1);
        agg.total_magnitude += e.magnitude;
        agg.count++;
    }
    
    void flush_aggregates(float time_window = 0.1f) {
        for (auto& [key, agg] : aggregates) {
            if (agg.time_window >= time_window) {
                emit_aggregated_event(agg);
                aggregates.erase(key);
            }
        }
    }
};
```

## Integration with posix_shm

```cpp
class SharedMemoryEventSystem {
    posix_shm::memory_pool pool;
    EventRingBuffer* buffer;
    
public:
    SharedMemoryEventSystem(const std::string& shm_name) 
        : pool(shm_name, sizeof(EventRingBuffer)) {
        
        buffer = pool.construct<EventRingBuffer>("event_buffer");
        buffer->magic = 0xD161574R;
        buffer->version = 1;
        buffer->capacity = EventRingBuffer::CAPACITY;
        buffer->event_size = sizeof(Event);
    }
    
    EventProducer create_producer() {
        return EventProducer{buffer};
    }
    
    EventConsumer create_consumer(const std::string& name) {
        int id = buffer->num_consumers.fetch_add(1);
        if (id >= 8) throw std::runtime_error("Too many consumers");
        
        buffer->consumers[id].pid = getpid();
        strncpy(buffer->consumers[id].name, name.c_str(), 31);
        
        return EventConsumer{buffer, id};
    }
};
```

## Event Flow Architecture

```
┌─────────────────┐
│  Physics Engine │
│                 │
│ - Spring breaks │
│ - Collisions    │
│ - Mergers       │
│ - Black holes   │
└────────┬────────┘
         │ emit()
         ▼
┌─────────────────┐
│  Event Buffer   │
│  (posix_shm)    │
│                 │
│  Lock-free      │
│  Ring Buffer    │
└────────┬────────┘
         │ poll()
    ┌────┴────┬────────┬──────────┐
    ▼         ▼        ▼          ▼
┌────────┐ ┌──────┐ ┌──────┐ ┌─────────┐
│OpenGL  │ │Audio │ │Stats │ │Network  │
│Renderer│ │Engine│ │Logger│ │Relay    │
└────────┘ └──────┘ └──────┘ └────┬────┘
                                   │
                              ┌────▼────┐
                              │ Remote  │
                              │ Clients │
                              └─────────┘
```

## Benefits

1. **Zero-Copy Local IPC** - Events shared via posix_shm, no serialization
2. **Lock-Free** - No mutex contention between producer/consumers
3. **Cache-Friendly** - 64-byte aligned events, one per cache line
4. **Scalable** - Multiple consumers can read independently
5. **Network-Ready** - Events designed for efficient serialization
6. **Extensible** - New event types easily added
7. **Debuggable** - Event stream can be logged/replayed

This event system provides the nervous system connecting all DigiStar components!
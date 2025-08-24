# Event System Architecture

## Overview

DigiStar's event system is designed to handle millions of events per second while providing flexible observation patterns for scripts. It combines channel-based routing, entity-specific observation, and intelligent filtering to achieve both scale and precision.

## Three-Layer Architecture

### 1. Channel-Based Events (Broad Categories)

Hierarchical publish-subscribe system for system-wide events.

```cpp
// Channel patterns like MQTT topics
"physics/collision/*"
"physics/spring/break"
"composite/formed"
"game/player/*"
```

**Implementation:**
```cpp
class HierarchicalEventBus {
    struct ChannelNode {
        std::vector<Subscription> subs;
        std::unordered_map<std::string, unique_ptr<ChannelNode>> children;
        BloomFilter quick_reject;  // Fast "no subscribers" check
    };
    
    void emit(const Event& event) {
        auto& queue = thread_queues[thread_id];
        queue.push(event);  // Lock-free, O(1)
    }
};
```

### 2. Entity-Specific Observation (Direct Tracking)

Observer pattern enhanced for entity lifecycle management.

```cpp
template<typename T>
class EntityObserver {
    // Observe specific entities by ID
    ObserverId observe(EntityHandle<T> entity, ObserverCallback cb);
    
    // Automatic cleanup when entities are destroyed
    void cleanupExpired();
};
```

**Use Cases:**
- Track specific spaceships, planets, black holes
- Monitor composite bodies of interest
- Follow particle trajectories

### 3. Property Watchers (Computed Values)

Monitor changes in computed properties with configurable thresholds.

```cpp
class PropertyWatcher {
    WatcherId watch(EntityHandle entity, 
                   const std::string& property,
                   PropertyHandler handler,
                   float threshold = 0.01);  // Only fire on significant change
};
```

## Event Filtering Strategies

### Hierarchical Filtering
```cpp
// Subscribe with pattern matching
event_system.subscribe(
    "physics/spring/*/break",  // Wildcard pattern
    handler,
    SpatialFilter{region}       // Additional filter
);
```

### Spatial Filtering
Events filtered by location using spatial index:
```cpp
struct SpatialFilter {
    Vec3 center;
    float radius;
    bool accepts(const Event& e) {
        return distance(e.position, center) < radius;
    }
};
```

### Bloom Filter Pre-Rejection
Fast probabilistic check to avoid unnecessary routing:
```cpp
if (!channel.bloom_filter.mightContain(event.type)) {
    return;  // Definitely no subscribers
}
```

## High-Frequency Event Handling

### Event Aggregation
For millions of spring events per second:

```cpp
class SpringEventAggregator {
    struct RegionStats {
        int breaks = 0;
        int formations = 0;
        float avg_stress = 0;
        Vec3 center_of_activity;
    };
    
    void flushAggregated() {
        for (auto& [cell, stat] : stats) {
            if (stat.breaks > threshold) {
                emit(SpringBreakCluster{cell, stat.breaks, ...});
            }
        }
    }
};
```

### Level-of-Detail (LOD) System
```cpp
enum EventLOD {
    CRITICAL = 0,    // Always deliver
    DETAILED = 1,    // Nearby/important
    SUMMARY = 2,     // Aggregated only
    STATISTICAL = 3  // Count only
};
```

## DSL Integration

### Event Subscription
```lisp
; Channel subscription with filters
(on-event "physics/collision"
  (lambda (event)
    (when (> (event-field event 'energy) 1000)
      (create-explosion (event-field event 'pos))))
  :filter (within-radius [0 0 0] 1000))

; Entity observation
(observe my-ship
  (lambda (event)
    (case (event-type event)
      ['damage (repair-damage my-ship)]
      ['fuel-low (navigate-to-fuel my-ship)])))

; Property watching
(watch-property my-ship 'structural-integrity
  (lambda (old new)
    (when (< new 0.5)
      (abandon-ship my-ship))))

; Aggregated events
(on-event-aggregate "physics/spring/*"
  (lambda (stats)
    (when (> (stats 'break-count) 100)
      (log "Major structural failure!")))
  :window 0.1
  :min-count 10)
```

## Thread Safety & Performance

### Lock-Free Event Queues
```cpp
// Per-thread queues avoid contention
std::vector<LockFreeQueue<Event>> thread_queues;

// Batch processing between frames
void processEvents() {
    auto events = collectFromQueues();  // Gather all
    routeToSubscribers(events);         // Can parallelize
}
```

### Memory Management
```cpp
// Object pool for event allocation
ObjectPool<Event> event_pool(10000);

// Weak references for entity observation
std::unordered_map<EntityId, std::weak_ptr<Entity>> entity_refs;
```

## Performance Targets

- **Event Rate**: 10M+ events/second
- **Subscription Lookup**: O(log n) for channels, O(1) for entities
- **Filtering**: < 10ns per filter check
- **Memory**: < 1KB per subscription
- **Latency**: < 1ms from emit to handler (same frame)

## Implementation Phases

### Phase 1: Core Infrastructure
- Basic event bus with channels
- Lock-free queues
- Simple filtering

### Phase 2: Entity Observation
- Entity handles and lifecycle
- Property watchers
- Weak reference management

### Phase 3: Scale Optimizations
- Event aggregation
- Bloom filters
- Spatial indexing
- LOD system

### Phase 4: DSL Integration
- Script event handlers
- Filter DSL
- Aggregation primitives

## Example: Complete Event Flow

```lisp
; Script monitoring a space battle
(define (battle-monitor)
  ; Watch for ship explosions
  (on-event "ship/explosion"
    (lambda (event)
      (let ([pos (event-field event 'position)]
            [yield (event-field event 'energy)])
        ; Create debris field
        (cloud :n (* 100 (/ yield 1e6))
               :center pos
               :radius 100
               :velocity-dist (explosion-profile yield)))))
  
  ; Track specific flagship
  (define flagship (find-ship "USS Victory"))
  (observe flagship
    (lambda (event)
      (when (eq? (event-type event) 'critical-damage)
        (order-retreat-all))))
  
  ; Monitor overall battle intensity
  (on-event-aggregate "combat/*"
    (lambda (stats)
      (when (> (stats 'total-damage) 1e12)
        (escalate-to-capital-ships)))
    :window 1.0))
```

## Future Enhancements

1. **Persistent Event Log**: Record events for replay/debugging
2. **Remote Events**: Network distribution for multiplayer
3. **Event Prediction**: Pre-allocate handlers for expected events
4. **Query Language**: SQL-like queries over event streams
5. **Time Travel**: Rewind and replay with different observers
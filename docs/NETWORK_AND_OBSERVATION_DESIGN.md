# Network and Observation Design

## Overview

DigiStar simulates 10+ million particles with complex interactions. No client can receive or process the complete world state. This document describes how clients observe and interact with the simulation through hierarchical, multi-resolution views.

## Core Principles

1. **The physics engine doesn't handle networking** - It provides queries and events
2. **Composites enable efficient simplification** - Send whole bodies or their parts based on distance
3. **Multiple resolution zones** - Different detail levels at different distances
4. **Server authoritative** - Physics engine is the single source of truth
5. **Event-driven updates** - Changes propagate through the event system

## Observation Hierarchy

### Distance-Based Level of Detail

Clients observe the world through concentric zones of decreasing detail:

```
┌─────────────────────────────────────────────┐
│                                             │
│            FAR FIELD (Statistical)          │
│   ┌─────────────────────────────────┐       │
│   │                                 │       │
│   │    MEDIUM FIELD (Simplified)    │       │
│   │   ┌─────────────────────┐       │       │
│   │   │                     │       │       │
│   │   │   NEAR FIELD        │       │       │
│   │   │   (Full Detail)     │       │       │
│   │   │                     │       │       │
│   │   │      [PLAYER]       │       │       │
│   │   │                     │       │       │
│   │   └─────────────────────┘       │       │
│   │                                 │       │
│   └─────────────────────────────────┘       │
│                                             │
└─────────────────────────────────────────────┘
```

### Information at Each Level

**Near Field (0-1000 units)**
- All individual particles
- Spring connections and stresses
- Composite internal structure
- Full temperature/radiation data
- All collision events
- Complete physics state

**Medium Field (1000-10,000 units)**
- Composites as single objects (no internal structure)
- Large individual particles (stars, black holes)
- Particle clouds as density fields
- Major events only (explosions, formations)
- Simplified physics (position, velocity, mass)

**Far Field (10,000+ units)**
- Statistical aggregates only
- "Galaxy with 1M solar masses at position X"
- "Hot region with average temp 5000K"
- Catastrophic events (supernovae, galaxy collisions)
- No individual objects unless massive

## Composite-Aware Simplification

Composites are perfect for hierarchical detail:

### Close Range
- Send all particles forming the composite
- Send all springs with their stress levels
- Send internal temperature distribution
- Full structural integrity data

### Medium Range
- Send composite as single object
- Center of mass, total mass, bounding radius
- Average temperature
- Structural integrity as single value
- Major damage events

### Long Range
- Include in statistical aggregate
- Or omit if below mass threshold
- Only track if very massive

## Query System

### Spatial Queries

The physics engine provides efficient spatial queries:

**Multi-Resolution Query**
- Client specifies observation point and detail zones
- Server returns appropriate representation for each zone
- Composites automatically simplified based on distance

**Bounding Box Query**
- Returns everything within a box
- Used for focused observations (telescope view)
- Can specify maximum detail level

**Statistical Query**
- Returns aggregate properties of large regions
- Total mass, average temperature, velocity field
- Number of objects by type

### Event Subscription

Clients subscribe to events within regions:

**Spatial Filtering**
- Only receive events within area of interest
- Different regions can have different detail levels

**Type Filtering**
- Subscribe to specific event types
- e.g., only black hole formations, only collisions above certain energy

**Magnitude Filtering**
- Only events above significance threshold
- Prevents spam from tiny collisions

## Network Protocol Concepts

### Initial Connection
1. Client connects and authenticates
2. Specifies observation point (controlled ship or camera)
3. Defines view parameters (zone radii, detail levels)
4. Receives initial world snapshot

### Continuous Updates
1. Delta updates for changes in view
2. Event stream for happenings
3. Periodic full refresh for synchronization
4. Prediction/interpolation hints

### Bandwidth Management

**Adaptive Detail**
- Reduce detail if bandwidth limited
- Prioritize important events
- Client can request quality level

**Delta Compression**
- Only send what changed
- Reference frame updates
- Predictive pre-loading

## Player Interaction

### Control Authority

Players can:
- **Direct Control**: Apply forces to owned composites
- **Spawn Limited Objects**: Projectiles, probes (with resource limits)
- **Trigger Events**: Fire engines, activate devices
- **Request Observations**: Point sensors, zoom views

Players cannot:
- Directly modify physics state
- Create massive objects without resources
- Delete other players' objects
- Change simulation parameters

### Action Validation

All player actions are validated:
1. Client sends action request
2. Server validates permissions and physics
3. Server applies action if valid
4. Results propagate through normal event system

## Bot and AI Integration

Bots are just clients that happen to be AI:

**Same Interface**
- Connect like human players
- Receive same observation data
- Send same action commands

**Potential Roles**
- NPC spacecraft pilots
- Background civilizations
- Ecosystem maintenance
- Dynamic storytelling
- Market makers in economy

**No Special Treatment**
- Physics engine doesn't distinguish bots from humans
- All follow same rules and limits

## Scalability Considerations

### Large Composite Handling

Problem: A space station might be visible from far away but has thousands of particles

Solution: 
- Multiple representation levels for same composite
- Icon/dot at extreme distance
- Bounding box at medium distance  
- Full detail when close

### Smooth Transitions

Problem: Popping as detail levels change

Solution:
- Overlapping detail zones
- Gradual composite assembly
- Predictive loading of approaching objects

### Multiple Observers

Problem: Many players in same area

Solution:
- Shared query caching
- Multicast for common data
- Region-based servers

## Implementation Phases

### Phase 1: Basic Observation
- Simple distance-based queries
- Fixed detail levels
- Single player

### Phase 2: Composite Simplification
- Smart composite handling
- Smooth LOD transitions
- Multiple players

### Phase 3: Advanced Features
- Predictive loading
- Bandwidth adaptation
- Bot integration

### Phase 4: Massive Scale
- Region servers
- Load balancing
- Persistent world

## Security and Fair Play

### Information Hiding
- Clients only receive what they can "see"
- No perfect information about distant objects
- Fog of war for unexplored regions

### Action Limits
- Rate limiting on spawning
- Resource costs for actions
- Cooldowns on powerful abilities

### Authoritative Server
- All physics calculated server-side
- Client prediction for responsiveness
- Server correction for discrepancies

## Summary

The observation system enables millions of particles to be simulated while clients receive manageable, relevant views. Composites provide natural hierarchy for simplification. The physics engine remains pure, providing queries and events, while network layer handles distribution and compression.

Key insight: **The same composite can be a single dot to distant observers, a rigid body to medium observers, and thousands of particles to nearby observers** - all simultaneously, all consistent, all emerging from the same simulation.
# DigiStar DSL Design

## Vision

A unified Domain-Specific Language that serves as the complete interface to DigiStar - for creation, interaction, analysis, and control of particle simulations.

## Core Philosophy

The DSL is not just for system generation, it's a **simulation operating system**:
- **Creation**: Define systems, bodies, forces
- **Interaction**: Query, modify, control in real-time
- **Analysis**: Measure, observe, extract information
- **Behavior**: Define rules, triggers, AI agents
- **Evolution**: Control time, branch realities

## Related Documentation

- [DSL Language Specification](DSL_LANGUAGE_SPEC.md) - Detailed syntax and semantics
- [DSL Advanced Features](DSL_ADVANCED_FEATURES.md) - Command queue, events, AI control
- [Event System Architecture](EVENT_SYSTEM_ARCHITECTURE.md) - High-performance event handling
- [Emergent Composite System](EMERGENT_COMPOSITE_SYSTEM.md) - Spring networks and composites

## Architectural Design

### Command Queue Pattern

The DSL uses a **command queue architecture** to enable safe concurrent script execution while maintaining deterministic physics simulation.

#### Why Command Queue?

1. **Thread Safety**: Scripts can run in parallel without locks on simulation state
2. **Determinism**: All mutations happen atomically between physics frames
3. **Performance**: Scripts don't block physics updates
4. **Scalability**: Can use 1 to N threads for script execution
5. **Natural Execution**: No need for complex continuation/resumption

#### How It Works

```
Frame N:
├── Physics Update (main thread)
├── Render (if needed)
└── Script Execution (worker threads)
    ├── Script A reads state → generates commands
    ├── Script B reads state → generates commands
    └── Script C reads state → generates commands
    
Frame Boundary:
└── Apply all commands atomically

Frame N+1:
└── Repeat...
```

#### Command Examples

Instead of direct mutation:
```lisp
; OLD: Direct mutation (not thread-safe)
(set! particle.velocity [100 0])
```

Scripts generate commands:
```lisp
; NEW: Command generation (thread-safe)
(set-velocity particle-id [100 0])
; → Generates: SetVelocityCommand{id: 42, vel: [100, 0]}
```

#### Thread Pool Configuration

```lisp
; Configure thread pool (default: 1 thread for CPU-only systems)
(set-script-threads 1)  ; Single thread for scripts
(set-script-threads 4)  ; Use 4 threads
(set-script-threads 0)  ; Auto-detect (hardware_concurrency - 1)
```

### Script Execution Model

Scripts run in three modes:

1. **Immediate**: One-shot commands
   ```lisp
   (particle :mass 10 :pos [0 0])  ; Generates CreateParticle command
   ```

2. **Reactive**: Event-driven
   ```lisp
   (on 'collision (lambda (p1 p2 energy)
     (when (> energy 1000)
       (explode p1))))  ; Queues explosion command when triggered
   ```

3. **Continuous**: Ongoing behaviors
   ```lisp
   (behavior 'orbit-controller
     (lambda ()
       (let ([pos (query-position 'satellite)])
         (apply-force 'satellite (calculate-correction pos)))))
   ```

### Read-Write Separation

- **Reads**: Always safe, see consistent snapshot
  ```lisp
  (query-position p)     ; Safe parallel read
  (measure-distance a b) ; Safe parallel read
  ```

- **Writes**: Generate commands for later application
  ```lisp
  (create-particle ...)  ; → CreateParticleCommand
  (destroy p)           ; → DestroyCommand
  (apply-force p f)     ; → ApplyForceCommand
  ```

## Design Principles

1. **Composable**: Build complex systems from simple primitives
2. **Declarative**: Describe what you want, not how to build it
3. **Unified**: Same language for all operations
4. **Physically Grounded**: Units and constants built-in
5. **Human-Readable**: S-expression syntax (Lisp-inspired)

## Core Language Elements

### Basic Particles

```lisp
; Single particle
(particle 
  :mass 1.0
  :pos [0 0]
  :vel [1 0]
  :temp 300)

; Particle cloud
(cloud
  :n 10000
  :distribution (gaussian :center [0 0] :sigma 100)
  :mass-dist (power-law :min 0.1 :max 10 :alpha -2.35)
  :temp 10)
```

### Orbital Systems

```lisp
; Solar system
(orbital-system
  :center (star :mass 1.989e30 :temp 5778)
  :bodies [
    (orbit :a 1.0 :e 0.017 
      :body (planet :mass 5.97e24 :radius 6.371e6))
    (orbit :a 5.2 :e 0.048 
      :body (gas-giant :mass 1.898e27))])
```

### Composite Bodies

```lisp
; Spaceship with springs
(define-body spaceship
  :structure (spring-mesh
    :shape (load-obj "ship.obj")
    :particles 5000
    :stiffness 10000
    :material 'titanium)
  :components [
    (engine :thrust 1e6 :fuel 1000)
    (reactor :power 1e9)])
```

### Forces and Fields

```lisp
; System with multiple forces
(with-forces [gravity radiation springs]
  (system
    :particles (cloud :n 100000)
    :temperature 10
    :periodic-boundary true))
```

## Interactive Operations

### Real-Time Queries

```lisp
; Find hot, massive objects
(query
  (select :particles
    (where :mass > 1e24)
    (where :temp > 5000))
  (return [:id :pos :vel]))

; Measure properties
(measure
  (center-of-mass (within-radius [0 0] 1000))
  (temperature (region :box [[0 0] [100 100]])))
```

### Modification Commands

```lisp
; Apply thrust to ship
(update
  (body :id 'ship-001)
  (set :thrust [100 0])
  (consume :fuel 0.1))

; Trigger explosion
(at (position [100 200])
  (explode :energy 1e15 :fragments 100))
```

### Behavioral Rules

```lisp
; Collision creates debris
(rule :collision-debris
  (when (collision-energy :a :b) > 1e6
    (fragment :smaller :n (random 10 100))
    (emit-event :type 'explosion)))

; Star formation
(rule :star-formation
  (when (and (density :cloud > 1e6)
             (temperature :cloud < 100))
    (collapse :cloud :into 'star)))
```

## Time and State Control

```lisp
; Time manipulation
(time-control
  (set-speed 100)    ; 100x faster
  (pause)
  (reverse)          ; Run backwards
  (jump-to :t 1000))

; Checkpointing
(checkpoint :name 'before-collision)
(rollback :to 'before-collision)

; Branch realities
(branch
  :from current-state
  :variations [(vary :gravity 1.1)
                (vary :temperature 0.5)])
```

## Analysis and Observation

```lisp
; Watch variables
(watch
  :every 0.1
  :log [(distance 'earth 'moon)
        (fuel-remaining 'ship)
        (structural-integrity 'station)])

; Pattern detection
(detect
  :patterns ['resonance 'spiral-arms 'accretion]
  :in (region :radius 10000))
```

## Complete Example: Interactive Session

```lisp
(session "explore-galaxy"
  ; Create galaxy
  (galaxy
    :type 'spiral
    :stars 1e6
    :center (black-hole :mass 4e6))
  
  ; Add player ship
  (spawn
    (spaceship :at [1000 0] :player true))
  
  ; Set up controls
  (controls
    :key-w (thrust :forward 100)
    :key-s (thrust :backward 100)
    :mouse-click (select-nearest)
    :key-space (fire-weapon))
  
  ; Define objectives
  (objectives
    (reach :center :distance 100)
    (survive :time 1000))
  
  ; Run simulation
  (run
    :dt 0.01
    :render true
    :interactive true))
```

## Implementation Phases

### Phase 1: Core Parser & Creation
- S-expression parser
- Basic particle/system creation
- Simple force definitions

### Phase 2: Query & Modification  
- Real-time queries
- State modification
- Event system

### Phase 3: Advanced Features
- Behavioral rules
- Time control
- Pattern detection

### Phase 4: Full Integration
- Python bindings
- Network protocol
- AI agent framework

## Benefits

1. **Unified Interface**: One language for everything
2. **Rapid Prototyping**: Test ideas quickly
3. **Reproducibility**: Share simulations as code
4. **Extensibility**: Easy to add new features
5. **Educational**: Learn physics through interaction

## Technical Notes

The DSL compiles to:
- Particle creation commands
- Force field definitions
- Event triggers
- Query operations

All operations map to the underlying C++ engine through a clean API. The DSL is the primary interface, but not the only one - direct C++ and Python bindings also available.

## Summary

The DSL transforms DigiStar from a particle simulator into a complete simulation environment where users can create universes, interact with them, analyze emergent phenomena, and even branch realities to explore "what if" scenarios. It's not just a configuration language - it's a physics playground.
# DSL Advanced Features

## Command Queue Architecture

### Overview
The DSL uses a command queue pattern to enable safe concurrent script execution while maintaining deterministic physics simulation.

### Benefits
1. **Thread Safety**: Scripts run in parallel without locks on simulation state
2. **Determinism**: All mutations happen atomically between physics frames
3. **Performance**: Scripts don't block physics updates
4. **Scalability**: Use 1 to N threads for script execution
5. **Natural Execution**: No complex continuation/resumption needed

### Execution Model
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

### Command Examples
```lisp
; Instead of direct mutation:
; (set! particle.velocity [100 0])  ; Not thread-safe

; Scripts generate commands:
(set-velocity particle-id [100 0])
; → Generates: SetVelocityCommand{id: 42, vel: [100, 0]}
```

## Thread Pool Configuration

```lisp
; Configure thread pool for script execution
(set-script-threads 1)   ; Single thread (good for CPU-only systems)
(set-script-threads 4)   ; Use 4 threads
(set-script-threads 0)   ; Auto-detect (hardware_concurrency - 1)

; Query thread configuration
(get-script-threads)     ; Returns current thread count
(get-max-threads)        ; Returns hardware thread count
```

## Event System Integration

### Event Subscription
```lisp
; Channel-based events
(on-event "physics/collision"
  (lambda (event)
    (when (> (event-field event 'energy) 1000)
      (explode (event-field event 'pos)))))

; Entity-specific observation
(define my-ship (find-ship "USS Enterprise"))
(observe my-ship
  (lambda (event)
    (case (event-type event)
      ['damage (repair-ship my-ship)]
      ['fuel-low (find-fuel-depot my-ship)])))

; Property watching
(watch-property my-ship 'structural-integrity
  (lambda (old new)
    (when (< new 0.5)
      (evacuate-ship my-ship))))
```

### Event Emission
```lisp
; Emit custom events
(emit-event "game/objective-complete" 
  :objective 'destroy-asteroid
  :time (sim-time)
  :score 1000)

; Emit with spatial information
(emit-spatial-event "explosion"
  :pos [100 200 300]
  :radius 50
  :energy 1e10)
```

### Event Aggregation
```lisp
; Handle high-frequency events efficiently
(on-event-aggregate "physics/spring/*"
  (lambda (stats)
    (when (> (stats 'break-count) 100)
      (structural-failure (stats 'center))))
  :window 0.1      ; 100ms aggregation window
  :min-count 10)   ; Only fire if 10+ events
```

## Composite Discovery and Control

### Monitoring Emergent Structures
```lisp
; Watch for interesting composites
(on-event "composite/formed"
  (lambda (event)
    (let ([comp (event-field event 'id)])
      (when (planet-like? comp)
        (adopt-planet comp)))))

; Adopt and enhance composite
(define (adopt-planet comp-id)
  (name-composite comp-id (generate-name))
  (strengthen-springs comp-id :factor 10)
  (add-atmosphere comp-id)
  (add-geological-features comp-id)
  (observe comp-id planet-controller))
```

### Procedural Generation
```lisp
; Generate complex structures
(define (create-space-station name pos)
  (let ([structure (generate-spring-mesh
                     :template 'geodesic-sphere
                     :radius 1000
                     :particles 10000
                     :stiffness 1e7)])
    
    ; Add modules
    (attach-module structure (create-docking-bay) :pos [100 0 0])
    (attach-module structure (create-reactor) :pos [0 0 0])
    (attach-module structure (create-habitat) :pos [-100 0 0])
    
    ; Spawn in world
    (spawn-composite structure :pos pos)
    (name-composite structure name)
    
    structure))

; Generate from templates
(define ship (generate-from-template 'fighter
                :scale 10
                :material 'titanium))
```

## AI and Behavioral Control

### Subsumption Architecture
```lisp
(define (create-ship-ai ship)
  (make-subsumption-controller
    ; Survival layer (highest priority)
    (layer 'survive
      :condition (lambda () (< (hull-integrity ship) 0.3))
      :action (lambda () (flee-to-safety ship)))
    
    ; Mission layer
    (layer 'mission
      :condition (lambda () (has-objective? ship))
      :action (lambda () (pursue-objective ship)))
    
    ; Exploration layer (default)
    (layer 'explore
      :condition (lambda () #t)
      :action (lambda () (explore-unknown ship)))))
```

### Behavior Trees
```lisp
(define patrol-behavior
  (behavior-tree
    (sequence
      (move-to waypoint-1)
      (wait 5)
      (move-to waypoint-2)
      (wait 5)
      (selector
        (if-enemy-detected engage-combat)
        (return-to-base)))))
```

## Orbital Mechanics

### Restorative Force Control
```lisp
; Create stable orbits using restorative forces
(define (stabilize-orbit object center)
  (apply-restorative-force object
    :center center
    :beta 0.01      ; Restoration strength
    :alpha 0.001    ; Distance decay
    :duration 100)) ; Apply for 100 frames

; Create orbital system
(define (generate-planetary-system)
  ; Enable temporary restorative force
  (with-restorative-field 
    :beta 0.01 
    :alpha 0.001
    :duration 1000
    
    ; Create and scatter objects
    (let ([star (create-star :mass 1e30)]
          [planets (map create-planet planet-specs)])
      
      ; Random initial positions/velocities
      (scatter-randomly planets :radius 1e9)
      
      ; Let restorative force organize them
      (run-simulation :time 1000)
      
      ; System now has stable orbits
      (make-system star planets))))
```

### Curl Forces for Complex Dynamics
```lisp
; Add swirling motion to debris field
(apply-curl-force
  :region (sphere-region center 1000)
  :vector-potential (lambda (pos)
                     (* 0.001 (z pos) 
                        [(- (y pos)) (x pos) 0]))
  :duration 'infinite)
```

## Time Control and Branching

### Simulation Control
```lisp
; Time manipulation
(set-simulation-speed 10)    ; 10x faster
(pause-simulation)
(resume-simulation)
(step-simulation 1)          ; Single frame

; Checkpointing
(define checkpoint (save-state))
(restore-state checkpoint)

; Branching realities
(branch-simulation
  :variations 
  [(lambda () (set-gravity 0.5))
   (lambda () (set-gravity 2.0))
   (lambda () (disable-force 'springs))]
  :duration 1000
  :compare-function analyze-divergence)
```

## Query and Analysis

### Spatial Queries
```lisp
; Find objects in region
(find-in-region 
  :center [0 0 0]
  :radius 1000
  :filter (lambda (p) (> (mass p) 100)))

; Nearest neighbor queries
(find-nearest
  :pos [100 200 300]
  :type 'star
  :max-distance 1e6)

; Ray queries
(raycast
  :origin [0 0 0]
  :direction [1 0 0]
  :max-distance 1000)
```

### Statistical Analysis
```lisp
; Measure properties
(measure
  :region (box-region corner1 corner2)
  :properties ['temperature 'density 'pressure])

; Track statistics over time
(track-statistic
  :name "total-energy"
  :function (lambda () (total-system-energy))
  :interval 0.1
  :history-size 1000)

; Pattern detection
(detect-patterns
  :types ['spiral 'cluster 'ring]
  :region (sphere-region center 10000))
```

## Performance Optimization

### Batch Operations
```lisp
; Batch particle creation
(create-particles
  :specs [(particle-spec :mass 1 :pos [0 0 0])
          (particle-spec :mass 2 :pos [1 0 0])
          ...])  ; Single command for many particles

; Batch modifications
(modify-particles
  :filter (lambda (p) (> (temp p) 1000))
  :action (lambda (p) (set-temp p 900)))
```

### Level-of-Detail Control
```lisp
; Reduce detail for distant objects
(set-lod-policy
  :distance-thresholds [100 1000 10000]
  :spring-update-rates [1 0.1 0.01]
  :collision-modes ['exact 'approximate 'none])
```

## Script Lifecycle Management

### Persistent Scripts
```lisp
; Create long-running script
(define-persistent-script "ecosystem-manager"
  :init (lambda () (setup-ecosystem))
  :update (lambda (dt) (update-ecosystem dt))
  :cleanup (lambda () (save-ecosystem-state)))

; Script control
(pause-script "ecosystem-manager")
(resume-script "ecosystem-manager")
(terminate-script "ecosystem-manager")
```

### Script Communication
```lisp
; Inter-script messaging
(send-message "ship-controller" 
  :command 'attack
  :target enemy-id)

(on-message 
  (lambda (msg)
    (case (msg 'command)
      ['attack (engage-target (msg 'target))]
      ['retreat (flee-to-base)])))
```

## Integration with External Systems

### Python Bridge
```lisp
; Call Python functions
(python-eval "import numpy as np")
(define result (python-call "np.fft.fft" signal-data))

; Register Python callbacks
(python-register "on_collision" 
  (lambda (p1 p2) 
    (python-call "ml_model.predict" p1 p2)))
```

### Data Export
```lisp
; Export simulation data
(export-data
  :format 'hdf5
  :file "simulation.h5"
  :include ['particles 'springs 'composites])

; Stream data to external process
(open-data-stream
  :protocol 'msgpack
  :endpoint "tcp://localhost:5555"
  :rate 30)  ; 30 Hz
```

## Future Features (Planned)

1. **Visual Scripting**: Node-based script editor
2. **Script Debugging**: Breakpoints, step-through, inspection
3. **Script Optimization**: JIT compilation for hot paths
4. **Distributed Scripts**: Run scripts across multiple machines
5. **Machine Learning**: Train neural networks from simulation data
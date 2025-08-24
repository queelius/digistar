# Emergent Composite System

## Overview

DigiStar's composite system allows complex structures to emerge from simple spring networks, be discovered by scripts, enhanced with additional features, and controlled by AI. Composites are identified through union-find on spring networks, creating ephemeral structures that scripts can adopt and name.

## Composite Lifecycle

### 1. Emergence (Physics-Driven)

Spring networks form automatically based on proximity and relative velocity:

```cpp
class SpringField {
    // Automatic spring formation between nearby bodies
    void updateSprings() {
        for (auto& [p1, p2] : nearby_pairs) {
            if (distance(p1, p2) < spring_threshold &&
                relativeVelocity(p1, p2) < velocity_threshold) {
                createSpring(p1, p2);
            }
        }
    }
};
```

### 2. Discovery (Union-Find)

Connected components in spring network identified each frame:

```cpp
class CompositeManager {
    UnionFind spring_network;
    
    void updateComposites() {
        auto components = spring_network.getComponents();
        
        for (auto& comp : components) {
            if (isNewComposite(comp.id)) {
                event_system.emit("composite/formed", {
                    comp.id, comp.particles, comp.springs,
                    comp.mass, comp.radius, comp.center
                });
            }
        }
    }
};
```

### 3. Adoption (Script-Driven)

Scripts monitor for interesting composites and adopt them:

```lisp
(on-event "composite/formed"
  (lambda (event)
    (let ([comp-id (event-field event 'id)]
          [mass (event-field event 'mass)]
          [density (calculate-density event)])
      
      ; Check if it meets criteria
      (when (planet-like? mass density)
        ; Adopt and enhance
        (name-composite comp-id (generate-planet-name))
        (strengthen-springs comp-id :factor 10)
        (add-geological-features comp-id)
        (observe comp-id planet-controller)))))
```

## Composite Properties

### Emergent Properties
Calculated from constituent particles:
- Center of mass
- Total mass and momentum
- Angular momentum
- Moment of inertia tensor
- Structural integrity (spring stress distribution)
- Temperature (average kinetic energy)

### Enhanced Properties
Added by scripts after adoption:
- Name and tags
- Strengthened springs
- Additional springs for features
- Behavioral controllers
- Custom rendering hints

## Spring Network Generators

### Procedural Generation Functions

```lisp
; Generate ship hull with internal structure
(define (generate-spaceship)
  (let ([hull (generate-spring-mesh
                :template "ship-hull"
                :particles 5000
                :spring-stiffness 1e6)]
        [engine (generate-component
                  :type 'engine
                  :thrust 1e9)]
        [reactor (generate-component
                   :type 'reactor
                   :power 1e12)])
    
    ; Connect components
    (attach-component hull engine :pos [0 -10 0])
    (attach-component hull reactor :pos [0 0 0])
    
    ; Strengthen critical connections
    (reinforce-springs hull :factor 5 :region 'joints)
    
    hull))

; Generate planet with layers
(define (generate-planet mass radius)
  (let* ([core-r (* 0.3 radius)]
         [mantle-r (* 0.7 radius)]
         [core (generate-sphere
                 :radius core-r
                 :density 8000
                 :springs 'crystalline)]
         [mantle (generate-shell
                   :inner core-r
                   :outer mantle-r
                   :density 4500
                   :springs 'viscous)]
         [crust (generate-shell
                  :inner mantle-r
                  :outer radius
                  :density 2800
                  :springs 'brittle)])
    
    ; Connect layers
    (connect-layers core mantle :stiffness 1e7)
    (connect-layers mantle crust :stiffness 1e6)
    
    ; Add surface features
    (add-tectonic-plates crust :count (random 7 15))
    (add-mountains crust :count (random 10 30))
    
    (make-composite core mantle crust)))
```

### Template-Based Generation

```lisp
; Spring mesh templates
(define-template 'crystalline
  :pattern 'face-centered-cubic
  :bond-angles [90 180]
  :coordination 12)

(define-template 'organic
  :pattern 'voronoi
  :randomness 0.3
  :flexibility 'high)

(define-template 'ship-hull
  :pattern 'geodesic
  :subdivisions 4
  :reinforcement 'stress-aligned)

; Use templates
(generate-from-template 'crystalline 
  :size 1000 
  :stiffness 1e5)
```

## Composite Enhancement

### Geological Features
```lisp
(define (add-mountains composite count)
  (repeat count
    (let ([pos (random-surface-point composite)]
          [height (random 1000 5000)]
          [radius (random 500 2000)])
      
      ; Create mountain as local spring cluster
      (add-spring-cluster composite
        :center pos
        :shape 'cone
        :height height
        :radius radius
        :stiffness 1e7))))

(define (add-ocean composite depth)
  (let ([ocean-particles (generate-fluid-particles
                           :volume (ocean-volume composite depth)
                           :density 1000)])
    
    ; Add as loosely coupled layer
    (add-particles composite ocean-particles)
    (couple-with-springs composite ocean-particles
      :stiffness 1e3  ; Weak coupling
      :damping 0.9)))  ; High damping for fluid
```

### Structural Reinforcement
```lisp
(define (strengthen-composite comp factor)
  ; Increase spring stiffness
  (for-each-spring comp
    (lambda (spring)
      (set-stiffness spring (* (get-stiffness spring) factor))))
  
  ; Add redundant springs for critical paths
  (let ([stress-paths (analyze-stress-paths comp)])
    (for-each (lambda (path)
                (add-redundant-springs path :count 3))
              stress-paths)))
```

## AI Controllers

### Subsumption Architecture for Ships
```lisp
(define (ship-ai-controller ship-name)
  (make-subsumption-controller
    ; Layer 0: Survival (highest priority)
    (layer 'survive
      :condition (lambda () (< (hull-integrity ship) 0.3))
      :action (lambda () (flee-to-safety ship)))
    
    ; Layer 1: Refuel
    (layer 'refuel
      :condition (lambda () (< (fuel ship) 0.2))
      :action (lambda () (find-fuel-depot ship)))
    
    ; Layer 2: Combat
    (layer 'combat
      :condition (lambda () (enemies-nearby? ship))
      :action (lambda () (engage-combat ship)))
    
    ; Layer 3: Mission
    (layer 'mission
      :condition (lambda () (has-objective? ship))
      :action (lambda () (pursue-objective ship)))
    
    ; Layer 4: Explore (default)
    (layer 'explore
      :condition (lambda () #t)
      :action (lambda () (explore-sector ship)))))
```

### Planet Evolution Controller
```lisp
(define (planet-evolution-controller planet)
  ; Geological evolution
  (every 100  ; Every 100 time units
    (lambda ()
      ; Tectonic movement
      (move-tectonic-plates planet)
      
      ; Volcanic activity
      (when (> (internal-pressure planet) threshold)
        (trigger-volcano planet (highest-pressure-point planet)))
      
      ; Erosion
      (apply-erosion planet :rate 0.001)))
  
  ; Atmospheric evolution
  (when (has-atmosphere? planet)
    (every 10
      (lambda ()
        (update-atmosphere planet)
        (apply-weather-patterns planet)))))
```

## Composite Merging and Splitting

### Merger Events
```lisp
(on-event "composite/collision"
  (lambda (event)
    (let ([comp1 (event-field event 'composite1)]
          [comp2 (event-field event 'composite2)]
          [energy (event-field event 'impact-energy)])
      
      (cond
        ; High energy: both shatter
        [(> energy 1e15)
         (shatter comp1 :fragments (random 10 50))
         (shatter comp2 :fragments (random 10 50))]
        
        ; Medium energy: deformation and merger
        [(> energy 1e12)
         (deform-composite comp1 :energy energy)
         (deform-composite comp2 :energy energy)
         (merge-composites comp1 comp2)]
        
        ; Low energy: elastic bounce
        [else
         (apply-impulse comp1 (collision-normal event))
         (apply-impulse comp2 (- (collision-normal event)))]))))
```

### Splitting Events
```lisp
(define (split-composite comp plane)
  (let* ([particles (get-particles comp)]
         [side1 (filter (lambda (p) (> (dot p plane) 0)) particles)]
         [side2 (filter (lambda (p) (<= (dot p plane) 0)) particles)])
    
    ; Create two new composites
    (let ([comp1 (make-composite side1)]
          [comp2 (make-composite side2)])
      
      ; Sever springs crossing the plane
      (for-each-spring comp
        (lambda (spring)
          (when (crosses-plane? spring plane)
            (break-spring spring))))
      
      ; Emit split event
      (emit-event 'composite/split 
        :original comp
        :fragments (list comp1 comp2))
      
      (list comp1 comp2))))
```

## Orbital System Management

### Using Restorative Forces
```lisp
(define (create-star-system star-mass planets)
  ; Enable restorative force for initial setup
  (set-orbital-restoration :beta 0.01 :alpha 0.001)
  
  ; Create star
  (define star (create-star :mass star-mass))
  
  ; Scatter planets randomly
  (define planet-list
    (map (lambda (spec)
           (particle :mass (spec 'mass)
                    :pos (random-sphere-point (spec 'orbit-radius))
                    :vel (random-vector 100)))
         planets))
  
  ; Let restorative force organize into stable orbits
  (run-simulation :time 1000)
  
  ; Disable restorative force
  (set-orbital-restoration :beta 0)
  
  ; Create named composite for entire system
  (name-composite-group 
    (cons star planet-list)
    "star-system"))
```

## Performance Considerations

### Spatial Indexing for Spring Formation
- Use octree/grid for O(n log n) nearest neighbor search
- Only check particles within spring formation radius
- Update spatial index incrementally

### Union-Find Optimization
- Path compression for O(α(n)) operations
- Incremental updates when springs break/form
- Cache component properties between frames

### LOD for Distant Composites
- Reduce spring update frequency for distant composites
- Aggregate small composites into single bodies
- Use simplified collision geometry

## Future Enhancements

1. **Composite Templates Library**: Pre-designed structures for common objects
2. **Genetic Algorithms**: Evolve composite designs for specific purposes
3. **Composite Serialization**: Save/load complex structures
4. **Procedural Texturing**: Generate visual details based on composition
5. **Composite AI Training**: Machine learning for composite controllers
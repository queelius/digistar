# DigiStar DSL Specification v2.0

## Overview

The DigiStar Domain-Specific Language (DSL) is a Lisp-inspired language designed for controlling and scripting the DigiStar space sandbox simulation. It provides powerful primitives for particle generation, physics control, event handling, and reactive programming.

## Language Fundamentals

### Syntax

The DSL uses S-expressions (symbolic expressions) as its fundamental syntax:

```lisp
(function-name arg1 arg2 ... argN)
```

### Data Types

| Type | Examples | Description |
|------|----------|-------------|
| **Number** | `42`, `3.14`, `1.5e6` | Integer or floating-point |
| **String** | `"hello"`, `'world` | Text data |
| **Symbol** | `particle`, `collision` | Identifiers |
| **Boolean** | `true`, `false`, `#t`, `#f` | Truth values |
| **Vector** | `[1 2 3]`, `[x y z]` | Position/velocity vectors |
| **List** | `(1 2 3)`, `(a b c)` | Collections |
| **Function** | `(lambda (x) (* x 2))` | Anonymous functions |
| **Pattern** | `(collision ?p1 ?p2 _)` | Pattern for matching |

### Variables and Binding

```lisp
; Define global variable
(define gravity-constant 6.67e-11)

; Local binding
(let ([x 10]
      [y 20])
  (+ x y))

; Sequential binding
(let* ([x 10]
       [y (* x 2)])
  y)  ; Returns 20
```

## Core Language Features

### 1. Pattern Matching

Pattern matching enables elegant destructuring and conditional logic:

```lisp
(match value
  ; Literal pattern
  [42 "The answer"]
  
  ; Variable binding pattern
  [?x (format "Got value: ~a" x)]
  
  ; List pattern with rest
  [(list ?head . ?tail) 
   (process-list head tail)]
  
  ; Guard pattern
  [(? number? ?n (> n 100))
   "Large number"]
  
  ; Wildcard pattern
  [_ "Default case"])
```

#### Pattern Types

| Pattern | Syntax | Description |
|---------|--------|-------------|
| **Literal** | `42`, `"hello"` | Exact match |
| **Variable** | `?name` | Binds to any value |
| **Wildcard** | `_` | Matches anything, no binding |
| **Type** | `(? predicate? ?var)` | Type checking |
| **List** | `(list p1 p2 ...)` | List destructuring |
| **Cons** | `(p1 . p2)` | Pair destructuring |
| **Guard** | `(guard pattern condition)` | Conditional match |
| **Or** | `(or p1 p2 ...)` | Alternative patterns |
| **And** | `(and p1 p2 ...)` | All must match |

### 2. Procedural Generation

Built-in primitives for mass particle creation:

```lisp
; Basic generation
(generate type
  :particles count
  :pattern pattern-spec
  :distribution dist-spec
  :region region-spec
  :properties property-spec)
```

#### Distribution Functions

```lisp
; Uniform distribution
(uniform min max)

; Gaussian/normal distribution
(gaussian mean stddev)

; Power law distribution
(power-law min max exponent)

; Salpeter IMF for stellar masses
(salpeter-imf min-mass max-mass)

; Maxwell-Boltzmann for velocities
(maxwell-boltzmann temperature)
```

#### Spatial Patterns

```lisp
; Grid pattern
(grid :spacing 10 :dimensions [100 100])

; Hexagonal pattern
(hexagonal :spacing 5 :layers 10)

; Spiral pattern
(spiral :arms 4 :turns 5 :radius 1000)

; Fractal pattern
(fractal :depth 5 :branching 3)

; Disk/ring pattern
(disk :inner-radius 100 :outer-radius 500)
```

#### Example Generators

```lisp
; Generate a galaxy
(generate 'galaxy
  :particles 1000000
  :pattern (spiral :arms 4 :turns 3)
  :distribution (salpeter-imf 0.1 100)
  :region (disk :radius 10000)
  :properties
  [(velocity (keplerian-profile))
   (temperature (range 10 100))])

; Generate asteroid belt
(generate 'asteroid-belt
  :particles 50000
  :pattern (torus :major-radius 500 :minor-radius 50)
  :distribution (power-law 0.1 100 -2.5)
  :properties
  [(composition (choice 'rock 'metal 'ice))
   (rotation (uniform 0 360))])

; Generate molecular cloud
(generate 'molecular-cloud
  :particles 100000
  :pattern (fractal :depth 4)
  :distribution (gaussian 1 0.5)
  :region (sphere :radius 1000)
  :properties
  [(temperature (range 10 50))
   (density (exponential 1e-3))])
```

### 3. Event System Integration

#### Event Handlers

```lisp
; Simple event handler
(on-event event-type handler-function)

; Pattern-based event handler
(on-event 'collision
  (match event
    [(collision ?p1 ?p2 (> ?energy 1000))
     (create-explosion p1 p2 energy)]
    [(collision ?p1 ?p2 _)
     (create-sparks p1 p2)]))

; Filtered event handler
(on-event 'spring-break
  :filter (lambda (e) (> (event-magnitude e) 100))
  :priority 10
  :handler handle-critical-break)
```

#### Event Emission

```lisp
; Emit custom event
(emit-event event-type
  :data event-data
  :position [x y z]
  :magnitude value)

; Emit with pattern
(emit-event 'custom-explosion
  :particles affected-particles
  :energy total-energy
  :radius blast-radius)
```

#### Reactive Programming

```lisp
; React to property changes
(when-change object property handler)

; Example: Monitor temperature
(when-change particle 'temperature
  (lambda (old new)
    (when (> new 1e7)
      (trigger-fusion particle))))

; Aggregate events
(aggregate-events event-type
  :window time-window
  :combine combiner-function
  :threshold threshold-value
  :action action-function)

; Example: Detect structure collapse
(aggregate-events 'spring-break
  :window 0.1
  :combine count
  :threshold 100
  :action (lambda (n) 
    (emit-event 'structural-collapse 
      :severity n)))
```

### 4. Physics Control

#### Particle Operations

```lisp
; Create particles
(create-particle
  :mass value
  :position [x y z]
  :velocity [vx vy vz]
  :properties {...})

; Batch creation
(create-particles count spec)

; Modify particles
(set-property particle property value)
(apply-force particle force-vector)
(apply-impulse particle impulse-vector)

; Query particles
(find-particles predicate)
(nearest-particles position count)
(particles-in-region region)
```

#### Spring Operations

```lisp
; Create spring
(create-spring particle1 particle2
  :stiffness k
  :damping c
  :rest-length l0)

; Query springs
(springs-of particle)
(springs-between p1 p2)

; Modify springs
(set-spring-property spring property value)
(break-spring spring)
```

#### Composite Bodies

```lisp
; Define composite
(define-composite name
  :particles particle-list
  :springs spring-list
  :properties {...})

; Operations on composites
(rotate-composite composite angle)
(translate-composite composite vector)
(apply-torque composite torque)
```

### 5. Control Flow

```lisp
; Conditional
(if condition then-expr else-expr)
(when condition body...)
(unless condition body...)
(cond
  [condition1 result1]
  [condition2 result2]
  [else default])

; Loops
(for ([i (range 10)])
  (create-particle ...))

(while condition
  body...)

(do ([i 0 (+ i 1)])
    [(>= i 10) result]
  body...)

; Higher-order functions
(map function list)
(filter predicate list)
(reduce function initial list)
(for-each function list)
```

### 6. Time Control

```lisp
; Schedule events
(at time-value action)
(after delay action)
(every interval action)

; Timeline definition
(timeline
  (at 0 (create-galaxy))
  (at 100 (spawn-player))
  (during [200 300] (accelerate-time 10))
  (every 50 (checkpoint)))

; Time manipulation
(set-time-scale factor)
(pause-simulation)
(resume-simulation)
(step-simulation steps)
```

### 7. Queries and Analysis

```lisp
; Spatial queries
(query-region region
  :filter predicate
  :select properties
  :limit count)

; Statistical queries
(compute-statistics particle-set
  :metrics [mass energy momentum]
  :aggregation [sum mean stddev])

; Graph queries (for spring networks)
(connected-components spring-network)
(shortest-path particle1 particle2)
(spanning-tree particles)
```

## Built-in Functions

### Math Functions
- `+`, `-`, `*`, `/`, `mod`, `pow`
- `sin`, `cos`, `tan`, `atan2`
- `sqrt`, `exp`, `log`
- `min`, `max`, `abs`
- `floor`, `ceil`, `round`

### Vector Operations
- `vec2`, `vec3` - Create vectors
- `vec+`, `vec-`, `vec*` - Vector arithmetic
- `dot`, `cross` - Vector products
- `magnitude`, `normalize` - Vector properties
- `distance` - Distance between points

### List Operations
- `list`, `cons`, `car`, `cdr`
- `append`, `reverse`, `length`
- `member`, `remove`, `sort`
- `take`, `drop`, `partition`

### String Operations
- `string-append`, `string-length`
- `substring`, `string-split`
- `format` - String formatting

### Type Predicates
- `number?`, `string?`, `symbol?`
- `list?`, `vector?`, `function?`
- `particle?`, `spring?`, `composite?`

## Bytecode Compilation

The DSL is compiled to bytecode for efficient execution:

### Compilation Process
1. **Parsing**: S-expressions → AST
2. **Analysis**: Type inference, optimization
3. **Compilation**: AST → Bytecode
4. **Execution**: Bytecode → Stack machine

### Optimizations
- Tail call optimization
- Constant folding
- Dead code elimination
- Common subexpression elimination
- Inline expansion for small functions

### Performance Targets
- **Particle generation**: 1M+ particles/second
- **Event processing**: 10K+ events/second
- **Script execution**: < 1ms for typical scripts
- **Memory usage**: < 1MB per script context

## Standard Library

### Prelude Functions
```lisp
; Loaded automatically
(define pi 3.14159265359)
(define e 2.71828182846)
(define c 299792458)  ; Speed of light

(define (square x) (* x x))
(define (cube x) (* x x x))
(define (between? x min max)
  (and (>= x min) (<= x max)))
```

### Physics Constants
```lisp
(define G 6.67430e-11)  ; Gravitational constant
(define k_B 1.380649e-23)  ; Boltzmann constant
(define h 6.62607015e-34)  ; Planck constant
```

### Utility Macros
```lisp
; Time measurement
(define-macro (time expr)
  `(let ([start (current-time)])
     (let ([result ,expr])
       (print (- (current-time) start))
       result)))

; Resource limits
(define-macro (with-limits limits body...)
  `(call-with-limits ,limits
     (lambda () ,@body)))
```

## Error Handling

```lisp
; Try-catch mechanism
(try
  (risky-operation)
  (catch error
    (handle-error error)))

; Assertions
(assert condition message)

; Logging
(log level message)
(debug message)
(warn message)
(error message)
```

## Examples

### Example 1: Binary Star System
```lisp
(define (create-binary-system mass1 mass2 separation)
  (let* ([cm-offset (* separation (/ mass2 (+ mass1 mass2)))]
         [star1 (create-particle 
                  :mass mass1
                  :position (- cm-offset)
                  :velocity [0 (orbital-velocity mass2 separation)])]
         [star2 (create-particle
                  :mass mass2  
                  :position cm-offset
                  :velocity [0 (- (orbital-velocity mass1 separation))])])
    (list star1 star2)))
```

### Example 2: Explosion Handler
```lisp
(on-event 'collision
  (match event
    [(collision ?p1 ?p2 ?energy)
     (when (> energy 1e9)
       (let ([fragments (fragment-particles p1 p2 energy)])
         (for-each create-particle fragments)
         (emit-event 'explosion
           :position (midpoint p1 p2)
           :energy energy)))]))
```

### Example 3: Procedural Asteroid Field
```lisp
(define (create-asteroid-field center radius count)
  (generate 'asteroids
    :particles count
    :pattern (random-sphere)
    :region (sphere center radius)
    :distribution (power-law 1 1000 -2.5)
    :properties
    [(velocity (maxwell-boltzmann 100))
     (rotation (uniform 0 360))
     (composition (weighted-choice
                    ['rock 0.7]
                    ['metal 0.2]
                    ['ice 0.1]))]))
```

## Performance Guidelines

1. **Use batch operations** when creating many particles
2. **Compile frequently-used scripts** to bytecode
3. **Cache pattern matchers** for repeated use
4. **Prefer built-in generators** over manual loops
5. **Use event aggregation** for high-frequency events
6. **Minimize global state** access in hot paths

## Safety and Limits

### Resource Limits
- Maximum particles per script: 10M
- Maximum execution time: 100ms
- Maximum memory per context: 100MB
- Maximum recursion depth: 1000

### Sandboxing
Scripts run in isolated environments with:
- No file system access
- No network access
- Limited CPU time
- Memory quotas
- Safe math (no division by zero)

## Version History

- **v2.0** (Current): Pattern matching, bytecode compilation, procedural generation
- **v1.0**: Basic S-expression evaluation, simple commands

## Future Extensions

- **Type System**: Optional static typing
- **Module System**: Code organization and reuse
- **Debugging Tools**: Breakpoints, stepping, inspection
- **Visual Programming**: Node-based editor
- **GPU Acceleration**: Compute shaders for generators
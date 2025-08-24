# DigiStar DSL Language Specification

## Overview

The DigiStar DSL is a Scheme-inspired language for controlling particle simulations. It follows Scheme's evaluation semantics with lexical scoping and first-class functions.

## Syntax

### Basic Forms

```lisp
; Comments start with semicolon
42                  ; Number
"hello"             ; String  
true, false         ; Booleans
[1.0 2.0 3.0]      ; Vector (3D coordinates)
:keyword           ; Keyword (self-evaluating)
symbol             ; Symbol (variable reference)
(f arg1 arg2)      ; Function application
'expr              ; Quote (literal expression)
```

### Special Forms

```lisp
(if condition then-expr else-expr)    ; Conditional
(define name value)                    ; Define binding in current environment
(set! name value)                      ; Mutate existing binding
(lambda (params...) body...)          ; Create closure
(let ([var val] ...) body...)         ; Local bindings
(begin expr1 expr2 ...)                ; Sequential evaluation
(when condition body...)               ; Conditional without else
(quote expr)                           ; Literal expression
```

## Evaluation Semantics

### Self-Evaluating Forms
- Numbers, strings, booleans, vectors, keywords evaluate to themselves
- Closures evaluate to themselves

### Symbol Evaluation
- Symbols are looked up in the environment chain
- If not found, raises an **unbound variable error**
- Keywords (starting with `:`) are self-evaluating

### List Evaluation
1. Empty list `()` evaluates to `nil`
2. Special forms are handled specially (see below)
3. Otherwise, function application:
   - Evaluate the first element to get function
   - Evaluate remaining elements to get arguments
   - Apply function to arguments

### Environment Model

#### Lexical Scoping
- Each evaluation has an environment
- Environments chain to parent environments
- Variable lookup searches up the chain

#### `define` Semantics
- Creates a NEW binding in the CURRENT environment
- Can shadow bindings in parent environments
- Returns the value being defined

#### `set!` Semantics  
- Searches up environment chain for existing binding
- If found, mutates that binding (even in parent env)
- If not found, raises an error
- Returns the new value

Example:
```lisp
(define x 10)              ; Define x in current env

(define (make-counter)
  (define count 0)         ; Define count in function's env
  (lambda ()
    (set! count (+ count 1))  ; Mutates count in function's env
    count))

(define c1 (make-counter))
(c1)  ; => 1
(c1)  ; => 2
```

### Closures

Lambdas capture their lexical environment:

```lisp
(define (make-adder n)
  (lambda (x)              ; Captures n from outer scope
    (+ x n)))

(define add5 (make-adder 5))
(add5 3)  ; => 8
```

### Error Handling

#### Unbound Variable Error
- Occurs when a symbol is not found in any environment
- Evaluation halts with error message
- Does NOT return nil

#### Arity Error
- Occurs when function called with wrong number of arguments
- Evaluation halts with error message

#### Type Error
- Occurs when operation expects specific type
- Evaluation halts with error message

#### Error Recovery
- Errors stop current evaluation
- In REPL, returns to prompt
- In persistent scripts, script continues but failed evaluation returns nil

## Built-in Functions

### Arithmetic
```lisp
(+ n1 n2 ...)         ; Addition
(- n1 n2 ...)         ; Subtraction  
(* n1 n2 ...)         ; Multiplication
(/ n1 n2 ...)         ; Division
(> n1 n2)             ; Greater than
(< n1 n2)             ; Less than
(= v1 v2)             ; Equality
```

### Particle Creation
```lisp
(particle :mass m :pos [x y] :vel [vx vy] :temp t)
(cloud :n count :center [x y] :radius r)
(star :mass m :pos [x y] :temp t)
```

### Queries
```lisp
(find :center [x y] :radius r)
(measure property particle-list)
(distance p1 p2)
```

### Control
```lisp
(set-velocity particle-id [vx vy])
(apply-force particle-id [fx fy])
(explode particle-id :energy e :fragments n)
```

### Behavioral (Persistent Scripts)
```lisp
(watch :every interval expr)
(rule name condition action)
(trigger condition actions...)
```

## Simulation Interface

### Base Environment Structure

The base environment contains bindings to the simulation state and built-in functions:

```
base_env
├── Built-in Functions (arithmetic, comparisons, etc.)
├── Particle Operations (particle, cloud, star, etc.) 
├── Query Operations (find, measure, distance, etc.)
├── Control Operations (set-velocity, apply-force, etc.)
└── Simulation State References
    ├── *particles* -> ParticlePool reference
    ├── *springs* -> SpringPool reference
    ├── *composites* -> CompositePool reference
    ├── *sim-time* -> Current simulation time
    ├── *dt* -> Timestep
    └── *config* -> PhysicsConfig reference
```

### Direct State Access

Special variables provide direct access to simulation state:

```lisp
*particles*     ; Access to particle pool
*springs*       ; Access to spring pool  
*composites*    ; Access to composite pool
*sim-time*      ; Current simulation time (read-only)
*dt*            ; Current timestep (read-only)
*gravity*       ; Gravity strength (modifiable)
```

### Built-in Function Implementation

Built-in functions generate commands instead of directly modifying state:

```cpp
// Example: particle creation function
builtins["particle"] = [this](args, ctx) {
    // Parse arguments
    auto cmd = std::make_unique<CreateParticleCommand>(mass, pos, vel);
    // Queue command for later execution
    command_queue.push(std::move(cmd));
    // Return future particle ID
    return particle_id_future;
};
```

### State Modification Protocol

1. **Read Access**: Any script can read simulation state (thread-safe snapshot)
2. **Write Access**: All modifications go through command queue
3. **Atomicity**: Commands applied between frames
4. **Thread Safety**: Multiple scripts can run in parallel
5. **Determinism**: Commands applied in deterministic order

### Command Queue Architecture

```
Scripts (parallel) → Commands → Queue → Apply (serial) → Next Frame
```

Benefits:
- No locks needed on simulation state
- Scripts can run continuously without blocking physics
- Natural execution model (no continuations)
- Configurable parallelism (1 to N threads)

### Event Integration

When events are implemented, they will be exposed as:

```lisp
(on-event 'collision 
  (lambda (p1 p2 energy)
    ; Handle collision
    ))

(emit-event 'explosion :pos [x y] :energy e)
```

### Persistent Script Execution

Persistent scripts maintain their own environment but share the base:

```
persistent_script_env
├── parent -> base_env
├── script-local bindings
└── continuation state
```

Each frame:
1. Active scripts are resumed with updated simulation state
2. Scripts can read current state through base environment
3. Modifications happen through built-in functions
4. Scripts yield after time slice

## Implementation Notes

### Nil vs Error
- `nil` is a valid value (empty list)
- Unbound variables are ERRORS, not nil
- Failed operations may return nil OR raise error (documented per operation)

### Memory Management
- All values are reference-counted via shared_ptr
- Environments capture parent via shared_ptr
- Closures capture environment via shared_ptr
- Simulation state is NOT copied, only referenced

### Performance
- Built-in functions implemented in C++
- Direct pointer access to simulation pools
- No serialization/deserialization overhead
- Tail recursion optimization planned
- Persistent scripts run incrementally each frame

### Safety
- DSL cannot corrupt memory (all access through safe interfaces)
- DSL cannot create reference cycles in simulation state
- Resource limits planned (max particles, max time per frame)
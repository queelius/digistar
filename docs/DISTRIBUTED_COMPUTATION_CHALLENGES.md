# Distributed Computation Challenges

## The Dream

We want a collaborative simulation where:
- Anyone can contribute compute power
- The simulation scales with more servers
- Each server might contribute 2-100 million particles worth of compute
- Together we simulate billions of particles
- One coherent, consistent universe emerges

## The Fundamental Problem

**Everything affects everything else.**

In an N-body gravity simulation:
- Every particle pulls on every other particle
- A particle 1 million units away still exerts force
- Particles form springs with neighbors
- Heat flows between touching particles
- Collisions happen at close range

How do you split "everything affects everything" across multiple servers?

## The Artificial Boundary Problem

**Reality has no boundaries. We do.**

Consider a "galaxy" of a million particles:
- Where does the galaxy end? 
- Is the outermost particle still "in" the galaxy?
- What about the particle it's pulling toward it?
- What about the empty space that defines the galaxy's shape?

Any boundary we draw is artificial. Particles don't respect our categories.

## Approaches and Their Problems

### Approach 1: Spatial Partitioning

"Server A handles this region of space"

**Problems:**
- Particles move! They leave regions
- Composite bodies span boundaries
- Springs connect across boundaries
- Empty space wastes server capacity
- Dense regions overload servers

### Approach 2: Particle ID Assignment

"Server A handles particles 0-1M, Server B handles 1M-2M"

**Problems:**
- Nearby particles might be on different servers
- Spring networks get split arbitrarily
- Collision detection requires massive communication
- Composite bodies torn apart

### Approach 3: Dynamic Load Balancing

"Assign particles to balance computational load"

**Problems:**
- Constant reassignment as load changes
- How to measure "load"?
- Migration overhead
- State synchronization complexity

### Approach 4: Force-Type Specialization

"Server A does gravity, Server B does collisions"

**Problems:**
- Massive data transfer between phases
- Serialized computation (can't parallelize)
- Single point of failure for each force type

## The N-Body Nightmare

### Full N² Calculation
- 10M particles = 100 trillion force calculations
- Must share all particle positions with all servers
- Bandwidth: 10M particles × 16 bytes × 60 Hz = 10 GB/s

### Barnes-Hut Approximation
- Distant particles grouped into masses
- Tree structure must be shared/synchronized
- Who builds the tree? Who owns it?
- Tree changes every timestep

### Particle-Mesh Methods
- Particles mapped to grid
- Grid forces calculated via FFT
- Requires global synchronization
- Grid boundaries are artificial

## The Consistency Challenge

### Option 1: Lockstep Simulation
```
All servers must complete timestep N before any start N+1
```
- **Pro:** Perfect consistency
- **Con:** Slowest server bottlenecks everyone
- **Con:** Network latency adds up

### Option 2: Asynchronous Simulation
```
Servers run at their own pace, synchronize periodically
```
- **Pro:** Better performance
- **Con:** Divergence accumulates
- **Con:** Reconciliation is complex

### Option 3: Regional Time
```
Different regions can be at different simulation times
```
- **Pro:** Natural load balancing
- **Con:** Boundary interactions become undefined
- **Con:** Causality violations

## The Trust Problem

If anyone can contribute a server:
- How do we know they're computing correctly?
- What if they drop particles?
- What if they inject energy?
- What if they're just slow?

### Validation Approaches
- **Redundant calculation:** Multiple servers compute same thing (expensive!)
- **Spot checking:** Randomly verify subsets (can miss problems)
- **Conservation laws:** Check physics invariants (catches some errors)
- **Reputation systems:** Trust builds over time (vulnerable initially)

## The Communication Overhead

### What Must Be Shared?

**Every timestep:**
- Particle positions (for gravity)
- Boundary particles (for migration)
- New springs formed
- Collision events
- Temperature changes

**Periodically:**
- Full state synchronization
- Checkpoints
- Validation data

### Bandwidth Requirements

Rough estimate for 10M particles across 10 servers:
- Position updates: 10M × 8 bytes × 60 Hz = 5 GB/s
- If every server needs all positions: 50 GB/s total
- Plus events, springs, overhead: Could reach 100 GB/s

## The Philosophical Challenge

**The simulation is continuous. Our computers are discrete.**

- Particles exist in continuous space
- Forces propagate instantaneously (in classical mechanics)
- But we compute in discrete timesteps
- On discrete machines
- Connected by discrete messages

How do we approximate the continuous with the discrete, distributed?

## Open Questions

1. **Is true distribution possible?** Or do we need a primary authority?

2. **What's the right granularity?** Particles? Composites? Regions? Forces?

3. **How much inconsistency is acceptable?** Perfect physics or good-enough?

4. **Can we leverage structure?** Are galaxies/composites natural boundaries after all?

5. **What about intentional boundaries?** Multiple universes connected by wormholes?

6. **Should physics be deterministic?** Same input → same output across servers?

7. **How do we handle time?** Global clock? Local clocks? Relative time?

8. **What about player authority?** Can players run their own ship simulation?

## Potential Directions

### Hybrid Authority
- Core servers maintain canonical state
- Contributor servers do read-only computation
- Results validated and integrated by core

### Hierarchical Simulation
- Global forces computed rarely (gravity between galaxies)
- Local forces computed frequently (collisions)
- Different update rates for different scales

### Eventual Consistency
- Accept temporary inconsistencies
- Physics will self-correct (conservation laws)
- Players see locally consistent view

### Sharded Universes
- Multiple independent simulations
- Wormholes/portals between them
- Each shard fully consistent internally

## Pragmatic Approximations

### Force Cutoffs

**Gravity cutoff:**
```
if (distance > GRAVITY_CUTOFF) {
    // Ignore force - it's negligible
    force = 0;
}
```
- At 10,000 units, gravity might be 0.0001% of local forces
- Can dramatically reduce N² to N×k where k << N
- Makes distribution actually feasible

**Different cutoffs for different forces:**
- Gravity: 10,000 units (still want galactic dynamics)
- Springs: 10 units (only immediate neighbors)
- Collisions: 5 units (contact only)
- Heat: 2 units (conduction is local)
- Radiation: 1,000 units (but falls off fast)

### Hierarchical Time Steps

Not everything needs updating every frame:
- Collisions: Every frame (critical)
- Local gravity: Every frame
- Distant gravity: Every 10 frames
- Galaxy-scale forces: Every 100 frames
- Background radiation: Every 1000 frames

### Statistical Approximations

Replace distant particles with statistical fields:
- 1M particles at distance > 10,000 → "Mass concentration of 1M at position X"
- Particle clouds → Density fields
- Distant heat sources → Ambient temperature

### Locality Assumptions

**Most interactions are local:**
- 99% of forces might come from nearest 100 particles
- Springs only form with touching particles
- Heat only conducts to neighbors
- Collisions are purely local

This suggests natural server boundaries at "interaction voids"

### Lazy Evaluation

Don't compute until needed:
- Player looking at region? Full simulation
- No players nearby? Reduced simulation
- Very distant? Statistical aggregation only

## The Reality Check

**Maybe true distribution isn't the answer.** Maybe we need:
- One authoritative simulation
- Many observers/viewers
- Contribution through resources, not direct computation
- Or accept limitations on scale

**Or maybe with smart approximations, distribution becomes tractable:**
- Force cutoffs eliminate most communication
- Local servers handle local physics
- Global server handles long-range approximations
- Good enough physics that still feels real

## Summary

Distributing particle simulation is fundamentally hard because:
1. **Everything affects everything** (N-body problem)
2. **Boundaries are artificial** (reality is continuous)
3. **Communication overhead** (sharing state is expensive)
4. **Consistency is hard** (distributed systems theorem)
5. **Trust is required** (but hard to verify)

We don't have the answer yet. But understanding the problem space is the first step toward finding one.

## Next Steps

Before implementing any distribution:
1. Maximize single-server performance
2. Measure actual bottlenecks
3. Prototype simple distribution models
4. Test with real network conditions
5. Accept that perfect distribution might be impossible

The perfect is the enemy of the good. A working 10M particle simulation on one beefy server might be better than a broken 100M particle simulation across many.
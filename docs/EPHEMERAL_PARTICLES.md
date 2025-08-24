# Ephemeral Particles - Reconsidered

## The Question

Should the physics engine have "ephemeral" particles with simplified physics? Or should all particles be equal?

## Key Insight: We're Already Meso-Scale

Our particles already represent meso-scale objects:
- A "gas particle" might be a cloud of billions of molecules
- A "dust particle" might be millions of grains
- Even our smallest particles are macro objects

## Do We Need Ephemeral Particles in Physics?

**Probably not.** Here's why:

### Solar Wind Example
Instead of ephemeral particles, model as:
- **Radiation pressure field** from stars (already in our design)
- Applies force to all particles based on distance/cross-section
- No particles needed

### Rocket Exhaust Example  
The physics engine handles:
- **Thrust force** on spacecraft (Newton's third law)
- **Mass reduction** of spacecraft (fuel consumed)
- **Heat generation** at engine

The visual client handles:
- Rendering exhaust plume (from ROCKET_THRUST event)
- Particle effects
- Glowing effects

### Debris/Shrapnel Example
This SHOULD be full particles because:
- Debris can hit other objects (needs collision)
- Debris has mass (affects gravity)
- Debris persists (doesn't just vanish)

## Conclusion: Keep It Pure

**All particles in the physics engine should be full particles.**

If we need to model something that would require too many particles:
1. Use **field effects** (radiation pressure, magnetic fields)
2. Use **statistical models** (gas pressure regions)
3. Use **event emissions** (let clients handle visuals)

## What the Physics Engine Does

When a rocket fires:
```
1. Apply thrust force to spacecraft
2. Reduce spacecraft mass (fuel consumed)
3. Emit ROCKET_THRUST event with position, direction, magnitude
4. That's it!
```

When a star emits solar wind:
```
1. Apply radiation pressure field to nearby particles
2. Emit SOLAR_WIND event with intensity
3. That's it!
```

## Benefits of This Approach

1. **Clean separation**: Physics is physics, rendering is rendering
2. **Consistency**: All particles follow same rules
3. **Simplicity**: No special cases in physics engine
4. **Flexibility**: Clients can render however they want

## The Real Optimization

If we need to handle millions of particles efficiently:
- **Spatial partitioning** (octree, grid)
- **Neighbor lists** (only check nearby particles)
- **Force cutoffs** (ignore distant weak forces)
- **Adaptive timestepping** (slow particles step less often)

But NOT different particle types with different physics.

## Summary

The physics engine deals with:
- **10+ million full particles** (all equal)
- **Field effects** (gravity, radiation, magnetic)
- **Event emissions** (for clients to interpret)

Visual effects are 100% the client's problem. The physics engine just does physics.
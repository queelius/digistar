# Soft Contact Forces and Collision System

## Core Philosophy

Soft contact forces create realistic deformation, bouncing, and breaking without explicit rigid body collision. Forces emerge at multiple scales:
1. **Particle-particle repulsion** - Direct overlaps
2. **Composite-composite collision** - Bounding region interactions
3. **Tidal deformation** - Gravity gradients stretch bodies

## Particle-Level Soft Contact

### Basic Repulsion Force

When particles overlap, they experience a repulsive force:

```cpp
struct ContactForce {
    float2 force;
    float penetration_depth;
    bool is_critical;  // May break springs
};

ContactForce calculateSoftContact(const Particle& p1, const Particle& p2) {
    float2 delta = p2.pos - p1.pos;
    float dist = length(delta);
    float min_dist = p1.radius + p2.radius;
    
    ContactForce contact = {{0, 0}, 0, false};
    
    if (dist < min_dist && dist > 0) {
        // Penetration depth
        contact.penetration_depth = min_dist - dist;
        
        // Hertzian contact model (realistic soft contact)
        float stiffness = CONTACT_STIFFNESS * sqrt(p1.radius * p2.radius / (p1.radius + p2.radius));
        float force_mag = stiffness * pow(contact.penetration_depth, 1.5f);
        
        // Add damping to prevent oscillation
        float2 v_rel = p2.vel - p1.vel;
        float v_normal = dot(v_rel, delta) / dist;
        force_mag += CONTACT_DAMPING * v_normal;
        
        // Apply force
        contact.force = normalize(delta) * force_mag;
        
        // Check if force is strong enough to break springs
        contact.is_critical = force_mag > SPRING_BREAK_THRESHOLD;
    }
    
    return contact;
}
```

### Spring Breaking from Contact

Contact forces can break springs, causing deformation:

```cpp
void applyContactForces(std::vector<Particle>& particles, 
                       std::vector<Spring>& springs) {
    // Spatial hash for efficient neighbor finding
    SpatialHash hash(particles);
    
    for (size_t i = 0; i < particles.size(); i++) {
        auto neighbors = hash.getNearby(particles[i].pos, particles[i].radius * 2);
        
        for (int j : neighbors) {
            if (i >= j) continue;  // Avoid double-counting
            
            ContactForce contact = calculateSoftContact(particles[i], particles[j]);
            
            if (contact.penetration_depth > 0) {
                // Apply repulsion
                particles[i].vel -= contact.force * dt / particles[i].mass;
                particles[j].vel += contact.force * dt / particles[j].mass;
                
                // Check springs connected to these particles
                if (contact.is_critical) {
                    for (auto& spring : springs) {
                        if ((spring.i == i || spring.i == j) && !spring.broken) {
                            // Contact force stresses the spring
                            spring.stress += length(contact.force);
                            
                            if (spring.stress > spring.break_force) {
                                spring.broken = true;
                                // Particles fly apart
                                particles[spring.i].vel -= normalize(contact.force) * BREAK_IMPULSE;
                                particles[spring.j].vel += normalize(contact.force) * BREAK_IMPULSE;
                            }
                        }
                    }
                }
            }
        }
    }
}
```

## Composite-Level Collision

### Bounding Region Overlap Detection

Fast broad-phase collision using bounding regions:

```cpp
struct CollisionPair {
    CompositeBody* body1;
    CompositeBody* body2;
    float overlap_amount;
    float2 collision_normal;
    float2 collision_point;
};

// Check bounding sphere overlap (fast)
bool checkBoundingSphereOverlap(const CompositeBody& b1, const CompositeBody& b2) {
    float dist = length(b2.center_of_mass - b1.center_of_mass);
    return dist < b1.bounding_radius + b2.bounding_radius;
}

// Check oriented box overlap (medium)
bool checkOrientedBoxOverlap(const OrientedBox& box1, const OrientedBox& box2) {
    // Separating Axis Theorem (SAT)
    float2 axes[4] = {
        rotate({1, 0}, box1.angle),
        rotate({0, 1}, box1.angle),
        rotate({1, 0}, box2.angle),
        rotate({0, 1}, box2.angle)
    };
    
    for (const auto& axis : axes) {
        float min1, max1, min2, max2;
        projectOntoAxis(box1, axis, min1, max1);
        projectOntoAxis(box2, axis, min2, max2);
        
        if (max1 < min2 || max2 < min1) {
            return false;  // Separation found
        }
    }
    return true;  // No separation = overlap
}
```

### Convex Hull Collision (Most Accurate)

For precise collision between complex shapes:

```cpp
struct ConvexHullCollision {
    bool colliding;
    float2 mtv;  // Minimum Translation Vector
    float penetration;
    float2 contact_point;
};

ConvexHullCollision checkConvexHullCollision(const ConvexHull& hull1, 
                                            const ConvexHull& hull2) {
    ConvexHullCollision result = {false, {0, 0}, 0, {0, 0}};
    
    float min_overlap = INFINITY;
    float2 min_axis;
    
    // Check all potential separating axes
    auto checkAxes = [&](const ConvexHull& hull) {
        for (size_t i = 0; i < hull.vertices.size(); i++) {
            // Edge normal as potential separating axis
            float2 edge = hull.vertices[(i+1) % hull.vertices.size()] - hull.vertices[i];
            float2 axis = normalize(perp(edge));
            
            float min1, max1, min2, max2;
            projectHullOntoAxis(hull1, axis, min1, max1);
            projectHullOntoAxis(hull2, axis, min2, max2);
            
            float overlap = std::min(max1, max2) - std::max(min1, min2);
            
            if (overlap < 0) {
                return false;  // Separation found
            }
            
            if (overlap < min_overlap) {
                min_overlap = overlap;
                min_axis = axis;
            }
        }
        return true;
    };
    
    // Check axes from both hulls
    if (!checkAxes(hull1) || !checkAxes(hull2)) {
        return result;  // No collision
    }
    
    // Collision detected
    result.colliding = true;
    result.penetration = min_overlap;
    result.mtv = min_axis * min_overlap;
    
    // Find contact point (deepest penetrating vertex)
    result.contact_point = findDeepestPoint(hull1, hull2, min_axis);
    
    return result;
}
```

### Composite Contact Response

Apply forces to entire composites based on collision:

```cpp
void resolveCompositeCollision(CompositeBody& body1, 
                              CompositeBody& body2,
                              const ConvexHullCollision& collision,
                              std::vector<Particle>& particles) {
    if (!collision.colliding) return;
    
    // Calculate relative velocity at contact point
    float2 v1 = body1.getVelocityAt(collision.contact_point);
    float2 v2 = body2.getVelocityAt(collision.contact_point);
    float2 v_rel = v2 - v1;
    
    // Check if bodies are separating
    if (dot(v_rel, collision.mtv) > 0) return;
    
    // Calculate impulse magnitude
    float e = RESTITUTION;  // Coefficient of restitution
    float v_normal = dot(v_rel, normalize(collision.mtv));
    
    float2 r1 = collision.contact_point - body1.center_of_mass;
    float2 r2 = collision.contact_point - body2.center_of_mass;
    
    float angular_factor1 = cross(r1, collision.mtv) * cross(r1, collision.mtv) / body1.moment_of_inertia;
    float angular_factor2 = cross(r2, collision.mtv) * cross(r2, collision.mtv) / body2.moment_of_inertia;
    
    float j = -(1 + e) * v_normal / 
              (1/body1.total_mass + 1/body2.total_mass + angular_factor1 + angular_factor2);
    
    // Apply impulse to all particles in both bodies
    float2 impulse = normalize(collision.mtv) * j;
    
    // Body 1
    for (int idx : body1.particle_indices) {
        particles[idx].vel -= impulse / body1.total_mass;
        
        // Add rotational component
        float2 r = particles[idx].pos - body1.center_of_mass;
        float angular_impulse = cross(r1, impulse) / body1.moment_of_inertia;
        particles[idx].vel -= perp(r) * angular_impulse;
    }
    
    // Body 2
    for (int idx : body2.particle_indices) {
        particles[idx].vel += impulse / body2.total_mass;
        
        float2 r = particles[idx].pos - body2.center_of_mass;
        float angular_impulse = cross(r2, impulse) / body2.moment_of_inertia;
        particles[idx].vel += perp(r) * angular_impulse;
    }
    
    // Position correction to resolve penetration
    float2 correction = collision.mtv * POSITION_CORRECTION_FACTOR;
    
    for (int idx : body1.particle_indices) {
        particles[idx].pos -= correction * body2.total_mass / (body1.total_mass + body2.total_mass);
    }
    
    for (int idx : body2.particle_indices) {
        particles[idx].pos += correction * body1.total_mass / (body1.total_mass + body2.total_mass);
    }
    
    // Check if collision should break springs
    if (j > COLLISION_BREAK_THRESHOLD) {
        breakSpringsNearContact(body1, body2, collision.contact_point, j);
    }
}
```

## Tidal Forces on Composites

Gravity gradients deform extended bodies:

```cpp
struct TidalDeformation {
    float stretch_factor;
    float2 stretch_axis;
    std::vector<int> particles_to_break;
};

TidalDeformation calculateTidalEffect(const CompositeBody& body,
                                     const Particle& massive_object,
                                     const std::vector<Particle>& particles) {
    TidalDeformation deform;
    
    float2 r_com = body.center_of_mass - massive_object.pos;
    float dist_com = length(r_com);
    deform.stretch_axis = normalize(r_com);
    
    // Tidal force gradient
    float tidal_gradient = 2.0f * G * massive_object.mass / (dist_com * dist_com * dist_com);
    
    // Check each particle in composite
    for (int idx : body.particle_indices) {
        float2 r = particles[idx].pos - body.center_of_mass;
        
        // Component along tidal axis
        float r_parallel = dot(r, deform.stretch_axis);
        
        // Tidal force on this particle
        float tidal_force = tidal_gradient * abs(r_parallel) * particles[idx].mass;
        
        // Check if this exceeds material strength
        if (tidal_force > particles[idx].material.tensile_strength) {
            deform.particles_to_break.push_back(idx);
        }
    }
    
    // Overall stretch factor
    deform.stretch_factor = 1.0f + tidal_gradient * body.bounding_radius;
    
    return deform;
}

void applyTidalDeformation(CompositeBody& body,
                          const TidalDeformation& tidal,
                          std::vector<Particle>& particles,
                          std::vector<Spring>& springs) {
    // Stretch particles along tidal axis
    for (int idx : body.particle_indices) {
        float2 r = particles[idx].pos - body.center_of_mass;
        float r_parallel = dot(r, tidal.stretch_axis);
        
        // Move particle to stretched position
        particles[idx].pos += tidal.stretch_axis * r_parallel * (tidal.stretch_factor - 1.0f);
    }
    
    // Break overstressed springs
    for (auto& spring : springs) {
        bool i_breaking = std::find(tidal.particles_to_break.begin(), 
                                   tidal.particles_to_break.end(), 
                                   spring.i) != tidal.particles_to_break.end();
        bool j_breaking = std::find(tidal.particles_to_break.begin(),
                                   tidal.particles_to_break.end(),
                                   spring.j) != tidal.particles_to_break.end();
        
        if ((i_breaking || j_breaking) && !spring.broken) {
            spring.broken = true;
            
            // Particles fly apart perpendicular to tidal axis
            float2 perp_axis = perp(tidal.stretch_axis);
            particles[spring.i].vel += perp_axis * TIDAL_BREAK_VELOCITY * (rand() / (float)RAND_MAX - 0.5f);
            particles[spring.j].vel -= perp_axis * TIDAL_BREAK_VELOCITY * (rand() / (float)RAND_MAX - 0.5f);
        }
    }
}
```

## Collision System Architecture

### Broad Phase

Quickly find potential collisions:

```cpp
class BroadPhaseCollision {
    SpatialHash particle_hash;
    std::vector<CollisionPair> potential_pairs;
    
public:
    void update(const std::vector<CompositeBody>& bodies) {
        potential_pairs.clear();
        
        // Use bounding spheres for broad phase
        for (size_t i = 0; i < bodies.size(); i++) {
            for (size_t j = i + 1; j < bodies.size(); j++) {
                if (checkBoundingSphereOverlap(bodies[i], bodies[j])) {
                    potential_pairs.push_back({&bodies[i], &bodies[j]});
                }
            }
        }
    }
    
    const std::vector<CollisionPair>& getPotentialPairs() { return potential_pairs; }
};
```

### Narrow Phase

Precise collision detection and response:

```cpp
class NarrowPhaseCollision {
public:
    void resolve(const std::vector<CollisionPair>& pairs,
                std::vector<Particle>& particles,
                std::vector<Spring>& springs) {
        for (const auto& pair : pairs) {
            // Build convex hulls if needed
            ConvexHull hull1 = buildConvexHull(*pair.body1, particles);
            ConvexHull hull2 = buildConvexHull(*pair.body2, particles);
            
            // Check precise collision
            ConvexHullCollision collision = checkConvexHullCollision(hull1, hull2);
            
            if (collision.colliding) {
                // Resolve collision
                resolveCompositeCollision(*pair.body1, *pair.body2, collision, particles);
                
                // May need to rebuild composites if springs broke
                pair.body1->needs_rebuild = true;
                pair.body2->needs_rebuild = true;
            }
        }
    }
};
```

## Performance Optimization

### Spatial Hashing

```cpp
class SpatialHash {
    float cell_size;
    std::unordered_map<uint64_t, std::vector<int>> cells;
    
    uint64_t hash(int x, int y) {
        return ((uint64_t)x << 32) | (uint32_t)y;
    }
    
public:
    SpatialHash(float size) : cell_size(size) {}
    
    void build(const std::vector<Particle>& particles) {
        cells.clear();
        for (size_t i = 0; i < particles.size(); i++) {
            int cx = (int)(particles[i].pos.x / cell_size);
            int cy = (int)(particles[i].pos.y / cell_size);
            cells[hash(cx, cy)].push_back(i);
        }
    }
    
    std::vector<int> getNearby(float2 pos, float radius) {
        std::vector<int> result;
        int r = (int)ceil(radius / cell_size);
        int cx = (int)(pos.x / cell_size);
        int cy = (int)(pos.y / cell_size);
        
        for (int dx = -r; dx <= r; dx++) {
            for (int dy = -r; dy <= r; dy++) {
                auto it = cells.find(hash(cx + dx, cy + dy));
                if (it != cells.end()) {
                    result.insert(result.end(), it->second.begin(), it->second.end());
                }
            }
        }
        return result;
    }
};
```

### Hierarchical Collision

Only check detailed collision for close objects:

```cpp
class HierarchicalCollision {
    struct CollisionLevel {
        float detail_threshold;  // Distance to use this level
        enum Type { SPHERE, BOX, HULL } type;
    };
    
    std::vector<CollisionLevel> levels = {
        {100.0f, CollisionLevel::SPHERE},  // Far: sphere only
        {50.0f, CollisionLevel::BOX},      // Medium: oriented box
        {10.0f, CollisionLevel::HULL}      // Close: full convex hull
    };
    
    CollisionLevel::Type getCollisionLevel(const CompositeBody& b1, 
                                          const CompositeBody& b2) {
        float dist = length(b2.center_of_mass - b1.center_of_mass);
        
        for (const auto& level : levels) {
            if (dist > level.detail_threshold) {
                return level.type;
            }
        }
        return CollisionLevel::HULL;
    }
};
```

## Emergent Behaviors

This system creates:

1. **Soft Deformation** - Bodies squish and deform on impact
2. **Structural Failure** - Collisions and tidal forces break composites apart
3. **Elastic Collisions** - Bodies bounce realistically
4. **Plastic Deformation** - Permanent shape changes from impacts
5. **Tidal Disruption** - Bodies torn apart by gravity gradients
6. **Cascading Failures** - One break leads to more breaks
7. **No Interpenetration** - Composite bounding regions prevent pass-through

## Integration Requirements for Contact Forces

### Critical Timestep Constraints
Our experiments revealed that contact forces require careful integration:

```cpp
// Contact stiffness creates same constraints as springs
const float CONTACT_STIFFNESS = 1000.0f;  // Requires dt < 0.063
const float MAX_TIMESTEP = 0.5f * 2.0f / sqrt(CONTACT_STIFFNESS / MIN_MASS);

// Adaptive timestep based on penetration depth
float computeContactTimestep(float penetration, float relative_velocity) {
    // Deep penetration = stiff response = smaller timestep
    float stiffness = CONTACT_STIFFNESS * (1 + penetration / PARTICLE_RADIUS);
    float dt_contact = 0.5f * 2.0f / sqrt(stiffness / reduced_mass);
    
    // High velocity impacts need smaller timesteps
    float dt_velocity = PARTICLE_RADIUS / abs(relative_velocity);
    
    return min(dt_contact, dt_velocity);
}
```

### Recommended Integration Methods
1. **Semi-implicit Euler**: Best general choice
   - Handles moderate contact stiffness
   - Energy dissipation helps damping
   
2. **Velocity Verlet**: For bouncy collisions
   - Better energy conservation
   - Good for elastic collisions
   
3. **Position-based methods**: For packed particles
   - Unconditionally stable
   - Handles simultaneous contacts

### Collision Cascade Handling
Our tests show collision cascades lose ~30% energy with discrete time:

```cpp
// Multi-resolution timesteps for collision cascades
void handleCollisionCascade(float dt_frame) {
    // Detect high-energy zones
    std::vector<int> collision_zones = detectCollisionClusters();
    
    // Subcycle collision zones with smaller timestep
    for (int zone : collision_zones) {
        float dt_local = dt_frame / 10;  // 10x smaller timestep
        for (int sub = 0; sub < 10; sub++) {
            updateContactForces(zone, dt_local);
            integrate(zone, dt_local);
        }
    }
    
    // Normal timestep for rest of simulation
    updateNonCollisionParticles(dt_frame);
}
```

## Constants

```cpp
// Contact forces (with integration requirements)
const float CONTACT_STIFFNESS = 1000.0f;  // Requires dt < 0.063
const float CONTACT_DAMPING = 10.0f;
const float SPRING_BREAK_THRESHOLD = 100.0f;
const float BREAK_IMPULSE = 5.0f;

// Collision
const float RESTITUTION = 0.6f;  // Bounciness
const float POSITION_CORRECTION_FACTOR = 0.8f;
const float COLLISION_BREAK_THRESHOLD = 500.0f;

// Tidal
const float TIDAL_BREAK_VELOCITY = 10.0f;
const float G = 1.0f;  // Gravitational constant

// Integration safety
const float TIMESTEP_SAFETY_FACTOR = 0.5f;  // Use half theoretical limit
const float MAX_PENETRATION = 0.1f * PARTICLE_RADIUS;  // Trigger subcycling
```

## Summary

The multi-level collision system ensures:
- **Particle overlaps** create local deformation
- **Composite collisions** handle bulk interactions efficiently  
- **Tidal forces** create realistic stretching and breaking
- **Convex hulls** prevent fast objects from passing through
- **Emergent soft-body dynamics** from simple contact forces

The beauty is that complex behaviors emerge naturally from these simple rules!
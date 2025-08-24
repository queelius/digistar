# Composite Bodies and Connected Components Design

## Core Philosophy

Composite bodies emerge from spring networks. Using efficient graph algorithms (Union-Find), we identify connected components and compute aggregate properties that enable new dynamics and behaviors.

## Connected Component Detection

### Union-Find Algorithm

The fastest way to track connected components with dynamic spring networks:

```cpp
class UnionFind {
private:
    std::vector<int> parent;
    std::vector<int> rank;
    
public:
    UnionFind(int n) : parent(n), rank(n, 0) {
        for (int i = 0; i < n; i++) {
            parent[i] = i;
        }
    }
    
    int find(int x) {
        if (parent[x] != x) {
            parent[x] = find(parent[x]);  // Path compression
        }
        return parent[x];
    }
    
    void unite(int x, int y) {
        int px = find(x);
        int py = find(y);
        if (px == py) return;
        
        // Union by rank
        if (rank[px] < rank[py]) {
            parent[px] = py;
        } else if (rank[px] > rank[py]) {
            parent[py] = px;
        } else {
            parent[py] = px;
            rank[px]++;
        }
    }
    
    bool connected(int x, int y) {
        return find(x) == find(y);
    }
};
```

**Performance**: Nearly O(1) per operation with path compression and union by rank!

### Building Components from Springs

```cpp
std::vector<CompositeBody> findCompositesBodies(
    const std::vector<Particle>& particles,
    const std::vector<Spring>& springs) {
    
    int n = particles.size();
    UnionFind uf(n);
    
    // Unite particles connected by springs
    for (const auto& spring : springs) {
        if (!spring.broken) {
            uf.unite(spring.i, spring.j);
        }
    }
    
    // Group particles by component
    std::unordered_map<int, std::vector<int>> components;
    for (int i = 0; i < n; i++) {
        int root = uf.find(i);
        components[root].push_back(i);
    }
    
    // Create composite bodies
    std::vector<CompositeBody> bodies;
    for (const auto& [root, indices] : components) {
        if (indices.size() > 1) {  // Skip singletons
            CompositeBody body;
            body.particle_indices = indices;
            body.computeProperties(particles);
            bodies.push_back(body);
        }
    }
    
    return bodies;
}
```

## Composite Body Properties

### Core Properties

```cpp
struct CompositeBody {
    // Composition
    std::vector<int> particle_indices;
    int num_particles;
    
    // Mass properties
    float total_mass;
    float2 center_of_mass;
    float moment_of_inertia;
    
    // Motion properties
    float2 mean_velocity;
    float angular_velocity;
    
    // Temperature properties
    float internal_temp;      // From velocity dispersion
    float average_temp;       // Mean particle temperature
    
    // Structural properties
    float compactness;        // How tightly packed
    float elongation;         // Aspect ratio
    float2 principal_axis;    // Main orientation
    
    // Spring network properties
    int num_springs;
    float avg_spring_stress;
    float structural_integrity;  // 0 = falling apart, 1 = solid
};
```

### Computing Properties

```cpp
void CompositeBody::computeProperties(const std::vector<Particle>& particles,
                                      const std::vector<Spring>& springs) {
    num_particles = particle_indices.size();
    if (num_particles == 0) return;
    
    // Reset aggregates
    total_mass = 0;
    center_of_mass = {0, 0};
    float2 total_momentum = {0, 0};
    average_temp = 0;
    
    // First pass: basic aggregates
    for (int idx : particle_indices) {
        const auto& p = particles[idx];
        total_mass += p.mass;
        center_of_mass.x += p.pos.x * p.mass;
        center_of_mass.y += p.pos.y * p.mass;
        total_momentum.x += p.vel.x * p.mass;
        total_momentum.y += p.vel.y * p.mass;
        average_temp += p.temp_internal;
    }
    
    center_of_mass /= total_mass;
    mean_velocity = total_momentum / total_mass;
    average_temp /= num_particles;
    
    // Second pass: relative properties
    float total_kinetic = 0;
    moment_of_inertia = 0;
    float angular_momentum = 0;
    
    // For elongation calculation
    float Ixx = 0, Iyy = 0, Ixy = 0;
    
    for (int idx : particle_indices) {
        const auto& p = particles[idx];
        
        // Position relative to COM
        float2 r = p.pos - center_of_mass;
        float r_sq = r.x * r.x + r.y * r.y;
        
        // Velocity relative to mean
        float2 v_rel = p.vel - mean_velocity;
        
        // Internal temperature from velocity dispersion
        total_kinetic += 0.5f * p.mass * (v_rel.x * v_rel.x + v_rel.y * v_rel.y);
        
        // Moment of inertia
        moment_of_inertia += p.mass * r_sq;
        
        // Angular momentum L = r × mv
        angular_momentum += p.mass * (r.x * v_rel.y - r.y * v_rel.x);
        
        // Inertia tensor components (for shape analysis)
        Ixx += p.mass * r.y * r.y;
        Iyy += p.mass * r.x * r.x;
        Ixy -= p.mass * r.x * r.y;
    }
    
    // Internal temperature from kinetic theory
    // T = 2 * KE / (N * k_B) for 2D
    internal_temp = 2.0f * total_kinetic / (num_particles * BOLTZMANN);
    
    // Angular velocity
    if (moment_of_inertia > 0) {
        angular_velocity = angular_momentum / moment_of_inertia;
    }
    
    // Shape analysis via eigenvalues of inertia tensor
    float trace = Ixx + Iyy;
    float det = Ixx * Iyy - Ixy * Ixy;
    float discriminant = trace * trace - 4.0f * det;
    
    if (discriminant > 0) {
        float sqrt_disc = sqrt(discriminant);
        float lambda1 = (trace + sqrt_disc) / 2.0f;
        float lambda2 = (trace - sqrt_disc) / 2.0f;
        
        // Elongation: ratio of principal moments
        if (lambda2 > 0) {
            elongation = sqrt(lambda1 / lambda2);
        } else {
            elongation = 1.0f;
        }
        
        // Principal axis (eigenvector for larger eigenvalue)
        if (abs(lambda1 - Ixx) > 0.001f) {
            principal_axis.x = Ixy;
            principal_axis.y = lambda1 - Ixx;
            float len = sqrt(principal_axis.x * principal_axis.x + 
                           principal_axis.y * principal_axis.y);
            if (len > 0) {
                principal_axis /= len;
            }
        } else {
            principal_axis = {1, 0};
        }
    }
    
    // Compactness: how close particles are to COM
    float avg_distance = 0;
    float max_distance = 0;
    for (int idx : particle_indices) {
        float2 r = particles[idx].pos - center_of_mass;
        float dist = sqrt(r.x * r.x + r.y * r.y);
        avg_distance += dist;
        max_distance = std::max(max_distance, dist);
    }
    avg_distance /= num_particles;
    
    // Compactness: 1 = very compact, 0 = very spread out
    if (max_distance > 0) {
        compactness = avg_distance / max_distance;
    } else {
        compactness = 1.0f;
    }
    
    // Spring network analysis
    num_springs = 0;
    avg_spring_stress = 0;
    float max_stress = 0;
    
    for (const auto& spring : springs) {
        // Check if spring connects particles in this body
        bool i_in_body = false, j_in_body = false;
        for (int idx : particle_indices) {
            if (idx == spring.i) i_in_body = true;
            if (idx == spring.j) j_in_body = true;
        }
        
        if (i_in_body && j_in_body && !spring.broken) {
            num_springs++;
            
            // Calculate spring stress
            float2 r = particles[spring.j].pos - particles[spring.i].pos;
            float dist = sqrt(r.x * r.x + r.y * r.y);
            float stretch = abs(dist - spring.rest_length) / spring.rest_length;
            float stress = stretch * spring.stiffness / spring.break_force;
            
            avg_spring_stress += stress;
            max_stress = std::max(max_stress, stress);
        }
    }
    
    if (num_springs > 0) {
        avg_spring_stress /= num_springs;
    }
    
    // Structural integrity (0 = falling apart, 1 = solid)
    // Based on spring connectivity and stress
    float connectivity = (float)num_springs / (num_particles * (num_particles - 1) / 2);
    structural_integrity = connectivity * (1.0f - avg_spring_stress);
}
```

## Composite-Level Effects

The Union-Find algorithm makes identifying composites blazingly fast (nearly O(1)), enabling rich physics that emerge from composite properties:

### 1. Resonance Effects

Composites can enter resonance when rotation matches natural frequencies:

```cpp
struct ResonanceEffect {
    float amplification;
    float stress_multiplier;
    bool is_destructive;
};

ResonanceEffect calculateResonance(const CompositeBody& body) {
    // Natural frequency from spring network
    float natural_freq = sqrt(body.avg_spring_stiffness / body.total_mass);
    
    // Check if rotation matches natural frequency
    float rotation_freq = abs(body.angular_velocity) / (2.0f * M_PI);
    float ratio = rotation_freq / natural_freq;
    
    // Resonance peaks at integer ratios
    float resonance = 0;
    for (int n = 1; n <= 3; n++) {
        float delta = abs(ratio - n);
        resonance += exp(-delta * delta * 10.0f);  // Sharp peaks
    }
    
    return {
        .amplification = 1.0f + resonance * 5.0f,
        .stress_multiplier = 1.0f + resonance * 2.0f,
        .is_destructive = resonance > 0.8f
    };
}
```

### 2. Thermal Conductivity

Heat flows through composites based on spring connectivity:

```cpp
float compositeThermaConductivity(const CompositeBody& body) {
    // More springs = better conduction
    float connectivity = (float)body.num_springs / body.num_particles;
    
    // Compactness affects conduction
    return connectivity * body.compactness * MATERIAL_CONDUCTIVITY;
}

void propagateHeatInComposite(CompositeBody& body, float dt) {
    float conductivity = compositeThermaConductivity(body);
    
    // Heat diffusion through spring network
    for (const Spring& spring : body.internal_springs) {
        float temp_diff = particles[spring.j].temp - particles[spring.i].temp;
        float heat_flow = conductivity * temp_diff * dt;
        
        particles[spring.i].temp += heat_flow / particles[spring.i].mass;
        particles[spring.j].temp -= heat_flow / particles[spring.j].mass;
    }
}
```

### 3. Collective Drag

Shape and orientation affect drag:

```cpp
float2 compositeDragForce(const CompositeBody& body, const float2& flow_velocity) {
    float2 v_rel = flow_velocity - body.mean_velocity;
    
    // Cross-section depends on orientation
    float cross_section;
    if (body.elongation > 1.5f) {
        // Elongated body - orientation matters
        float angle = dot(normalize(v_rel), body.principal_axis);
        cross_section = body.bounding_radius * 
                       (1.0f + (body.elongation - 1.0f) * abs(angle));
    } else {
        // Roughly spherical
        cross_section = body.bounding_radius;
    }
    
    // Drag coefficient from compactness
    float drag_coeff = 0.5f * (2.0f - body.compactness);
    return -normalize(v_rel) * drag_coeff * cross_section * length(v_rel);
}
```

### 4. Structural Vibrations

Spring networks create vibrational modes:

```cpp
void applyStructuralVibration(CompositeBody& body, float frequency, float amplitude) {
    // Each particle oscillates based on distance from COM
    for (int idx : body.particle_indices) {
        float2 r = particles[idx].pos - body.center_of_mass;
        float phase = length(r) / body.bounding_radius * M_PI;
        
        float oscillation = amplitude * sin(frequency * time + phase);
        particles[idx].pos += normalize(r) * oscillation;
    }
}
```

## Bounding Regions & Manipulation

### Bounding Region Strategies

Different bounding regions serve different purposes:

```cpp
// 1. Simple sphere - fast, good for roughly circular bodies
struct BoundingSphere {
    float2 center;
    float radius;
    
    bool contains(const float2& point) {
        return distance(point, center) <= radius;
    }
    
    static BoundingSphere fromComposite(const CompositeBody& body) {
        return {body.center_of_mass, body.bounding_radius};
    }
};

// 2. Oriented box - better for elongated bodies
struct OrientedBoundingBox {
    float2 center;
    float2 half_extents;  // Width/height from center
    float angle;           // Rotation angle
    
    bool contains(const float2& point) {
        // Transform point to box space
        float2 local = rotate(point - center, -angle);
        return abs(local.x) <= half_extents.x && 
               abs(local.y) <= half_extents.y;
    }
    
    static OrientedBoundingBox fromComposite(const CompositeBody& body) {
        float angle = atan2(body.principal_axis.y, body.principal_axis.x);
        
        // Project particles onto principal axes
        float min_x = INFINITY, max_x = -INFINITY;
        float min_y = INFINITY, max_y = -INFINITY;
        
        for (int idx : body.particle_indices) {
            float2 local = rotate(particles[idx].pos - body.center_of_mass, -angle);
            min_x = min(min_x, local.x);
            max_x = max(max_x, local.x);
            min_y = min(min_y, local.y);
            max_y = max(max_y, local.y);
        }
        
        return {
            body.center_of_mass,
            {(max_x - min_x) / 2.0f, (max_y - min_y) / 2.0f},
            angle
        };
    }
};

// 3. Convex hull - most accurate but slower
struct ConvexHull {
    std::vector<float2> vertices;
    
    bool contains(const float2& point) {
        // Point-in-polygon test
        for (size_t i = 0; i < vertices.size(); i++) {
            float2 edge = vertices[(i+1) % vertices.size()] - vertices[i];
            float2 to_point = point - vertices[i];
            if (cross(edge, to_point) < 0) return false;
        }
        return true;
    }
    
    static ConvexHull fromComposite(const CompositeBody& body);  // Graham scan
};
```

### Manipulation API

Programmatic and interactive control:

```cpp
class CompositeManipulator {
    CompositeBody* selected = nullptr;
    BoundingRegion* bounds = nullptr;
    
public:
    // Selection
    void selectAt(const float2& cursor) {
        for (auto& body : composite_bodies) {
            if (body.getBounds().contains(cursor)) {
                selected = &body;
                bounds = body.getBounds();
                break;
            }
        }
    }
    
    // Direct manipulation
    void applyForce(const float2& force, const float2& application_point) {
        if (!selected) return;
        
        // Linear acceleration
        float2 accel = force / selected->total_mass;
        
        // Torque creates rotation
        float2 r = application_point - selected->center_of_mass;
        float torque = cross(r, force);
        float angular_accel = torque / selected->moment_of_inertia;
        
        // Apply to all particles
        for (int idx : selected->particle_indices) {
            particles[idx].vel += accel * dt;
            
            // Rotational component
            float2 r_particle = particles[idx].pos - selected->center_of_mass;
            float2 v_rot = perp(r_particle) * angular_accel;
            particles[idx].vel += v_rot * dt;
        }
    }
    
    // Dragging
    void dragTo(const float2& target) {
        if (!selected) return;
        
        float2 delta = target - selected->center_of_mass;
        float2 force = delta * DRAG_STRENGTH;
        applyForce(force, selected->center_of_mass);
    }
    
    // Rotation
    void rotate(float angle_delta) {
        if (!selected) return;
        
        float torque = angle_delta * selected->moment_of_inertia * ROTATION_STRENGTH;
        
        for (int idx : selected->particle_indices) {
            float2 r = particles[idx].pos - selected->center_of_mass;
            float2 tangent = perp(r);
            particles[idx].vel += tangent * torque / (selected->moment_of_inertia * length(r));
        }
    }
    
    // Impulse (for gameplay)
    void applyImpulse(const float2& impulse) {
        if (!selected) return;
        
        float2 vel_change = impulse / selected->total_mass;
        for (int idx : selected->particle_indices) {
            particles[idx].vel += vel_change;
        }
    }
};
```

## Hierarchical Composites

Composites can contain other composites:

```cpp
struct HierarchicalComposite {
    CompositeBody primary;
    std::vector<HierarchicalComposite*> sub_composites;
    int depth = 0;  // Nesting level
    
    // Aggregate properties
    float total_mass() {
        float mass = primary.total_mass;
        for (auto* sub : sub_composites) {
            mass += sub->total_mass();
        }
        return mass;
    }
    
    float2 center_of_mass() {
        float2 com = primary.center_of_mass * primary.total_mass;
        float total = primary.total_mass;
        
        for (auto* sub : sub_composites) {
            float sub_mass = sub->total_mass();
            com += sub->center_of_mass() * sub_mass;
            total += sub_mass;
        }
        
        return com / total;
    }
    
    // Apply force recursively
    void applyForce(const float2& force, const float2& point) {
        // Distribute force based on mass ratios
        float total = total_mass();
        
        float primary_fraction = primary.total_mass / total;
        applyForceToComposite(primary, force * primary_fraction, point);
        
        for (auto* sub : sub_composites) {
            float sub_fraction = sub->total_mass() / total;
            sub->applyForce(force * sub_fraction, point);
        }
    }
    
    // Break into sub-composites under stress
    void checkStructuralIntegrity() {
        if (primary.structural_integrity < 0.3f) {
            // Promote sub-composites to independent bodies
            for (auto* sub : sub_composites) {
                independent_composites.push_back(sub);
            }
            sub_composites.clear();
        }
    }
    
    // Recursive spring breaking
    void propagateBreakage(float stress_level) {
        // Break weak springs in primary
        for (auto& spring : primary.internal_springs) {
            if (spring.stress > spring.break_force * stress_level) {
                spring.broken = true;
                primary.needs_rebuild = true;
            }
        }
        
        // Propagate to sub-composites
        for (auto* sub : sub_composites) {
            sub->propagateBreakage(stress_level * 0.8f);  // Dampen with depth
        }
    }
};

// Build hierarchy from connected components
HierarchicalComposite* buildHierarchy(const std::vector<int>& particle_indices,
                                      const std::vector<Spring>& springs,
                                      int max_depth = 3) {
    if (particle_indices.size() < MIN_COMPOSITE_SIZE || max_depth == 0) {
        return nullptr;
    }
    
    auto* composite = new HierarchicalComposite();
    composite->depth = 3 - max_depth;
    
    // Use Union-Find to find sub-clusters
    UnionFind uf(particle_indices.size());
    
    // Connect particles with strong springs
    for (const auto& spring : springs) {
        if (spring.stiffness > HIERARCHY_STIFFNESS_THRESHOLD) {
            uf.unite(spring.i, spring.j);
        }
    }
    
    // Build sub-composites from clusters
    std::unordered_map<int, std::vector<int>> clusters;
    for (int idx : particle_indices) {
        clusters[uf.find(idx)].push_back(idx);
    }
    
    for (const auto& [root, cluster] : clusters) {
        if (cluster.size() > MIN_CLUSTER_SIZE) {
            auto* sub = buildHierarchy(cluster, springs, max_depth - 1);
            if (sub) {
                composite->sub_composites.push_back(sub);
            }
        }
    }
    
    return composite;
}
```

## Performance Optimization

### Incremental Updates

```cpp
class CompositeTracker {
    UnionFind uf;
    std::vector<CompositeBody> bodies;
    bool needs_rebuild = true;
    
    void onSpringAdded(int i, int j) {
        int root_i = uf.find(i);
        int root_j = uf.find(j);
        
        if (root_i != root_j) {
            // Merge components
            uf.unite(i, j);
            needs_rebuild = true;
        }
    }
    
    void onSpringBroken(int i, int j) {
        // Breaking might split component
        // Full rebuild needed (Union-Find doesn't support split)
        needs_rebuild = true;
    }
    
    void update(const std::vector<Particle>& particles,
                const std::vector<Spring>& springs) {
        if (needs_rebuild) {
            bodies = findCompositesBodies(particles, springs);
            needs_rebuild = false;
        } else {
            // Just update properties
            for (auto& body : bodies) {
                body.computeProperties(particles, springs);
            }
        }
    }
};
```

### Spatial Hierarchy

For very large composites, use spatial subdivision:

```cpp
struct HierarchicalComposite {
    CompositeBody root;
    std::vector<CompositeBody> sub_bodies;  // Spatial regions
    
    void buildHierarchy() {
        // Divide particles into spatial clusters
        // Each cluster becomes a sub-body
        // Root aggregates sub-body properties
    }
};
```

## Use Cases

### 1. Spacecraft
- Multiple particles connected by rigid springs
- High structural integrity required
- Tracks orientation for navigation
- Internal temperature affects crew

### 2. Asteroids
- Loosely bound rubble pile
- Low structural integrity
- Can fragment under stress
- Rotation affects stability

### 3. Space Stations
- Complex spring network
- Multiple sub-components
- Thermal management important
- Docking requires precise orientation

### 4. Planetary Rings
- Many small composites
- Frequent merging/splitting
- Collective dynamics
- Tidal forces cause alignment

## Emergent Behaviors

This system enables:

1. **Rigid Body Dynamics** - Composites rotate and translate as units
2. **Structural Failure** - Bodies break apart under stress
3. **Thermal Gradients** - Different parts at different temperatures
4. **Collective Motion** - Flocking/swarming of composite bodies
5. **Resonances** - Orbital/rotational resonances between bodies

## Implementation Phases

1. **Phase 1**: Basic Union-Find and component detection
2. **Phase 2**: Core properties (COM, mass, velocity)
3. **Phase 3**: Shape analysis (elongation, orientation)
4. **Phase 4**: Structural integrity and stress
5. **Phase 5**: Composite-level forces and torques

The beauty is that complex rigid body dynamics emerge naturally from particles and springs!
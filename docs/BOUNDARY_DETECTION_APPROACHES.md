# Boundary Detection Approaches for Composite Collisions

## Overview

When two composite bodies collide, we need to apply forces locally at the contact point, not uniformly across the entire composite. This document explores different approaches for efficiently detecting boundary particles and contact regions.

## Approach 1: Convex Hull Boundary Detection

### Concept
If we maintain a convex hull for each composite, particles on the hull ARE the boundary particles.

### Implementation

```cpp
struct CompositeBody {
    std::vector<uint32_t> particle_ids;
    std::vector<uint32_t> hull_particle_ids;  // Subset that forms convex hull
    std::vector<float2> hull_points;          // Hull vertices for collision detection
    
    void update_convex_hull() {
        // Extract positions
        std::vector<float2> positions;
        for (uint32_t id : particle_ids) {
            positions.push_back(particles[id].pos);
        }
        
        // Compute hull (Graham scan or QuickHull)
        hull_particle_ids = compute_convex_hull_indices(positions);
        
        // Store hull points for polygon tests
        hull_points.clear();
        for (uint32_t id : hull_particle_ids) {
            hull_points.push_back(particles[id].pos);
        }
    }
    
    bool is_boundary_particle(uint32_t id) {
        return std::find(hull_particle_ids.begin(), 
                        hull_particle_ids.end(), id) != hull_particle_ids.end();
    }
};
```

### Advantages
- Exact boundary detection for convex shapes
- Hull particles are guaranteed to be on the boundary
- Can use hull for fast collision detection (SAT algorithm)

### Disadvantages
- Only works for convex shapes (concave objects have interior points on hull)
- O(n log n) to compute hull
- Need to recompute when composite deforms

### When to Use
- Rigid or semi-rigid composites
- When composites are roughly convex
- When you need exact boundary

## Approach 2: Spring Connectivity Boundary Detection

### Concept
A particle is on the boundary if it has fewer springs than interior particles, or if it has springs only in certain directions.

```cpp
bool is_boundary_particle_by_springs(uint32_t id, CompositeBody& comp) {
    auto springs = spring_network.get_springs_for(id);
    
    // Method 1: Spring count (boundary has fewer connections)
    if (springs.size() < INTERIOR_SPRING_THRESHOLD) {
        return true;
    }
    
    // Method 2: Angular coverage (boundary has gaps in directions)
    std::vector<float> angles;
    float2 p_pos = particles[id].pos;
    
    for (const Spring& spring : springs) {
        uint32_t other = (spring.id1 == id) ? spring.id2 : spring.id1;
        if (!comp.contains(other)) continue;
        
        float2 dir = particles[other].pos - p_pos;
        angles.push_back(atan2(dir.y, dir.x));
    }
    
    std::sort(angles.begin(), angles.end());
    
    // Check for large gaps in angular coverage
    for (size_t i = 0; i < angles.size(); i++) {
        float gap = angles[(i+1) % angles.size()] - angles[i];
        if (gap < 0) gap += 2 * M_PI;
        if (gap > M_PI / 2) {  // 90-degree gap suggests boundary
            return true;
        }
    }
    
    return false;
}
```

### Advantages
- Works for any shape (convex or concave)
- Natural for spring-based composites
- Can detect "structural" boundaries

### Disadvantages
- O(k) per particle where k = springs per particle
- May miss boundary particles in dense networks
- Depends on spring topology

## Approach 3: Overlap Region Detection (Your Suggestion)

### Concept
Don't compute boundary particles at all. Instead, find the overlap region between bounding shapes and apply forces to all particles near that region.

```cpp
struct CollisionRegion {
    float2 center;           // Approximate contact point
    float radius;            // Region of influence
    float2 normal;           // Collision normal
    float penetration_depth; // Overlap amount
};

CollisionRegion find_collision_region(CompositeBody& comp1, CompositeBody& comp2) {
    CollisionRegion region;
    
    // Method 1: Bounding circle intersection
    float2 diff = comp2.center_of_mass - comp1.center_of_mass;
    float dist = diff.length();
    float overlap = (comp1.radius + comp2.radius) - dist;
    
    if (overlap > 0) {
        region.center = comp1.center_of_mass + diff * (comp1.radius / dist);
        region.radius = std::min(comp1.radius, comp2.radius) * 0.3f;
        region.normal = diff / dist;
        region.penetration_depth = overlap;
    }
    
    return region;
}

void apply_regional_forces(CompositeBody& comp1, CompositeBody& comp2) {
    CollisionRegion region = find_collision_region(comp1, comp2);
    
    if (region.penetration_depth <= 0) return;
    
    // Find particles in the collision region
    auto comp1_particles = get_particles_near(comp1, region.center, region.radius);
    auto comp2_particles = get_particles_near(comp2, region.center, region.radius);
    
    // Apply forces based on distance from contact center
    float force_mag = contact_stiffness * pow(region.penetration_depth, 1.5f);
    
    for (uint32_t id : comp1_particles) {
        float dist = (particles[id].pos - region.center).length();
        float falloff = 1.0f - (dist / region.radius);
        particles[id].force -= region.normal * force_mag * falloff;
    }
    
    for (uint32_t id : comp2_particles) {
        float dist = (particles[id].pos - region.center).length();
        float falloff = 1.0f - (dist / region.radius);
        particles[id].force += region.normal * force_mag * falloff;
    }
}
```

### Advantages
- Very fast - no boundary detection needed
- Works for any shape
- Natural force distribution
- Easily tunable (region size, falloff)

### Disadvantages
- Less physically accurate for rigid bodies
- May apply forces to interior particles unnecessarily
- Contact point is approximate

## Approach 4: Hybrid - Oriented Bounding Box with Edge Detection

### Concept
Use OBB (Oriented Bounding Box) for broad phase, then check particles near the overlapping edges.

```cpp
struct OBB {
    float2 center;
    float2 half_extents;  // Half width and height
    float angle;          // Rotation angle
    
    std::vector<float2> get_edges() {
        // Return 4 edge lines of the box
    }
};

void apply_obb_collision(CompositeBody& comp1, CompositeBody& comp2) {
    OBB obb1 = compute_obb(comp1);
    OBB obb2 = compute_obb(comp2);
    
    // Find overlapping edges using SAT
    auto overlap_edges = find_overlapping_edges(obb1, obb2);
    
    if (overlap_edges.empty()) return;
    
    // Apply forces only to particles near overlapping edges
    for (const Edge& edge : overlap_edges) {
        auto nearby = get_particles_near_line(edge.start, edge.end, EDGE_INFLUENCE_DIST);
        // Apply forces based on distance from edge
    }
}
```

## Approach 5: Grid-Based Boundary Detection

### Concept
Use the spatial grid itself to detect boundary particles - those in cells with empty neighbors.

```cpp
bool is_boundary_by_grid(uint32_t particle_id, SparseGrid& grid) {
    uint64_t cell = grid.get_cell(particles[particle_id].pos);
    auto neighbors = grid.get_neighbor_cells(cell);
    
    // Check if any neighbor cells are empty or contain non-composite particles
    for (uint64_t ncell : neighbors) {
        if (grid.cells[ncell].empty()) {
            return true;  // Empty neighbor = boundary
        }
        
        // Check if neighbor contains particles from other composites
        for (uint32_t id : grid.cells[ncell]) {
            if (composite_tracker.find(id) != composite_tracker.find(particle_id)) {
                return true;  // Different composite = boundary
            }
        }
    }
    
    return false;
}
```

## Performance Comparison

| Method | Computation Cost | Memory | Accuracy | Best For |
|--------|-----------------|---------|----------|----------|
| Convex Hull | O(n log n) build, O(1) query | O(√n) | Exact for convex | Rigid bodies |
| Spring Connectivity | O(k) per query | O(1) | Good | Soft bodies |
| Overlap Region | O(1) | O(1) | Approximate | Fast games |
| OBB Edges | O(n) build, O(1) query | O(1) | Good | Box-like objects |
| Grid-Based | O(1) per query | O(1) | Approximate | Already using grid |

## Recommendation

For DigiStar, I suggest a **hybrid approach**:

1. **Use Overlap Region (Approach 3) as default** - It's fastest and good enough for most cases
2. **Cache convex hulls for rigid composites** - When accuracy matters
3. **Fall back to spring connectivity for soft bodies** - When deformation is significant

```cpp
class CompositeCollisionSystem {
    void handle_collision(CompositeBody& comp1, CompositeBody& comp2) {
        // Fast path: overlap region for most collisions
        if (comp1.is_soft || comp2.is_soft) {
            apply_regional_forces(comp1, comp2);
        }
        // Accurate path: hull-based for rigid bodies
        else if (comp1.has_valid_hull && comp2.has_valid_hull) {
            apply_hull_based_forces(comp1, comp2);
        }
        // Fallback
        else {
            apply_regional_forces(comp1, comp2);
        }
    }
};
```

This gives us speed when we need it and accuracy when it matters!
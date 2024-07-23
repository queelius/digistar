# Composite Modeling in Large-Scale Space Simulation

We present an approach to modeling composites in a large-scale space simulation, where Big Atoms are the fundamental entities.

We use the Moore potential to model interactions between Big Atoms. When the attraction between Big Atoms
is sufficiently large, the distance between them is less than a threshold, and their interaction vectors align,
we automatically insert this information in a list of virtual connections. We also allow for the connections to
be explicitly defined with (tensor) springs with dampening, which can be used to customize composite objects
instead of relying on the Moore potential solely.

Based on the connectivity information, we cluster Big Atoms into composites using disjoint sets. These composites
can be treated as single entities for certain calculations, such as bounding volume approximations for interactions
between composites and big atoms and for visual representation (such as a convex hull or bounding sphere for rendering).

## Disjoint Sets and Clustering

When provided a list of links (whether virtually constructed and deconstructed from the Moore potential or explicitly defined as springs),
we can use disjoint sets to efficiently cluster Big Atoms into clusters, which we call composites.
The union-find algorithm allows for efficient clustering based on the connectivity information.

### Implementation

```cpp
class DisjointSet {
private:
    std::vector<int> parent;
    std::vector<int> rank;

public:
    DisjointSet(int n) : parent(n), rank(n, 0) {
        for (int i = 0; i < n; i++) parent[i] = i;
    }

    int find(int x) {
        if (parent[x] != x) parent[x] = find(parent[x]);
        return parent[x];
    }

    void unite(int x, int y) {
        int rootX = find(x), rootY = find(y);
        if (rootX == rootY) return;
        if (rank[rootX] < rank[rootY]) parent[rootX] = rootY;
        else if (rank[rootX] > rank[rootY]) parent[rootY] = rootX;
        else { parent[rootY] = rootX; rank[rootX]++; }
    }
};

struct Spring {
    int atom1, atom2;
    Tensor3x3 stiffness;
    float damping;
    float equilibrium;
    float breakingTension; // when F_spring > breakingTension, the spring breaks
};

// Function to update clusters based on virtual springs
void updateClusters(DisjointSet& ds, const std::vector<VirtualSpring>& springs) {
    for (const auto& spring : springs) {
        ds.unite(spring.atom1, spring.atom2);
    }
}
```

## 2. Clustering Big Atoms as Composites

### Process
1. Maintain a list of virtual springs. We call them virtual springs because they are not physical springs but rather represent the forces that hold Big Atoms together. They are dynamic and can be created or destroyed based on local interactions. We discuss the dynamics of these virtual springs in another section.
2. Use DisjointSet to efficiently cluster connected Big Atoms.
3. Treat each cluster as a composite for certain calculations.

For presentation purposes, a client may decide to render composites differently depending on the characteristics of the spring connections and Big Atom properties. For example, a composite with strong spring connections may be rendered as a solid object, while a composite with weak spring connections may be rendered as a cloud of particles.

### Advantages
- Allows for dynamic formation and breaking of composites.
- Efficient O(α(n)) time complexity for union and find operations.

## 3. Bounding Volumes for Composites

### Options
1. **Axis-Aligned Bounding Box (AABB)**
   - Fastest to compute and check for intersections.
   - O(n) computation time.
   
   ```cpp
   struct AABB {
       float3 min, max;
   };

   AABB computeAABB(const std::vector<BigAtom>& atoms) {
       AABB box = {atoms[0].position, atoms[0].position};
       for (const auto& atom : atoms) {
           box.min = min(box.min, atom.position - atom.radius);
           box.max = max(box.max, atom.position + atom.radius);
       }
       return box;
   }
   ```

2. **Bounding Sphere**
   - Better for roughly spherical clusters.
   - O(n) computation time.

   ```cpp
   struct BoundingSphere {
       float3 center;
       float radius;
   };

   BoundingSphere computeBoundingSphere(const std::vector<BigAtom>& atoms) {
       float3 center = {0, 0, 0};
       for (const auto& atom : atoms) center += atom.position;
       center /= atoms.size();

       float maxRadiusSq = 0;
       for (const auto& atom : atoms) {
           float distSq = lengthSquared(atom.position - center);
           maxRadiusSq = max(maxRadiusSq, distSq + atom.radius * atom.radius);
       }

       return {center, sqrt(maxRadiusSq)};
   }
   ```

### Usage
- Quick overlap tests between composites.
- Approximate collision detection and response.

## 4. Composite Properties

### Temperature
- Based on the kinetic energy of constituent Big Atoms relative to the composite's center of mass.

```cpp
float calculateTemperature(const std::vector<BigAtom>& atoms, float3 comVelocity) {
    float kineticEnergy = 0;
    for (const auto& atom : atoms) {
        float3 relativeVelocity = atom.velocity - comVelocity;
        kineticEnergy += 0.5f * atom.mass * dot(relativeVelocity, relativeVelocity);
    }
    return kineticEnergy / (1.5f * atoms.size() * BOLTZMANN_CONSTANT);
}
```

### Rotation
- Calculated using the angular momentum and moment of inertia of the composite.

```cpp
float3 calculateAngularMomentum(const std::vector<BigAtom>& atoms, float3 com, float3 comVelocity) {
    float3 L = {0, 0, 0};
    for (const auto& atom : atoms) {
        float3 r = atom.position - com;
        float3 v = atom.velocity - comVelocity;
        L += cross(r, atom.mass * v);
    }
    return L;
}

Tensor3x3 calculateMomentOfInertia(const std::vector<BigAtom>& atoms, float3 com) {
    Tensor3x3 I = {0};
    for (const auto& atom : atoms) {
        float3 r = atom.position - com;
        float r2 = dot(r, r);
        I.xx += atom.mass * (r2 - r.x * r.x);
        I.yy += atom.mass * (r2 - r.y * r.y);
        I.zz += atom.mass * (r2 - r.z * r.z);
        I.xy -= atom.mass * r.x * r.y;
        I.xz -= atom.mass * r.x * r.z;
        I.yz -= atom.mass * r.y * r.z;
    }
    I.yx = I.xy; I.zx = I.xz; I.zy = I.yz;
    return I;
}

float3 calculateAngularVelocity(float3 L, Tensor3x3 I) {
    // Solve I * ω = L for ω
    // This is a simplification; in practice, you'd need to invert I
    return L / (I.xx + I.yy + I.zz);
}
```

## 5. Composite Interactions

### Repulsions
- Use bounding volumes for quick overlap tests.
- Apply repulsive forces to overlapping composites.

```cpp
void applyRepulsion(Composite& c1, Composite& c2, float repulsionStrength) {
    if (!spheresOverlap(c1.boundingSphere, c2.boundingSphere)) return;

    float3 direction = c2.centerOfMass - c1.centerOfMass;
    float distance = length(direction);
    float overlap = c1.boundingSphere.radius + c2.boundingSphere.radius - distance;

    if (overlap > 0) {
        float3 force = normalize(direction) * repulsionStrength * overlap;
        applyForceToComposite(c1, -force);
        applyForceToComposite(c2, force);
    }
}
```

### Force Distribution
- When applying forces to composites, distribute them to constituent Big Atoms.

```cpp
void applyForceToComposite(Composite& composite, float3 force) {
    for (auto& atom : composite.atoms) {
        atom.velocity += force * (composite.deltaTime / composite.totalMass);
    }
}
```

## Conclusion

This approach allows for dynamic modeling of composites in the simulation:
- Big Atoms remain the fundamental entities.
- Virtual springs create temporary bonds, forming composites.
- Disjoint sets efficiently manage clustering.
- Composite properties (temperature, rotation) emerge from constituent Big Atoms.
- Bounding volumes provide quick approximate interactions between composites.

The system maintains flexibility, allowing for both fine-grained interactions between individual Big Atoms and efficient handling of larger-scale composite behaviors.

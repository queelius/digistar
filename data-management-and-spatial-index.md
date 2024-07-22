

## Data Management and Spatial Indexing


### Spatial Indexing

Many forces and intearctions in physics simulations depend on the distance
between objects. To calculate these distances efficiently, we can use spatial
indexing to quickly find nearby objects. This is especially useful when the
number of objects is large, as in our large-scale Big Atom simulation.


##### OctreeNode Structure

```cpp
struct OctreeNode {
    // etc, not sure what properties we need, and this is older code anyway
    // we need to be able to use it to compute many different forces.
    // interactions, and queries.
    float3 center;
    float size;
    BoundingBox boundingBox;
    float3 aggregatedMassCenter;
    float aggregatedMass;
    float3 aggregatedChargeCenter;
    float aggregatedCharge;
    float3 averageVelocity;
    float3 totalMomentum;
    float3 angularMomentum;
    float totalKineticEnergy;
    float boundingRadius;

    float3 minBounds;
    float3 maxBounds;
    std::vector<int> atomIndices;
    // or: std::vector<BigAtom*> atoms;

    OctreeNode* children[8];
};
```

Here is a preliminary implementation of the Octree class. It supports
insertion of BigAtoms and querying for neighbors and ranges. The actual
implementation may vary based on the specific requirements of the simulation,
and this implementation is likely to both be optimized and greatly expanded
as the needs of the simulation become more clear.

```cpp
class Octree {
public:
    Octree(const std::vector<BigAtom>& atoms, float3 min, float3 max);
    void insert(const BigAtom& atom);
    std::vector<int> queryRange(const float3& min, const float3& max);
    std::vector<int> queryNeighbors(const float3& position, float radius);

private:

    OctreeNode* root;
    void insert(Node* node, const BigAtom& atom, int depth);
    void queryRange(Node* node, const float3& min, const float3& max, std::vector<int>& result);
    void queryNeighbors(Node* node, const float3& position, float radius, std::vector<int>& result);
};
```

### Data Handling Strategies

#### Struct of Arrays (SoA) vs Array of Structs (AoS)

When designing data structures for GPU-accelerated simulations, the choice between Struct of Arrays (SoA) and Array of Structs (AoS) is crucial for performance.

##### Array of Structs (AoS)

Each element in the array is a struct containing multiple fields.

```cpp
struct BigAtom {
    float3 position;
    float3 velocity;
    float mass;
    float radius;
    // many more properties...
};

BigAtom atoms[N];
```

**Advantages**:
- Simplicity in managing individual objects.
- Easier to pass around single objects.

**Disadvantages**:
- Less efficient memory access patterns on GPUs due to non-coalesced memory accesses.
- May lead to lower cache utilization.

#### Struct of Arrays (SoA)

Each field of the struct is a separate array.

```cpp
struct BigAtomSoA {
    float3* positions;
    float3* velocities;
    float* masses;
    float* radii;
    // many more arrays...
};
BigAtomSoA atomsSoA;
```

**Advantages**:
- Better memory access patterns on GPUs, leading to coalesced memory accesses.
- Higher cache utilization and potentially better performance.

**Disadvantages**:
- More complex to manage and update individual objects.
- Requires additional bookkeeping for synchronization.

### Choosing Between AoS and SoA

**When to Use AoS**:
- When operations on individual atoms are independent and localized.
- When ease of use and simplicity is prioritized over raw performance.

**When to Use SoA**:
- When operations involve processing large arrays of data in a uniform manner.
- When performance is critical, especially for GPU acceleration.

### Implementation of SoA

```cpp
struct BigAtomSoA {
    float3* positions;
    float3* velocities;
    float* masses;
    float* charges;
    float* radii;
};

void initializeBigAtomSoA(BigAtomSoA& atomsSoA, int numAtoms) {
    atomsSoA.positions = (float3*)malloc(numAtoms * sizeof(float3));
    atomsSoA.velocities = (float3*)malloc(numAtoms * sizeof(float3));
    atomsSoA.masses = (float*)malloc(numAtoms * sizeof(float));
    atomsSoA.charges = (float*)malloc(numAtoms * sizeof(float));
    atomsSoA.radii = (float*)malloc(numAtoms * sizeof(float));
    // Initialize other arrays as needed
}

void freeBigAtomSoA(BigAtomSoA& atomsSoA) {
    free(atomsSoA.positions);
    free(atomsSoA.velocities);
    free(atomsSoA.masses);
    free(atomsSoA.charges);
    free(atomsSoA.radii);
    // Free other arrays as needed
}
```



## Spatial Indexing

Many forces and intearctions in physics simulations depend on the distance
between objects. To calculate these distances efficiently, we can use spatial
indexing to quickly find nearby objects. This is especially useful when the
number of objects is large, as in our large-scale Big Atom simulation.


##### OctreeNode Structure

```cpp
struct OctreeNode {
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
    OctreeNode* children[8];
    std::vector<BigAtom*> atoms;
};
```



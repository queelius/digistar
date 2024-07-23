### Grid-Based Spatial Indexing

#### Overview
We consider replacing the octree with a grid-based spatial index. Each server's region is divided into a grid of cells, and particles within each cell are managed locally. This approach may be easier to update and could be more efficient on a GPU.

#### Grid-Based Index vs. Octree
- **Grid-Based Index**: Simplifies updates and may offer better GPU performance. In each grid cell, we perform pair-wise computations, and for adjacent cells, we handle interactions in detail. Distant cells are treated as point masses/charges.
- **Octree**: A recursive tree-based index providing hierarchical spatial partitioning, which may be more complex to update but offers efficient querying for certain types of spatial computations.

#### Implementation Details
In a grid-based approach, each cell is responsible for:
1. **Pair-wise Computations**: Handle interactions between particles within the cell.
2. **Adjacent Cells**: Compute interactions with particles in adjacent cells.
3. **Distant Cells**: Treat distant cells as single point masses/charges.

#### Experimental Validation
We will conduct experiments to compare the efficiency and performance of the grid-based approach versus the octree. Metrics to consider include:
- Update times
- Computation times for interactions
- Memory usage
- Scalability on GPUs

### Conclusion
Implementing a toroidal space topology and distributing the simulation across multiple servers with a potential grid-based spatial index allows for efficient and scalable n-body simulations. By managing boundaries through wrap-around and inter-server communication, we maintain a continuous and unbounded simulation space. This approach leverages the strengths of both spatial indexing and networked computation, providing a robust framework for large-scale simulations.

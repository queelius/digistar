### Toroidal Space Topology

#### Overview
In our simulation, we employ a toroidal space topology, where the simulation space wraps around in all dimensions. This design avoids traditional boundary conditions, simplifying computations and ensuring a continuous, unbounded environment. Objects exiting one side of the space re-enter from the opposite side, maintaining consistent interactions across the entire simulation.

#### Motivation
The toroidal topology is advantageous for simulations with many interacting particles, such as our n-body simulation with big atoms and various forces. It eliminates edge effects and ensures all particles remain within the simulation space, reducing computational overhead and simplifying boundary management.

#### Modifying Distance Calculations
In a toroidal space, distances between points are calculated considering the wrap-around. For points \(A(x_1, y_1, z_1)\) and \(B(x_2, y_2, z_2)\) in a 3D toroidal space with dimensions \(L_x, L_y, L_z\):

\[
\Delta x = \min(|x_2 - x_1|, L_x - |x_2 - x_1|)
\]
\[
\Delta y = \min(|y_2 - y_1|, L_y - |y_2 - y_1|)
\]
\[
\Delta z = \min(|z_2 - z_1|, L_z - |z_2 - z_1|)
\]

The distance \(d\) is given by:

\[
d = \sqrt{\Delta x^2 + \Delta y^2 + \Delta z^2}
\]

This ensures the shortest path through the toroidal space is always considered.

#### Simplification of Boundary Conditions
Using a toroidal topology avoids complex boundary conditions. Particles naturally wrap around the edges, maintaining continuous interaction across the entire simulation space. This also ensures big atoms always remain within bounds, reducing computational overhead.

#### Visualization
Here is a 2D representation of a toroidal space to illustrate the concept:

![Toroidal Space Visualization](https://i.imgur.com/9sZ5J2K.png)

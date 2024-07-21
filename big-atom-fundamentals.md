### Design Document: Fundamental Properties and Relational Properties of Big Atoms

#### Section: Fundamental Properties and Relational Properties of Big Atoms

In this section, we detail the fundamental properties of big atoms and their derived relational properties. These properties form the foundation of our space sandbox simulation, enabling realistic interactions and dynamic behavior on the scale of large spaceships to multi-star systems. The goal is to ensure that the physics and dynamics of the simulation are both interesting and fun for players, allowing them to explore, battle, and interact within a consistent and engaging environment.

#### Fundamental Properties of Big Atoms

Each big atom in the simulation possesses a set of fundamental properties that define its basic characteristics and behavior. These properties include both observable and internal attributes that influence the interactions and dynamics within the simulation.

1. **Mass (\(m\))**
   - The amount of matter in the big atom.
   - Essential for calculating gravitational forces and inertia.

2. **Radius (\(r\))**
   - The size of the big atom.
   - Important for collision detection, interaction ranges, and volume calculations.

3. **Position (\(\vec{p}\))**
   - The spatial location of the big atom in the simulation.
   - Used for calculating distances and interactions with other big atoms.

4. **Velocity (\(\vec{v}\))**
   - The rate of change of the big atom's position.
   - Necessary for dynamic simulations and predicting future positions.

5. **Charge (\(q\))**
   - The electric charge of the big atom.
   - Influences electrostatic interactions and potential fields.

6. **Interaction Vector (\(\vec{i}\))**
   - A vector representing the interaction characteristics of the big atom.
   - Used to modulate interaction properties like bond strength and distance.

7. **Rotational State (\(\vec{\omega}\))**
   - The angular velocity of the big atom.
   - Determines the rotational energy and dynamics around its axis.

8. **Internal Temperature (\(T_{\text{internal}}\))**
   - Represents the internal kinetic energy or thermal state of the big atom.
   - Influences the repulsive potential energy field due to thermal effects.

9. **Magnetic Moment (\(\vec{\mu}\))**
   - Represents the magnetic properties of the big atom.
   - Important for simulating magnetic interactions and fields.

10. **Resource Content (\(\vec{R}\))**
    - A vector representing quantities of various resources within the big atom.
    - Examples include metals, carbon, and silicon.

#### Relational Properties

Relational properties are derived from the interactions and combinations of fundamental properties, particularly within clusters of big atoms. These properties are essential for understanding the dynamic and interactive behavior of big atoms in the simulation.

1. **Density (\(\rho\))**
   - Calculated as:
     \[
     \rho = \frac{m}{\frac{4}{3} \pi r^3}
     \]

2. **Rotational Energy (\(E_{\text{rot}}\))**
   - Calculated as:
     \[
     E_{\text{rot}} = \frac{1}{5} m r^2 \omega^2
     \]

3. **Internal Kinetic Energy (\(E_{\text{internal}}\))**
   - Calculated as:
     \[
     E_{\text{internal}} = k_B T_{\text{internal}}
     \]
   - Where \(k_B\) is a proportionality constant related to thermal energy.

4. **Equilibrium Bond Distance (\(r_e\))**
   - Modulated by the interaction vectors and fundamental properties:
     \[
     r_e = (r_1 + r_2) \cdot (1 + \text{cosine\_similarity}(\vec{i_1}, \vec{i_2}) \cdot k_r)
     \]
   - Where \(k_r\) is a scaling factor.

5. **Depth of the Potential Well (\(D_e\))**
   - A constant base value adjusted by interaction vectors:
     \[
     D_e = D_{e,\text{base}} \cdot (1 + \text{cosine\_similarity}(\vec{i_1}, \vec{i_2}) \cdot k_D)
     \]

6. **Width of the Potential Well (\(a\))**
   - A constant base value adjusted by interaction vectors:
     \[
     a = a_{\text{base}} \cdot (1 + \text{cosine\_similarity}(\vec{i_1}, \vec{i_2}) \cdot k_a)
     \]

#### Cluster Properties

When big atoms form clusters based on their interactions (e.g., via the Morse potential), we can define cluster-level properties that influence the collective behavior and dynamics.

1. **Structural Integrity (SI)**
   - Determined by the number and strength of bonds within a cluster.
   - High structural integrity implies a well-connected and stable cluster.

2. **Cluster Temperature**
   - Internal kinetic energy of the cluster:
     \[
     T_{\text{cluster}} = \sum ||(\vec{v_i} - \vec{v_{\text{cluster}}})||^2
     \]
   - Where \(\vec{v_{\text{cluster}}}\) is the average velocity of the cluster.

3. **Cluster Mass and Velocity**
   - Total mass and center-of-mass velocity of the cluster.

4. **Cluster Resource Content**
   - Sum of the resource contents of the big atoms within the cluster.

### Summary

By focusing on fundamental properties and deriving relational properties through interactions and clustering, we maintain flexibility and realism in our space sandbox simulation. This approach supports emergent behaviors, allowing players to experience a rich and engaging simulation environment. The fundamental properties of big atoms include mass, radius, position, velocity, charge, interaction vector, rotational state, internal temperature, magnetic moment, and resource content. These properties provide the basis for modeling complex interactions and dynamics, ensuring both physical realism and exciting gameplay.

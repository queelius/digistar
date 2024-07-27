### Radiation Pressure Model

In this model, each big atom emits radiation energy based on its internal energy \( U \), radius \( R \), and mass \( m \). The emitted energy influences nearby objects by exerting radiation pressure, modeled as a potential energy function. The radius \( R \) of a big atom dynamically changes based on its internal energy and mass, allowing for continuous contraction without a predefined maximum energy limit.

### Radiation Emission and Dynamic Radius

1. **Dynamic Radius Model**:
   - The radius \( R \) of a big atom changes as a function of its internal energy \( U \):
     \[ R(U, m) = R_0 \left(1 - \beta \frac{U}{\eta m}\right) \]
   - Here, \( R_0 \) is the base radius, \(\beta\) is a small constant controlling the degree of shrinkage, and \(\eta\) is a proportionality constant for energy density.

2. **Energy Emission Rate**:
   - The energy emitted per time step \(\Delta E\) is modeled as:
     \[ \Delta E = \alpha \frac{U^4}{R(U, m)^{10}} \Delta t \]
   - Here, \(\alpha\) is a proportionality constant simplifying the computation, and \(\Delta t\) is the time step duration.

3. **Internal Energy Update**:
   - For each big atom, the internal energy \( U \) is updated by deducting \(\Delta E\):
     \[ U_{\text{new}} = U - \Delta E \]

### Radiation Pressure Influence

1. **Radiation Pressure**:
   - The radiation pressure \( P \) of an emitting big atom with internal energy \( U \) and dynamic radius \( R(U, m) \) is:
     \[ P = k \frac{U}{R(U, m)^3} \]
   - Here, \( k \) is a constant relating the internal energy and radius to the radiation pressure.

2. **Force on Nearby Objects**:
   - The force \( F_{ij} \) on an object \( j \) at distance \( r_{ij} \) from the emitting atom \( i \) with radius \( R_j \) is calculated as:
     \[ F_{ij} = P_i \cdot A_j \cdot \left( \frac{R_j}{R(U_i, m_i) + r_{ij}} \right)^2 \]
   - Where \( A_j = \pi R_j^2 \) is the cross-sectional area of object \( j \).

### Composite Structures

Composite structures made of big atoms can extend in space such that different parts of the structure are at varying distances from the radiating big atom. This setup can result in differential radiation pressure across the structure, potentially applying a torque and causing rotational dynamics.

### Implementation Steps

1. **Define Parameters**:
   - \(\alpha\): Proportionality constant for energy emission.
   - \( k \): Proportionality constant for radiation pressure.
   - \(\beta\): Constant controlling the degree of shrinkage.
   - \(\eta\): Proportionality constant for energy density.
   - \( R_0 \): Base radius.
   - \(\Delta t\): Time step duration.

2. **Initialize Big Atoms**:
   - Each big atom has initial internal energy \( U \), position \( \vec{r} \), velocity \( \vec{v} \), mass \( m \), and radius \( R \).

3. **Update Loop**:
   - For each time step:
     1. **Energy Emission**:
        - Calculate the dynamic radius \( R(U, m) \) for each big atom:
          \[ R(U, m) = R_0 \left(1 - \beta \frac{U}{\eta m}\right) \]
        - Calculate \(\Delta E\) for each big atom:
          \[ \Delta E = \alpha \frac{U^4}{R(U, m)^{10}} \Delta t \]
        - Update the internal energy \( U \):
          \[ U_{\text{new}} = U - \Delta E \]
     2. **Radiation Pressure**:
        - Calculate the radiation pressure \( P \) for each emitting atom:
          \[ P = k \frac{U}{R(U, m)^3} \]
        - Calculate the force \( F_{ij} \) on each nearby object \( j \):
          \[ F_{ij} = P_i \cdot A_j \cdot \left( \frac{R_j}{R(U_i, m_i) + r_{ij}} \right)^2 \]
     3. **Velocity and Position Update**:
        - Update the velocities and positions of the big atoms based on the calculated forces.
        - For composite structures, compute the net force and torque, and update the rotational dynamics accordingly.

### Example Calculation

1. **Initial State**:
   - Object A: \( U_A = 1000 \, \text{J} \), \( R_A = 1 \, \text{m} \), \( m_A = 10^6 \, \text{kg} \)
   - Object B: \( U_B = 2000 \, \text{J} \), \( R_B = 0.5 \, \text{m} \), \( m_B = 10^5 \, \text{kg} \)

2. **Dynamic Radius**:
   - Calculate the dynamic radius for Object A:
     \[ R_A(U) = 1 \, \text{m} \left(1 - \beta \frac{1000 \, \text{J}}{\eta (10^6 \, \text{kg})}\right) \]
   - Calculate the dynamic radius for Object B:
     \[ R_B(U) = 0.5 \, \text{m} \left(1 - \beta \frac{2000 \, \text{J}}{\eta (10^5 \, \text{kg})}\right) \]

3. **Radiation Emission**:
   - Calculate \(\Delta E\) for Object A:
     \[ \Delta E_A = \alpha \frac{1000^4}{R_A(U)^{10}} \Delta t \]
   - Calculate \(\Delta E\) for Object B:
     \[ \Delta E_B = \alpha \frac{2000^4}{R_B(U)^{10}} \Delta t \]

4. **Radiation Pressure**:
   - Calculate the radiation pressure for Object A:
     \[ P_A = k \frac{U_A}{R_A(U)^3} \]

5. **Force on Object B**:
   - Calculate the force \( F_{AB} \) on Object B:
     \[ F_{AB} = P_A \cdot A_B \cdot \left( \frac{R_B}{R_A(U) + r_{AB}} \right)^2 \]

6. **Update Velocities and Positions**:
   - Update the velocities and positions based on the calculated forces.
   - For composite structures, compute the net force and torque to update rotational dynamics.

### Conclusion

This simplified model provides a robust and scalable approach for simulating radiation pressure and thermal contraction in a sandbox space simulation. By allowing the radius to dynamically change based on internal energy and mass, and by modeling radiation pressure as a potential energy function, the simulation can capture complex interactions and interesting behaviors. This approach ensures that high-energy states can lead to significant contraction, potentially down to relativistic scales, adding another layer of realism and complexity to the simulation.

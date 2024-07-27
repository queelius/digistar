# Basic Units

This section outlines the base units for our sandbox simulation. It details the unit choices for mass, distance, and time, provides conversions for key astronomical quantities, and discusses the rationale behind these choices. The document focuses on ensuring stability, numerical precision, and realistic system design. Additionally, it provides relevant physical constants and their conversions to these units.

### Basic Units

1. **Mass Unit (mu)**: $1 \text{ mu} = 10^6 \text{ kg}$
2. **Distance Unit (du)**: $1 \text{ du} = 10^6 \text{ meters}$
3. **Time Unit (tu)**: $1 \text{ tu} = 10^6 \text{ seconds}$

### Key Conversions

#### Distance

- 1 Astronomical Unit (AU): $1.5 \times 10^1 \text{ du}$
- 1 Earth Radius: $6.371 \times 10^{-4} \text{ du} \approx 10^{-3} \text{ du}$
- 1 Moon Radius: $1.737 \times 10^{-4} \text{ du} \approx 10^{-3} \text{ du}$
- 1 Light Year: $9.461 \times 10^5 \text{ du} \approx 10^6 \text{ du}$
- 1 Parsec: $3.086 \times 10^6 \text{ du} \approx 3 \times 10^6 \text{ du}$

#### Mass

- 1 Earth Mass: $5.972 \times 10^4 \text{ mu} \approx 6 \times 10^4 \text{ mu}$
- 1 Solar Mass: $1.989 \times 10^{10} \text{ mu} \approx 2 \times 10^{10} \text{ mu}$
- 1 Jupiter Mass: $1.898 \times 10^7 \text{ mu} \approx 2 \times 10^7 \text{ mu}$
- 1 Moon Mass: $7.34 \times 10^2 \text{ mu} \approx 10^3 \text{ mu}$

#### Time

- 1 Year: $3.154 \times 10^1 \text{ tu} \approx 30 \text{ tu}$
- 1 Day: $8.64 \times 10^{-2} \text{ tu} \approx 0.1 \text{ tu}$
- 1 Hour: $3.6 \times 10^{-3} \text{ tu}$
- 1 Minute: $6 \times 10^{-5} \text{ tu}$
- 1 Second: $1 \times 10^{-6} \text{ tu}$

### Example Objects

#### Gigantic Battleship

1. **Length**: 100 miles $\approx 1.6 \times 10^{-2} \text{ du}$
2. **Mass**: 200 aircraft carriers (assuming each carrier is $10^8 \text{ kg}$) $= 2 \times 10^{-4} \text{ mu}$

#### Dyson Sphere

1. **Radius**: 1 AU $\approx 15 \text{ du}$
2. **Surface Area**: $4\pi r^2 \approx 3 \times 10^3 \text{ du}^2$
3. **Mass**: Assuming 1% of Earth's mass $\approx 6 \times 10^2 \text{ mu}$

### Relevant Physical Constants

#### Newton's Universal Gravitational Constant (G)

- Standard Units: $G = 6.6743 \times 10^{-11} \text{ m}^3 
  \text{ kg}^{-1} \text{ s}^{-2}$
- In preferred units:

$$
\begin{align*}
G_{\text{new}} &=
G \times \left(\frac{1 \text{ du}}{10^{10} \text{ m}}\right)^3 \times \left(\frac{10^{20} \text{ kg}}{1 \text{ mu}}\right) \times \left(\frac{10^6 \text{ s}}{1 \text{ tu}}\right)^2\\
&= 6.6743 \times 10^{-11} \times 10^{-30} \times 10^{20} \times 10^{12} \\
&\approx 6.6743 \times 10^{-9} \text{ du}^3 \text{ mu}^{-1} \text{ tu}^{-2}\\
&\approx 10^{-8} \text{ du}^3 \text{ mu}^{-1} \text{ tu}^{-2}
\end{align*}
$$

#### Speed of Light (c)

- Standard Units: $c = 2.998 \times 10^8 \text{ m/s}$
- In preferred units:

$$
\begin{align*}
c_{\text{new}} &=
c \times \frac{\text{tu}}{\text{du}} \\
&= 2.998 \times 10^8 \times \frac{10^6}{10^{10}} \\
&\approx 3 \times 10^4 \text{ du/tu}
\end{align*}
$$

### Derived Units

#### Velocity Unit (vu)

- Standard Units: $\text{m/s}$
- In preferred units: $1 \text{vu} = \frac{1 \text{ du}}{1 \text{ tu}} = 10^4 \text{ m/s}$

#### Acceleration Unit (au)

- Standard Units: $\text{m/s}^2$
- In preferred units: $1 \text{ au} = \frac{1 \text{ du}}{1 \text{ tu}^2} = 10^{-2} \text{ m/s}^2$

#### Force Unit (fu)

- Standard Units: $\text{N} = \text{kg} \cdot \text{m/s}^2$
- In preferred units: $1 \text{ fu} = 1 \text{ mu} \cdot \frac{1 \text{ du}}{1 \text{ tu}^2} = 10^{20} \text{ kg} \cdot 10^{-2} \text{ m/s}^2 = 10^{18} \text{ N}$

#### Energy Unit (eu)

- Standard Units: $\text{J} = \text{kg} \cdot \text{m}^2 / \text{s}^2$
- In preferred units: $1 \text{ eu} = 1 \text{ mu} \cdot \frac{1 \text{ du}^2}{1 \text{ tu}^2} = 10^{20} \text{ kg} \cdot 10^{20} \text{ m}^2 / 10^{12} \text{ s}^2 = 10^{28} \text{ J}$

### Complete Example: Earth

#### Mass of Earth

- Standard Units: $5.972 \times 10^{24} \text{ kg}$
- In preferred units: $5.972 \times 10^4 \text{ mu}$

#### Radius of Earth

- Standard Units: $6.371 \times 10^6 \text{ m}$
- In preferred units: $6.371 \times 10^{-4} \text{ du}$

#### Surface Gravity of Earth

- Standard Units: $g = \frac{G \cdot M}{R^2}$
- Using standard values:

$$g = \frac{6.67430 \times 10^{-11} \times 5.972 \times 10^{24}}{(6.371 \times 10^6)^2} = 9.81 \text{ m/s}^2$$

- In preferred units:

$$
\begin{align*}
g_{\text{new}} &= \frac{G_{\text{new}} \cdot M_{\text{new}}}{R_{\text{new}}^2} \\
&= \frac{6.67430 \times 10^{-9} \times 5.972 \times 10^4}{(6.371 \times 10^{-4})^2} \\
&\approx 981 \text{ du/tu}^2
\end{align*}
$$

### Additional Examples

#### Orbital Period

For a circular orbit, the period $T$ is given by:

$$T = 2\pi\sqrt{\frac{r^3}{GM}}$$

Where $r$ is the orbital radius, $G$ is the gravitational constant, and $M$ is the mass of the central body.

In our preferred units:

$$T_{\text{new}} = 2\pi\sqrt{\frac{r_{\text{new}}^3}{G_{\text{new}}M_{\text{new}}}}$$

#### Escape Velocity

The escape velocity $v_e$ is given by:

$$v_e = \sqrt{\frac{2GM}{r}}$$

In our preferred units:

$$v_{e,\text{new}} = \sqrt{\frac{2G_{\text{new}}M_{\text{new}}}{r_{\text{new}}}}$$

### Rationale for Unit Choices

1. **Stability**: These units ensure that values used in the simulation remain within a manageable range, avoiding extremely large or small numbers that could lead to numerical instability or overflow/underflow issues.

2. **Avoiding Round-Off Errors**: By keeping the units appropriately scaled, we minimize the risk of round-off errors that can accumulate in large-scale simulations.

3. **Realistic System Design**: The chosen units allow for a realistic representation of celestial bodies and man-made objects, from small asteroids to massive stars and hypothetical megastructures.

4. **Scalability**: These units allow for the simulation of a wide range of objects without needing to frequently adjust the scale, making the simulation flexible and adaptable to different scenarios.

5. **Compatibility with Astrophysical Phenomena**: The units are well-suited for simulating large-scale astrophysical phenomena while excluding constants relevant only at quantum scales.

### Conclusion

The chosen units of mass, distance, and time provide a robust foundation for the multi-star system sandbox simulation. They ensure numerical stability and precision while allowing for realistic and scalable system design. By adhering to these units, we can create a versatile and accurate simulation environment that can accommodate various astronomical and artificial objects, facilitating meaningful exploration and experimentation within the simulation.

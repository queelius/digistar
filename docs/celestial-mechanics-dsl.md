### Celestial Mechanics DSL

#### Overview

The Celestial Mechanics Domain-Specific Language (DSL) provides a high-level abstraction for specifying celestial systems. This DSL allows for the definition of celestial bodies and their initial states in terms of intrinsic and extrinsic properties, enabling the setup of complex n-body simulations. The DSL supports incremental loading, allowing users to add new bodies or systems into an existing simulation without affecting already defined entities.

#### Structure

The JSON file consists of a flat structure where each celestial body is defined at the top level. Each body can optionally specify a parent body, indicating its relationship in the system. If a parent is not specified, the body is considered to be in relation to a fixed reference frame, typically the origin of the simulation space.

#### Intrinsic Properties

Intrinsic properties are inherent to the celestial bodies and do not change over time within the simulation. These include mass, charge, radius, and color.

```json
{
  "Sun": {
    "intrinsic_properties": {
      "mass": 1.989e30,
      "radius": 6.9634e8,
      "color": [1.0, 1.0, 0.0],
      "charge": 0.0
    }
  }
}
```

- `mass`: The mass of the body in kilograms.
- `radius`: The radius of the body.
- `color`: The color of the body represented as an RGB array.
- `charge`: The electric charge of the body (optional).

#### State Properties

State properties define the initial conditions of the celestial bodies. These include position, velocity, and orbital parameters. State properties can be specified using different methods, including relative positions, orbital parameters, and angle-based specifications.

#### Methods of Specifying State Properties

1. **Orbit**:
   - `semi_major_axis`: The semi-major axis of the orbit in meters.
   - `eccentricity`: The eccentricity of the orbit.
   - `inclination`: The inclination of the orbit in radians.
   - `longitude_of_ascending_node`: The longitude of the ascending node in radians.
   - `argument_of_perigee`: The argument of perigee in radians.
   - `starting_angle` (or `true_anomaly`): The initial angle within the orbit in radians.

2. **Relative Position and Velocity**:
   - `position`: The initial position of the body as a 3D vector [x, y, z] relative to the parent.
   - `velocity`: The initial velocity of the body as a 3D vector [vx, vy, vz] relative to the parent.

3. **Angle-Based Specification**:
   - `distance`: The radial distance from the parent body.
   - `angle`: The angular position from a reference direction.
   - `velocity_magnitude`: The magnitude of the velocity vector.
   - `velocity_angle`: The direction of the velocity vector relative to a reference direction.

### Example Structure

Here is an example structure reflecting these considerations:

```json
{
  "Sun": {
    "intrinsic_properties": {
      "mass": 1.989e30,
      "radius": 6.9634e8,
      "color": [1.0, 1.0, 0.0],
      "charge": 0.0
    },
    "state": {
      "relative_position": {
        "position": [0, 0, 0],
        "velocity": [0, 0, 0]
      }
    }
  },
  "Earth": {
    "parent": "Sun",
    "intrinsic_properties": {
      "mass": 5.972e24,
      "radius": 6.371e6,
      "color": [0.0, 0.5, 1.0],
      "charge": 0.0
    },
    "state": {
      "orbit": {
        "semi_major_axis": 1.496e11,
        "eccentricity": 0.0167,
        "inclination": 0.00005,
        "longitude_of_ascending_node": -11.26064,
        "argument_of_perigee": 102.94719,
        "starting_angle": 0.0
      }
    }
  },
  "Moon": {
    "parent": "Earth",
    "intrinsic_properties": {
      "mass": 7.342e22,
      "radius": 1.737e6,
      "color": [0.6, 0.6, 0.6],
      "charge": 0.0
    },
    "state": {
      "orbit": {
        "semi_major_axis": 3.844e8,
        "eccentricity": 0.0549,
        "inclination": 0.089,
        "longitude_of_ascending_node": 125.08,
        "argument_of_perigee": 318.15,
        "starting_angle": 0.0
      }
    }
  },
  "Beetlejuice": {
    "intrinsic_properties": {
      "mass": 2.78e31,
      "radius": 1.234e12,
      "color": [1.0, 0.0, 0.0],
      "charge": 0.0
    },
    "state": {
      "relative_position": {
        "position": [6.0e12, 2.0e12, 0.0],
        "velocity": [0.0, 0.0, 0.0]
      }
    }
  }
}
```

### Incremental Loading

The DSL supports incremental loading, allowing new bodies or systems to be added to an existing simulation without disrupting already defined entities. If a parent body already exists in the simulation, new children can be added under that parent, and their states will be defined relative to the existing parent. This structure ensures that each body can be loaded independently and in any order, provided that dependencies are respected.

### Generative DSL

While the Celestial Mechanics DSL focuses on specifying systems, another subsection, the Generative DSL, will cover the generation of random systems such as asteroid belts, star systems, and multi-star systems. This generative approach can leverage the Celestial Mechanics DSL functions to define the generated entities.

### Example: Minimal Circular Orbit

Here's a minimal example specifying a body with orbital parameters for uniform circular motion:

```json
{
  "Sun": {
    "intrinsic_properties": {
      "mass": 1.989e30,
      "radius": 6.9634e8,
      "color": [1.0, 1.0, 0.0],
      "charge": 0.0
    },
    "state": {
      "relative_position": {
        "position": [0, 0, 0],
        "velocity": [0, 0, 0]
      }
    }
  },
  "Earth": {
    "parent": "Sun",
    "intrinsic_properties": {
      "mass": 5.972e24,
      "radius": 6.371e6,
      "color": [0.0, 0.5, 1.0],
      "charge": 0.0
    },
    "state": {
      "orbit": {
        "semi_major_axis": 1.496e11,
        "eccentricity": 0.0,
        "inclination": 0.0,
        "longitude_of_ascending_node": 0.0,
        "argument_of_perigee": 0.0,
        "starting_angle": 0.0
      }
    }
  }
}
```

This structure ensures flexibility and extensibility while providing a clear distinction between intrinsic properties and state properties. The Celestial Mechanics DSL facilitates the high-level specification of celestial systems, enabling complex simulations with ease.

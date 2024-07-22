# Space Simulation Sandbox Design Document

## 1. Introduction

The Space Simulation Sandbox is an ambitious project aimed at creating a large-scale, interactive space simulation sandbox. Our primary goal is to generate complex, emergent behaviors and structures from simple, fundamental building blocks and rules.

### Purpose
- To create a highly scalable space simulation that can handle millions of entities
- To provide a platform for exploring emergent phenomena in a space-like sandbox environment
- To enable users to interact with and build within the simulation

### Key Principles
1. Simplicity at the core: Use simple, atomic entities as the basic building blocks
2. Emergent complexity: Allow complex behaviors to arise from simple rules and interactions
3. Scalability: Design the system to efficiently handle millions of entities
4. Extensibility: Create a flexible framework that can be easily expanded and modified
5. User engagement: Provide tools for users to explore, build, and interact within the simulation
6. Layered complexity: While based on simple atomic entities, the system should support emergent properties and complex force fields at higher levels of organization

### Motivation
Our approach is inspired by the idea that complex systems in the universe often arise from relatively simple fundamental particles and forces. By focusing on simple atomic entities and basic force interactions, we aim to:

1. Maximize computational efficiency, allowing for large-scale simulations
2. Create a system where emergent behaviors are a natural outcome, rather than explicitly programmed
3. Provide a flexible foundation that can be easily extended or modified to explore different phenomena
4. Enable users to understand and interact with the simulation at a fundamental level
5. Create a foundation that supports not just simple particle interactions, but also emergent phenomena and complex force fields arising from aggregations of particles

## 2. Core Simulation Components

### 2.1 Atomic Entities

Atomic entities are the fundamental building blocks of our simulation. They are designed to be as simple as possible while still allowing for complex emergent behaviors when combined in large numbers.

#### Properties
- Mass: Determines the entity's influence on gravitational interactions
- Position: 3D coordinates in the simulation space
- Velocity: 3D vector representing the entity's motion
- Color: A property that can influence certain interactions (e.g., spring formation)

#### Data Structure
We use a Struct of Arrays (SoA) approach for storing entity data:

```cpp
struct AtomicEntities {
    float* mass;    // Array of masses
    float* pos_x;   // Array of x-coordinates
    float* pos_y;   // Array of y-coordinates
    float* pos_z;   // Array of z-coordinates
    float* vel_x;   // Array of x-velocities
    float* vel_y;   // Array of y-velocities
    float* vel_z;   // Array of z-velocities
    int* color;     // Array of color values
};
```

This structure is optimized for GPU processing, allowing for efficient parallel computations and memory access patterns.

#### Motivation for Simplicity
1. Computational Efficiency: Simple entities with few properties are faster to process, allowing for larger simulations.
2. Emergent Complexity: Complex behaviors can emerge from the interactions of many simple entities, rather than being encoded in complex individual entities.
3. Flexibility: Simple entities can be easily combined and reconfigured to create a wide range of structures and behaviors.
4. Scalability: The simplicity of atomic entities makes it easier to scale the simulation to millions of particles.
5. Foundation for Emergence: The simplicity of atomic entities allows for the emergence of complex properties and behaviors at higher levels of organization, such as temperature gradients or radiation fields in composite structures.

### 2.2 Forces

Forces are the fundamental interactions between atomic entities. They drive the dynamics of the simulation and are the primary source of emergent behaviors.

#### Types of Forces

1. Gravity
   - Long-range force acting between all pairs of entities
   - Proportional to the product of masses and inversely proportional to the square of the distance
   - Formula: F = G * (m1 * m2) / r^2, where G is the gravitational constant

2. Repulsion (Collision Handling)
   - Short-range force that prevents entities from overlapping
   - Activates when the distance between entities is less than the sum of their radii
   - Formula: F = k * overlap, where k is a repulsion constant and overlap is the difference between the sum of radii and the actual distance

3. Springs
   - Dynamic connections between pairs of entities
   - Can form and break based on certain conditions (e.g., proximity, shared color)
   - Formula: F = -k * (x - x0) - c * v, where k is the spring constant, x is the current length, x0 is the equilibrium length, c is the damping coefficient, and v is the relative velocity

4. Additional Force Fields (Preview)
   - While not directly a property of atomic entities, the simulation architecture will support additional force fields
   - These fields may arise from emergent properties of composite entities (e.g., temperature gradients)
   - Implementation details will be discussed in later sections on composite entities and emergent behaviors

#### Implementation Approach

1. Gravity:
   - Use a grid-based approximation method for efficient computation
   - Divide the simulation space into cells and compute cell-to-cell interactions
   - Update gravity forces periodically rather than every timestep for efficiency

2. Repulsion:
   - Use the same spatial partitioning as gravity for identifying potential collisions
   - Apply repulsion forces directly in the force calculation kernel

3. Springs:
   - Maintain a list of active springs
   - Update spring forces every timestep in a separate kernel or on CPU if the number of springs is relatively small

#### Motivation for Force Selection

1. Fundamental Interactions: Gravity and repulsion represent the most basic attractive and repulsive forces found in nature.
2. Emergent Structures: Springs allow for the formation of stable structures and composite entities, enabling more complex emergent behaviors.
3. Computational Feasibility: These forces can be efficiently computed for large numbers of entities, especially with appropriate approximation methods.
4. Extensibility: This set of basic forces provides a foundation that can be easily extended with additional force types if needed.
5. Extensibility to Complex Fields: While focusing on fundamental forces at the atomic level, this framework allows for the introduction of more complex force fields arising from emergent properties of composite entities in later stages of the simulation.


### 2.3 Composite Entities

[... previous sections remain unchanged ...]

### 2.2 Forces

Forces are the fundamental interactions that drive the dynamics of the simulation at the atomic level.

#### Types of Forces

1. Gravity
   - Long-range force acting between all pairs of entities
   - Proportional to the product of masses and inversely proportional to the square of the distance
   - Formula: F = G * (m1 * m2) / r^2, where G is the gravitational constant

2. Repulsion (Collision Handling)
   - Short-range force that prevents entities from overlapping
   - Activates when the distance between entities is less than the sum of their radii
   - Formula: F = k * overlap, where k is a repulsion constant and overlap is the difference between the sum of radii and the actual distance

3. Springs
   - Dynamic connections between pairs of entities
   - Can form and break based on certain conditions (e.g., proximity, shared color)
   - Formula: F = -k * (x - x0) - c * v, where k is the spring constant, x is the current length, x0 is the equilibrium length, c is the damping coefficient, and v is the relative velocity

4. Composite Entity-Generated Force Fields

   - Long-range forces generated by composite entities (e.g., temperature gradients)
   - Integrated into the same GPU simulation framework as fundamental forces
   - Affect atomic entities directly, influencing their behavior and interactions

#### Implementation Approach

1. Gravity:
   - Use a grid-based approximation method for efficient computation
   - Divide the simulation space into cells and compute cell-to-cell interactions
   - Update gravity forces periodically rather than every timestep for efficiency

2. Repulsion:
   - Use the same spatial partitioning as gravity for identifying potential collisions
   - Apply repulsion forces directly in the force calculation kernel

3. Springs:
   - Maintain a list of active springs
   - Update spring forces every timestep in a separate kernel or on CPU if the number of springs is relatively small

4. Composite Entity Force Fields:
   - Generate field data on CPU based on composite entity properties
   - Transfer field data to GPU, representing it in a grid or other efficient spatial structure
   - Incorporate field calculations into the same force computation kernels as gravity and other long-range forces
   - Update field data periodically, balancing accuracy with performance

#### Motivation for Force Selection

1. Fundamental Interactions: Gravity and repulsion represent the most basic attractive and repulsive forces found in nature.
2. Emergent Structures: Springs allow for the formation of stable structures and composite entities, enabling more complex emergent behaviors.
3. Computational Feasibility: These forces can be efficiently computed for large numbers of entities, especially with appropriate approximation methods.
4. Extensibility: This set of basic forces provides a foundation that can be easily extended with additional force types if needed.

### 2.3 Composite Entities

Composite entities are structures formed by collections of atomic entities connected through spring networks. They exhibit emergent properties and behaviors that arise from the collective interactions of their constituent parts.

#### Definition and Formation

- Composite entities are defined by connected components in the spring network
- They can form spontaneously based on proximity and shared properties of atomic entities
- The spring network can dynamically evolve, allowing composite entities to grow, shrink, or break apart

#### Emergent Properties

We briefly discuss several emergent properties that composite entities can exhibit, but we delay detailed implementation considerations to later sections.

1. Aggregate Mass

   Aggegrate (total) mass is the sum of masses of constituent entities:
   $M = \sum_i m_i$, where $m_i$ is the mass of the $i$-th entity.

   This is a straightforward property of composite entities. It does not induce a gravitational field, since mass is a fundamental property of atomic entities already.

2. Center of Mass
  
   Center of mass is the weighted average position of all constituents based on their masses: $\bar{r} = \sum_i m_i r_i / M$, where $r_i$ is the position of the $i$-th entity.

3. Momentum

   The momentum of an atomic entity is just $m_i v_i$, where $v_i$ is the velocity of the $i$-th entity. However, when thinking about the momentum of a composite entity, we have an additional complication: many of the constituent entities may be moving in different directions. If we were to only be able to see the composite entity as a whole, say with some boundary around it that defines its extension, we would see the boundary moving with some velocity. Thus, the momentum of a composite entity is this velocity times the total mass of the composite entity $M$.

   The momentum of a composite entity is the sum of the momenta of its constituents:

   Derived from the collective motion of constituent entities

3. Angular Momentum
   
   Now we come to the first emergent property that only exists at the composite level and does not exist at the atomic level. Rotation arises from the collective motion of constituent entities: $\omega = \sum_i m_i r_i \times v_i / I$, where $v_i$ is the velocity of the $i$-th entity, $I$ is the moment of inertia of the composite entity, and $\times$ denotes the cross product.
   
   The angular momentum of a composite entity is the sum of the angular momenta of its constituents:

   Derived from the collective motion of constituent entities


4. Temperature
   - Calculated based on the relative velocities of constituent atomic entities
   - Formula: $T = \sum_i (v_i - \bar{v})^2$, where $v_i$ is the velocity of the $i$-th atomic entity and $\bar{v}$ is the average velocity of the composite entity.



#### Emergent Force Fields

Composite entities can generate force fields that affect surrounding atomic entities. These fields arise from the emergent properties and collective behavior of the composite but act exclusively on atomic entities.

In many ways, they are similar to the fundamental forces but are generated by higher-level structures rather than being inherent properties of the atomic entities themselves. In theory, these force fields could arise from atomic entities, but for computational efficiency and to directly model many abstractions in physics, we treat them as emergent properties of composite entities.

1. Temperature Gradients

- Generated by high-temperature composite entities (e.g., "stars")
- Affects the behavior of nearby atomic entities, potentially influencing their velocity or spring formation
- Implemented as a long-range force field in the GPU simulation

2. Radiation Pressure

- Emitted by hot or energetic composite entities
- Exerts a repulsive force on nearby atomic entities, simulating effects like solar wind
- Integrated into the GPU force calculations alongside gravity and other forces

#### Implementation Approach

1. Identification and Tracking
   - Use graph algorithms to identify connected components in the spring network
   - Maintain a list of composite entities, updating it as springs form or break

2. Property Calculation
   - Compute emergent properties (temperature, rotation, etc.) on the CPU
   - Update these properties periodically rather than every timestep for efficiency

3. Force Field Generation
   - Generate field data for each composite entity based on its properties
   - Create a grid or other spatial structure to represent these fields efficiently

4. Integration with GPU Simulation

- Represent composite entity force fields in a format compatible with the GPU force calculation kernels
- Update the GPU kernels to include calculations for these additional force fields
- Ensure that all force calculations (fundamental and composite-generated) are performed exclusively on atomic entities in a unified manner

5. Performance Considerations

- Use adaptive resolution for force fields, with higher detail near the source and lower detail at greater distances
- Implement cutoff distances or approximation methods for very long-range interactions to maintain performance
- Balance the frequency of force field updates with the need for accuracy and performance


#### Interaction Dynamics

- All forces, including those generated by composite entities, act directly on atomic entities only
- Composite entities are affected indirectly through the changes in their constituent atomic entities
- The spring network propagates the effects of forces on individual atomic entities throughout the composite structure
- This creates a multi-scale dynamic where atomic-level interactions lead to emergent behaviors in composite entities



#### Motivation and Significance

1. Emergent Complexity: Allows for the simulation of complex astronomical objects and phenomena from simple building blocks
2. Scalability: Provides a way to manage and simulate large-scale structures efficiently
3. Rich Interactions: Enables a wide range of interactions between different scales of the simulation
4. User Engagement: Creates identifiable "objects" within the simulation that users can observe and interact with
5. Extensibility: The composite entity framework can be easily extended to include new properties or behaviors as needed
6. Unified Physics Model: By ensuring all forces act directly on atomic entities, we maintain a consistent and efficient physics simulation across all scales of the system
7. Emergent Behaviors: This approach allows for complex, large-scale phenomena to emerge naturally from the interactions of simple atomic entities, without the need for separate simulation systems for different scales
8. Scalability: The system can handle interactions ranging from individual particle collisions to the gravitational effects of massive celestial bodies, all within the same computational framework

## 3. Physics Engine

### 3.1 Force Calculations
- Gravity approximation: grid-based method
- Local force calculations: repulsion, springs

### 3.2 Integration Method
- Time step considerations
- Ensuring numerical stability

### 3.3 Collision Detection and Handling
- Efficient algorithms for millions of entities

## 4. GPU Acceleration
- Optimized data structures for GPU
- CUDA kernel designs
- Memory management strategies

## 5. Composite Entity Management
- Spring network formation and breaking rules
- Algorithms for identifying and tracking composite entities
- Calculation of emergent properties

## 6. Simulation Space
- Implementation of a boundless universe
- Spatial partitioning for efficient force calculations

## 7. Extensibility
- Framework for adding new forces or entity properties
- API for defining and interacting with composite entities

## 8. Performance Considerations
- Benchmarking methodology
- Scalability targets

## 9. Future Extensions
- User interaction framework
- Visualization system
- Network architecture for multi-user interaction

## 10. Implementation Phases
- Phase 1: Core physics engine with atomic entities
- Phase 2: Composite entities and emergent properties
- Phase 3: User interaction and visualization
- Phase 4: Multi-user capabilities and advanced features

## 11. Testing and Validation
- Unit testing strategies
- System-level tests for emergent behaviors
- Performance testing

## 12. Appendices
- Mathematical formulas for force calculations
- Pseudocode for key algorithms

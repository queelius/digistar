### Design Document: Sandbox Space Simulation Game

#### 1. **Motivation**

The primary motivation for this project is to create a highly interactive and scalable sandbox space simulation game. This game will simulate a vast number of "big atoms" interacting through various interesting forces, both global (e.g., gravity, electric fields) and local (e.g., repulsions, collisions). By leveraging GPU acceleration and optimized data structures like octrees, we aim to achieve high performance and handle a large number of simultaneous players and AI bots efficiently.

#### 2. **Purpose of the Game**

The game aims to provide an immersive space simulation environment where players can explore, interact, and experiment with various physical phenomena. Key objectives include:
- Simulating a dynamic universe with realistic physics.
- Allowing players to manipulate and observe the behavior of "big atoms" under different force fields.
- Supporting a large number of concurrent players and AI bots for a rich multiplayer experience.

#### 3. **Optimization Goals**

To achieve the desired scale and performance, we will focus on several key optimizations:
- **GPU Acceleration**: Offload computationally intensive tasks to the GPU to leverage parallel processing capabilities.
- **Efficient Data Structures**: Use octrees to manage spatial queries and force calculations efficiently.
- **Batch Processing**: Handle multiple bounding box queries and force calculations in parallel to reduce overhead and improve performance.

### 4. **Core Features**

#### 4.1 Physics Simulation
- **Big Atoms**: Fundamental units of the simulation, each with properties such as position, velocity, mass, charge, and radius.
- **Force Fields**: Includes global forces (e.g., gravity, electric fields) and local forces (e.g., repulsion, collision).
- **Octree Structure**: Utilized for efficient spatial partitioning and force calculations.

#### 4.2 Bounding Box Queries
- Efficiently handle multiple bounding box queries using batched processing on the GPU.
- Utilize octrees to quickly determine atoms within specified regions, supporting dynamic game scenarios and AI behaviors.

#### 4.3 Networking
- **RESTful Interface**: Provide a lightweight and fast HTTP-based interface for managing game state and interactions.
- **Binary UDP Interface**: Handle high-throughput, low-latency communication for real-time multiplayer interactions.

#### 4.4 Scripting and AI
- **Python Integration**: Expose a rich API to the Python interpreter, allowing for flexible scripting and AI control.
- **AI Bots**: Implement a base class `Agent` and derived class `SubsumptionAgent` to facilitate the creation of reactive, intelligent bots.

### 5. **Detailed Design**

#### 5.1 Physics Simulation

##### BigAtom Structure

```cpp
struct BigAtom {
    float3 position;
    float3 velocity;
    float mass;
    float charge;
    float radius;
    float3 color;
};
```

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

##### Force Calculation

- **Global Forces**: Compute using octree aggregation and walking.
- **Local Forces**: Compute using neighbor queries and distance checks.

##### Kernel Functions

- **Gravity Calculation**: Walk the octree for efficient force computation.
- **Bounding Box Queries**: Batch process multiple queries to minimize overhead.

#### 5.2 Networking

##### RESTful Interface
- Use a fast, lightweight web framework (e.g., FastAPI) to handle HTTP requests.
- Provide endpoints for game state management, player interactions, and administrative tasks.

##### Binary UDP Interface
- Design a custom binary protocol for efficient real-time communication.
- Ensure low latency and high throughput to support smooth multiplayer experiences.

#### 5.3 Scripting and AI

##### Python Integration
- Expose C++ simulation core functionalities to Python using a binding library (e.g., pybind11).
- Provide a comprehensive API for scripting game logic and AI behaviors.

##### AI Bot Framework
- Define a base class `Agent` and a derived class `SubsumptionAgent` for creating intelligent, reactive bots.
- Enable flexible AI scripting through Python, allowing for rapid development and testing of AI behaviors.

```python
class Agent:
    def __init__(self, id):
        self.id = id

    def update(self):
        pass

class SubsumptionAgent(Agent):
    def __init__(self, id):
        super().__init__(id)
        self.behaviors = []

    def add_behavior(self, behavior):
        self.behaviors.append(behavior)

    def update(self):
        for behavior in self.behaviors:
            if behavior.should_run():
                behavior.run()
                break
```

### 6. **Future Work**

- **Further Optimization**: Continuously profile and optimize GPU kernels and data structures.
- **Advanced AI**: Develop more sophisticated AI behaviors and decision-making processes.
- **Expanded Features**: Introduce new gameplay elements, force types, and interactive objects.

### 7. **Conclusion**

This design document outlines the foundational aspects of our sandbox space simulation game. By leveraging GPU acceleration, efficient data structures, and a robust networking and scripting framework, we aim to create a scalable and engaging simulation experience. This document serves as a reference for the initial implementation and future enhancements, guiding our development efforts toward achieving high performance and rich interactivity.



### Sub-README: Extended Features and Implementation Strategies

This document provides detailed information on additional features and implementation strategies for the sandbox space simulation game, including message passing, data handling, and optimization techniques.

---

## Table of Contents

- [Table of Contents](#table-of-contents)
- [Message Passing Interface](#message-passing-interface)
  - [UUID for Big Atoms](#uuid-for-big-atoms)
  - [Message Types and Structures](#message-types-and-structures)
    - [Apply Force](#apply-force)
    - [Get Big Atom](#get-big-atom)
    - [Make Big Atom](#make-big-atom)
    - [Get Bounding Box](#get-bounding-box)
    - [Bounding Box Query](#bounding-box-query)
    - [Mutation Request](#mutation-request)
  - [ZeroMQ Setup and Handling](#zeromq-setup-and-handling)
    - [Server-Side Implementation](#server-side-implementation)
    - [Client-Side Implementation](#client-side-implementation)
- [Data Handling Strategies](#data-handling-strategies)
  - [Struct of Arrays (SoA) vs Array of Structs (AoS)](#struct-of-arrays-soa-vs-array-of-structs-aos)
    - [Array of Structs (AoS)](#array-of-structs-aos)
    - [Struct of Arrays (SoA)](#struct-of-arrays-soa)
  - [Choosing Between AoS and SoA](#choosing-between-aos-and-soa)
  - [Implementation of SoA](#implementation-of-soa)
- [Future Work and Considerations](#future-work-and-considerations)

---

## Message Passing Interface

### UUID for Big Atoms

To uniquely identify each big atom and facilitate client-server interactions, we will use UUIDs (Universally Unique Identifiers). This approach allows clients to track identities and manage state efficiently.

### Message Types and Structures

#### Apply Force

```cpp
struct ApplyForceMessage {
    MessageType type;
    UUID uuid;
    float3 force;
};
```

#### Get Big Atom

```cpp
struct GetBigAtomMessage {
    MessageType type;
    UUID uuid;
};
```

#### Make Big Atom

```cpp
struct MakeBigAtomMessage {
    MessageType type;
    BigAtom newAtom; // or a structure containing initialization parameters
};
```

#### Get Bounding Box

```cpp
struct GetBoundingBoxMessage {
    MessageType type;
    BoundingBox box;
};
```

#### Bounding Box Query

```cpp
struct BoundingBoxQueryMessage {
    MessageType type;
    BoundingBox box;
};
```

#### Mutation Request

```cpp
struct MutationRequestMessage {
    MessageType type;
    UUID uuid;
    float3 force;
};
```

### ZeroMQ Setup and Handling

#### Server-Side Implementation

```cpp
void setupZeroMQ() {
    zmq::context_t context(1);
    zmq::socket_t socket(context, ZMQ_REP);
    socket.bind("tcp://*:5555");

    while (true) {
        zmq::message_t request;
        socket.recv(&request);

        MessageType msgType = *(MessageType*)request.data();

        switch (msgType) {
            case APPLY_FORCE: {
                ApplyForceMessage* msg = (ApplyForceMessage*)request.data();
                applyForce(msg->uuid, msg->force);
                break;
            }
            case GET_BIG_ATOM: {
                GetBigAtomMessage* msg = (GetBigAtomMessage*)request.data();
                BigAtom atom = getBigAtom(msg->uuid);
                zmq::message_t reply(sizeof(BigAtom));
                memcpy(reply.data(), &atom, sizeof(BigAtom));
                socket.send(reply);
                break;
            }
            case MAKE_BIG_ATOM: {
                MakeBigAtomMessage* msg = (MakeBigAtomMessage*)request.data();
                UUID newUuid = makeBigAtom(msg->newAtom);
                zmq::message_t reply(sizeof(UUID));
                memcpy(reply.data(), &newUuid, sizeof(UUID));
                socket.send(reply);
                break;
            }
            case GET_BOUNDING_BOX: {
                GetBoundingBoxMessage* msg = (GetBoundingBoxMessage*)request.data();
                std::vector<BigAtom> atoms = getBoundingBox(msg->box);
                zmq::message_t reply(atoms.size() * sizeof(BigAtom));
                memcpy(reply.data(), atoms.data(), atoms.size() * sizeof(BigAtom));
                socket.send(reply);
                break;
            }
            case BOUNDING_BOX_QUERY: {
                BoundingBoxQueryMessage* msg = (BoundingBoxQueryMessage*)request.data();
                std::vector<BigAtom> atoms = queryBoundingBox(msg->box);
                zmq::message_t reply(atoms.size() * sizeof(BigAtom));
                memcpy(reply.data(), atoms.data(), atoms.size() * sizeof(BigAtom));
                socket.send(reply);
                break;
            }
            case MUTATION_REQUEST: {
                MutationRequestMessage* msg = (MutationRequestMessage*)request.data();
                mutateBigAtom(msg->uuid, msg->force);
                break;
            }
            default:
                // Handle unknown message type
                break;
        }
    }
}
```

#### Client-Side Implementation

```cpp
void sendApplyForce(zmq::socket_t& socket, UUID uuid, float3 force) {
    ApplyForceMessage msg;
    msg.type = APPLY_FORCE;
    msg.uuid = uuid;
    msg.force = force;
    zmq::message_t request(sizeof(ApplyForceMessage));
    memcpy(request.data(), &msg, sizeof(ApplyForceMessage));
    socket.send(request);
}

BigAtom sendGetBigAtom(zmq::socket_t& socket, UUID uuid) {
    GetBigAtomMessage msg;
    msg.type = GET_BIG_ATOM;
    msg.uuid = uuid;
    zmq::message_t request(sizeof(GetBigAtomMessage));
    memcpy(request.data(), &msg, sizeof(GetBigAtomMessage));
    socket.send(request);

    zmq::message_t reply;
    socket.recv(&reply);
    BigAtom atom = *(BigAtom*)reply.data();
    return atom;
}

UUID sendMakeBigAtom(zmq::socket_t& socket, BigAtom newAtom) {
    MakeBigAtomMessage msg;
    msg.type = MAKE_BIG_ATOM;
    msg.newAtom = newAtom;
    zmq::message_t request(sizeof(MakeBigAtomMessage));
    memcpy(request.data(), &msg, sizeof(MakeBigAtomMessage));
    socket.send(request);

    zmq::message_t reply;
    socket.recv(&reply);
    UUID newUuid = *(UUID*)reply.data();
    return newUuid;
}

std::vector<BigAtom> sendGetBoundingBox(zmq::socket_t& socket, BoundingBox box) {
    GetBoundingBoxMessage msg;
    msg.type = GET_BOUNDING_BOX;
    msg.box = box;
    zmq::message_t request(sizeof(GetBoundingBoxMessage));
    memcpy(request.data(), &msg, sizeof(GetBoundingBoxMessage));
    socket.send(request);

    zmq::message_t reply;
    socket.recv(&reply);
    int numAtoms = reply.size() / sizeof(BigAtom);
    std::vector<BigAtom> atoms(numAtoms);
    memcpy(atoms.data(), reply.data(), reply.size());
    return atoms;
}
```

## Data Handling Strategies

### Struct of Arrays (SoA) vs Array of Structs (AoS)

When designing data structures for GPU-accelerated simulations, the choice between Struct of Arrays (SoA) and Array of Structs (AoS) is crucial for performance.

#### Array of Structs (AoS)

Each element in the array is a struct containing multiple fields.

```cpp
struct BigAtom {
    float3 position;
    float3 velocity;
    float mass;
    float charge;
    float radius;
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
    float* charges;
    float* radii;
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
}

void freeBigAtomSoA(BigAtomSoA& atomsSoA) {
    free(atomsSoA.positions);
    free(atomsSoA.velocities);
    free(atomsSoA.masses);
    free(atomsSoA.charges);
    free(atomsSoA.radii);
}
```

## Future Work and Considerations

1. **Further Optimization**:
   - Continuously profile and optimize GPU kernels and data structures.
   - Investigate advanced optimization techniques such as warp-level primitives and shared memory utilization.

2. **Advanced AI**:
   - Develop more sophisticated AI behaviors and decision-making processes.
   - Implement a flexible scripting API for AI and game logic.

3. **Expanded Features**:
   - Introduce new gameplay elements, force types, and interactive objects.
   - Enhance the networking layer to support more complex interactions and real-time updates.

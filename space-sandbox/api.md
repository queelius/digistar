## Simulation Initialization and Object Loading API

### Overview

This API provides a robust and flexible system for initializing and adding objects to a running simulation of celestial bodies. The primary goal is to handle hierarchical relationships during initialization while ensuring that objects operate independently during the simulation. The API includes mechanisms to respect dependencies and manage the state of objects dynamically.

### Key Components

1. **Body Structure**: Defines the properties of each celestial body.
2. **Initialization Functions**: Functions to add or update objects based on their relationships.
3. **Loading Functions**: Functions to load objects from JSON files, respecting parent-child relationships.
4. **Dependency Management**: Ensures that dependencies are respected during initialization.

### Body Structure

The `Body` structure represents a celestial object in the simulation.

```c++
struct Body {
    double3 position;
    double3 velocity;
    float3 color;
    double mass;
    double radius;
    char name[50];
    bool active;
    int parentIndex; // -1 if no parent
};
```

### Helper Functions

#### Get Body by Name

Returns the index of a body given its name. Returns -1 if the body does not exist.

```c++
int get_body_by_name(const char* name);
```

#### Add or Update Body

Adds a new body or updates an existing body without disrupting the simulation.

```c++
void addOrUpdateBody(const char* name,
                     double3 position,
                     double3 velocity,
                     double mass,
                     double radius,
                     float3 color,
                     int parentIndex);
```

### Loading Functions

#### Load Bodies from JSON with Parent Index

Loads bodies from a JSON file, using the specified parent index to set initial positions and velocities.

```c++
void loadBodiesFromJson(const std::string& filename, int parentIndex);
```

#### Load Bodies from JSON with Position and Velocity Offset

Loads bodies from a JSON file, using the specified position and velocity offsets.

```c++
void loadBodiesFromJson(const std::string& filename,
                        double3 position_offset = {0.0, 0.0, 0.0},
                        double3 velocity_offset = {0.0, 0.0, 0.0});
```

### Directory Loading Function

#### Load Bodies from JSON Directory

Loads all JSON files in a specified directory, ensuring dependencies are respected.

```c++
void loadBodiesFromJsonDir(const std::string& dirPath);
```

### Default Behaviors and Corner Cases

1. **Parents**: Leave existing parents unchanged.
2. **Children**: Update children if they already exist; otherwise, add them.
3. **Orphan Children**: Handle cases where children are loaded without their parents.
4. **Parent Updates**: Ensure that updating a parent object doesn’t inadvertently leave children with incorrect initial states.

### Example Usage

#### Loading with Parent Index

```c++
loadBodiesFromJson("jupiter_system.json", get_body_by_name("Jupiter"));
```

#### Loading with Position and Velocity Offset

```c++
double3 offsetPosition = make_double3(1.496e11, 0.0, 0.0);
double3 offsetVelocity = make_double3(0.0, 2.978e4, 0.0);
loadBodiesFromJson("mars_system.json", offsetPosition, offsetVelocity);
```

### Detailed Function Descriptions

#### `get_body_by_name`

**Description**: Searches for a body by name and returns its index.

**Parameters**:
- `name`: The name of the body to search for.

**Returns**: The index of the body if found, otherwise -1.

#### `addOrUpdateBody`

**Description**: Adds a new body to the simulation or updates an existing body.

**Parameters**:
- `name`: The name of the body.
- `position`: The position of the body.
- `velocity`: The velocity of the body.
- `mass`: The mass of the body.
- `radius`: The radius of the body.
- `color`: The color of the body.
- `parentIndex`: The index of the parent body. Use -1 if no parent.

#### `loadBodiesFromJson` (Parent Index)

**Description**: Loads bodies from a JSON file using the specified parent index.

**Parameters**:
- `filename`: The path to the JSON file.
- `parentIndex`: The index of the parent body.

#### `loadBodiesFromJson` (Position and Velocity Offset)

**Description**: Loads bodies from a JSON file using the specified position and velocity offsets.

**Parameters**:
- `filename`: The path to the JSON file.
- `position_offset`: The position offset to apply to the parent body.
- `velocity_offset`: The velocity offset to apply to the parent body.

#### `loadBodiesFromJsonDir`

**Description**: Loads all JSON files in a specified directory, ensuring dependencies are respected.

**Parameters**:
- `dirPath`: The path to the directory containing JSON files.

### Summary

This API provides a structured and flexible approach to initializing and adding celestial bodies to a simulation. By managing dependencies and relationships during initialization, the API ensures that objects are added in a meaningful way without disrupting the ongoing simulation. This approach balances flexibility and control, allowing for the creation of rich and dynamic simulations.

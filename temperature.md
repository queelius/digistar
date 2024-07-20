### Temperature Calculation for Composites

1. **Composite Temperature:**
   - Calculate the temperature of a composite based on the velocities of its constituent big atoms relative to the composite's center of mass velocity.
   - Use the kinetic energy formula to derive the temperature.

2. **Convex Hull Volume:**
   - Calculate the volume of the composite's convex hull to normalize the temperature.

### Implementation Steps

#### Step 1: Calculate Center of Mass and Composite Velocity

1. **Center of Mass:**
   ```cpp
   float3 calculateCenterOfMass(const Composite& composite, const BigAtomSoA& atoms) {
       float3 centerOfMass = make_float3(0.0f, 0.0f, 0.0f);
       float totalMass = 0.0f;

       for (const auto& atomID : composite.atoms) {
           float mass = atoms.masses[atomID];
           centerOfMass += atoms.positions[atomID] * mass;
           totalMass += mass;
       }

       return centerOfMass / totalMass;
   }
   ```

2. **Composite Velocity:**
   ```cpp
   float3 calculateCompositeVelocity(const Composite& composite, const BigAtomSoA& atoms) {
       float3 velocity = make_float3(0.0f, 0.0f, 0.0f);
       float totalMass = 0.0f;

       for (const auto& atomID : composite.atoms) {
           float mass = atoms.masses[atomID];
           velocity += atoms.velocities[atomID] * mass;
           totalMass += mass;
       }

       return velocity / totalMass;
   }
   ```

#### Step 2: Calculate Kinetic Energy and Temperature

1. **Kinetic Energy:**
   ```cpp
   float calculateKineticEnergy(const Composite& composite, const BigAtomSoA& atoms) {
       float3 compositeVelocity = calculateCompositeVelocity(composite, atoms);
       float kineticEnergy = 0.0f;

       for (const auto& atomID : composite.atoms) {
           float3 relativeVelocity = atoms.velocities[atomID] - compositeVelocity;
           kineticEnergy += 0.5f * atoms.masses[atomID] * dot(relativeVelocity, relativeVelocity);
       }

       return kineticEnergy;
   }
   ```

2. **Composite Temperature:**
   ```cpp
   float calculateTemperature(const Composite& composite, const BigAtomSoA& atoms) {
       float kineticEnergy = calculateKineticEnergy(composite, atoms);
       float volume = calculateConvexHullVolume(composite, atoms);

       return kineticEnergy / volume;
   }
   ```

#### Step 3: Cooling Mechanism

1. **Cooling Over Time:**
   ```cpp
   void coolComposite(Composite& composite, float coolingRate) {
       composite.temperature -= coolingRate * composite.atoms.size();
       if (composite.temperature < ambientTemperature) {
           composite.temperature = ambientTemperature;
       }
   }
   ```

### Convex Hull Volume Calculation

1. **Using CGAL for Convex Hull Volume:**
   ```cpp
   float calculateConvexHullVolume(const Composite& composite, const BigAtomSoA& atoms) {
       std::vector<Point> points;

       for (const auto& atomID : composite.atoms) {
           points.push_back(Point(atoms.positions[atomID].x, atoms.positions[atomID].y, atoms.positions[atomID].z));
       }

       Polyhedron_3 hull;
       CGAL::convex_hull_3(points.begin(), points.end(), hull);

       return CGAL::to_double(CGAL::Polygon_mesh_processing::volume(hull));
   }
   ```

### Example Workflow

1. **Server Initialization:**
   - Identify initial composites and set up their structures.
   ```cpp
   std::vector<Composite> composites = initializeComposites();
   ```

2. **Main Simulation Loop:**
   - Update composites, calculate temperatures, and apply forces dynamically.
   ```cpp
   while (simulationRunning) {
       updateAtomPositions();
       updateComposites(composites);
       calculateTemperatures(composites, atoms);
       handleFusionAndFission(atoms, octree);
       applyTemperatureGradients(composites);
       handleClientInteractions();
       renderScene();
   }
   ```

3. **Client-Side Interaction:**
   - Apply forces to composites and send updates to the server.
   ```cpp
   void main() {
       // Initialize networking and GUI
       initializeNetwork();
       initializeGUI();

       // Create actuators for composites
       CompositeActuator earthActuator("UUID_Earth");
       CompositeActuator moonActuator("UUID_Moon");

       // Main loop
       while (true) {
           // Receive atom data from the server
           receiveAtomDataFromServer();

           // Visualize the reconstructed surfaces and composites
           visualizeSurfaces();

           // Handle user interactions
           Interaction interaction = getUserInteraction();
           if (interaction.isValid()) {
               if (interaction.type == InteractionType::Torque) {
                   earthActuator.applyTorque(interaction.value);
               } else if (interaction.type == InteractionType::LinearForce) {
                   moonActuator.applyLinearForce(interaction.value);
               } else if (interaction.type == InteractionType::Heat) {
                   earthActuator.applyHeat(interaction.value);
               }
           }

           // Render the scene
           renderScene();
       }
   }
   ```

### Summary

By calculating the temperature of composites based on the kinetic energy of their constituent big atoms, you can accurately model temperature-related effects. This approach allows for dynamic interactions and realistic simulation of thermal processes within your system.

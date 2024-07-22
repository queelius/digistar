#include <unistd.h>
#include "simulation.h"
#include "simulation_api.h"

int main() {
    // Create a simulation object
    Simulation sim(1000, 0.1);

    // Create an API object
    SimulationAPI api(&sim);

    // Load bodies from JSON files
    // api.loadBodiesFromJsonDir("./solar_system");
    api.loadBodiesFromJson("./solar_system/main_planets.json");

    // Start the simulation
    sim.startSimulationThread();

    // Keep the main thread alive to allow queries
    while (true) {
        sim.updateBodies();
        auto bodies = sim.getBodies();
        for (int i = 0; i < sim.getActiveBodyCount(); i++) {
            printf("Body %s: (%f, %f, %f)\n", bodies[i].name, bodies[i].position.x, bodies[i].position.y, bodies[i].position.z);
        }
        sleep(10);
    }

    return 0;
}

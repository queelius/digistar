#ifndef SIMULATION_CONTROLLER_H
#define SIMULATION_CONTROLLER_H

#include "simulation.h"
#include <string>
#include "json.hpp"

using json = nlohmann::json;

class SimulationController {
public:
    SimulationController(Simulation * sim) : sim(sim) {}

    void loadBodiesFromJson(const std::string& filename, double3 position_offset = {0.0, 0.0, 0.0}, double3 velocity_offset = {0.0, 0.0, 0.0});
    void loadBodiesFromJsonDir(const std::string& dirPath);
    void saveBodiesToJson(const std::string& filename);

    // simulation has a CommandQueue that we can push commands to
    // this is how we can control the simulation from the controller
    // we do not want to provide a low-level CommandQueue interface, as
    // we want the shared resouce to be managed by the simulation -- in this
    // way, it can orchtstrate the commands that are pushed to it without
    // having to worry about synchronization, as it will take care of it.

    void pause();
    void resume();
    void isPaused();
    void setGrav(double grav);
    void setColoumb(double coloumb);
    void setTimeStep(double timeStep);
    void sendCommand(const Command& cmd) {
        sim->sendCommand(cmd);
    }

private:
    Simulation * sim;
};

#endif // SIMULATION_CONTROLLER_H

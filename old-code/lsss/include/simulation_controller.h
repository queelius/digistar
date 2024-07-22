#ifndef SIMULATION_CONTROLLER_H
#define SIMULATION_CONTROLLER_H

#include "control_data.h"
#include <string>
#include "json.hpp"

using json = nlohmann::json;

class SimulationController {
public:
    SimulationController();

    void loadBodiesFromJson(const std::string& filename, double3 position_offset = {0.0, 0.0, 0.0}, double3 velocity_offset = {0.0, 0.0, 0.0});
    void loadBodiesFromJsonDir(const std::string& dirPath);
    void addOrUpdateBody(const char* name, const BodyCommand& params);
    void sendCommand(Command cmd, double timeStep = 0.0);
    void sendBodyCommand(const BodyCommand& bodyCmd);

private:
    void parseAndAddBody(const json& body, double3 position_offset, double3 velocity_offset, int parentIndex);

    ControlData* controlData;
    Body* bodies;
    int maxBodies;

    int getBodyIndexByName(const char* name) const;
    int getActiveBodyCount() const;
};

#endif // SIMULATION_CONTROLLER_H

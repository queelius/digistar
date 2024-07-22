// simulation_controller.cpp

#include "simulation_controller.h"
#include <fstream>
#include <dirent.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <cstring>

#define CTRL_SHM_NAME "/control_data"
#define SHM_NAME "/bodies"

SimulationController::SimulationController() {
    int ctrl_fd = shm_open(CTRL_SHM_NAME, O_RDWR, 0666);
    if (ctrl_fd == -1) {
        perror("shm_open");
        exit(1);
    }
    controlData = (ControlData*)mmap(NULL, sizeof(ControlData), PROT_READ | PROT_WRITE, MAP_SHARED, ctrl_fd, 0);
    if (controlData == MAP_FAILED) {
        perror("mmap");
        exit(1);
    }
    controlData->commandQueueSize = 0;
    close(ctrl_fd);

    int shm_fd = shm_open(SHM_NAME, O_RDWR, 0666);
    if (shm_fd == -1) {
        perror("shm_open (bodies)");
        exit(1);
    }
    bodies = (Body*)mmap(NULL, N * sizeof(Body), PROT_READ | PROT_WRITE, MAP_SHARED, shm_fd, 0);
    if (bodies == MAP_FAILED) {
        perror("mmap (bodies)");
        exit(1);
    }
    close(shm_fd);

    maxBodies = N;
}

void SimulationController::loadBodiesFromJson(const std::string& filename, double3 position_offset, double3 velocity_offset) {
    std::ifstream file(filename);
    json j;
    file >> j;

    double3 parPos = position_offset;
    double3 parVel = velocity_offset;
    int parIdx = -1;

    if (j.contains("parent")) {
        const auto& par = j["parent"];
        const char* parentName = par["name"].get<std::string>().c_str();
        parIdx = getBodyIndexByName(parentName);
        if (parIdx != -1) {
            parPos = bodies[parIdx].position;
            parVel = bodies[parIdx].velocity;
        } else {
            double3 relPos = make_double3(par["position"][0].get<double>(), par["position"][1].get<double>(), par["position"][2].get<double>());
            double3 relVel = make_double3(par["velocity"][0].get<double>(), par["velocity"][1].get<double>(), par["velocity"][2].get<double>());
            BodyCommand parentCommand;
            parentCommand.command = ADD_BODY;
            strcpy(parentCommand.name, parentName);
            parentCommand.position = make_double3(parPos.x + relPos.x, parPos.y + relPos.y, parPos.z + relPos.z);
            parentCommand.velocity = make_double3(parVel.x + relVel.x, parVel.y + relVel.y, parVel.z + relVel.z);
            parentCommand.mass = par["mass"].get<double>();
            parentCommand.radius = par["radius"].get<double>();
            parentCommand.color = make_float3(par["color"][0].get<float>(), par["color"][1].get<float>(), par["color"][2].get<float>());
            sendBodyCommand(parentCommand);
            parIdx = getActiveBodyCount() - 1;
        }
    }

    for (const auto& body : j["children"]) {
        parseAndAddBody(body, parPos, parVel, parIdx);
    }
}

void SimulationController::loadBodiesFromJsonDir(const std::string& dirPath) {
    DIR* dir;
    struct dirent* ent;
    if ((dir = opendir(dirPath.c_str())) != NULL) {
        while ((ent = readdir(dir)) != NULL) {
            std::string filename = ent->d_name;
            if (filename.find(".json") != std::string::npos) {
                std::string filepath = dirPath + "/" + filename;
                loadBodiesFromJson(filepath);
            }
        }
        closedir(dir);
    } else {
        perror("opendir");
    }
}

void SimulationController::parseAndAddBody(const json& body, double3 position_offset, double3 velocity_offset, int parentIndex) {
    const char* childName = body["name"].get<std::string>().c_str();
    int childIdx = getBodyIndexByName(childName);
    if (childIdx != -1) {
        printf("Warning: Body with name %s already exists. Ignoring new body.\n", childName);
        return;
    }

    BodyCommand childCommand;
    childCommand.command = ADD_BODY;
    strcpy(childCommand.name, childName);
    childCommand.position = make_double3(position_offset.x + body["position"][0].get<double>(), position_offset.y + body["position"][1].get<double>(), position_offset.z + body["position"][2].get<double>());
    childCommand.velocity = make_double3(velocity_offset.x + body["velocity"][0].get<double>(), velocity_offset.y + body["velocity"][1].get<double>(), velocity_offset.z + body["velocity"][2].get<double>());
    childCommand.mass = body["mass"].get<double>();
    childCommand.radius = body["radius"].get<double>();
    childCommand.color = make_float3(body["color"][0].get<float>(), body["color"][1].get<float>(), body["color"][2].get<float>());
    childCommand.parentIndex = parentIndex;

    sendBodyCommand(childCommand);
}

void SimulationController::addOrUpdateBody(const char* name, const BodyCommand& params) {
    int index = getBodyIndexByName(name);
    if (index == -1) {
        // Body does not exist, add it
        sendBodyCommand(params);
    } else {
        // Body exists, update it
        BodyCommand updateCmd = params;
        updateCmd.command = UPDATE_BODY;
        sendBodyCommand(updateCmd);
    }
}

void SimulationController::sendCommand(Command cmd, double timeStep) {
    int ctrl_fd = shm_open(CTRL_SHM_NAME, O_RDWR, 0666);
    if (ctrl_fd == -1) {
        perror("shm_open");
        exit(1);
    }
    ControlData* controlData = (ControlData*)mmap(NULL, sizeof(ControlData), PROT_READ | PROT_WRITE, MAP_SHARED, ctrl_fd, 0);
    if (controlData == MAP_FAILED) {
        perror("mmap");
        exit(1);
    }
    controlData->commandQueue[controlData->commandQueueSize].command = cmd;
    controlData->commandQueue[controlData->commandQueueSize].position.reset();
    controlData->commandQueue[controlData->commandQueueSize].velocity.reset();
    controlData->commandQueue[controlData->commandQueueSize].mass.reset();
    controlData->commandQueue[controlData->commandQueueSize].radius.reset();
    controlData->commandQueue[controlData->commandQueueSize].color.reset();
    controlData->commandQueueSize++;
    munmap(controlData, sizeof(ControlData));
    close(ctrl_fd);
}

void SimulationController::sendBodyCommand(const BodyCommand& bodyCmd) {
    int ctrl_fd = shm_open(CTRL_SHM_NAME, O_RDWR, 0666);
    if (ctrl_fd == -1) {
        perror("shm_open");
        exit(1);
    }
    ControlData* controlData = (ControlData*)mmap(NULL, sizeof(ControlData), PROT_READ | PROT_WRITE, MAP_SHARED, ctrl_fd, 0);
    if (controlData == MAP_FAILED) {
        perror("mmap");
        exit(1);
    }
    if (controlData->commandQueueSize < 10) { // Check if there is space in the command queue
        controlData->commandQueue[controlData->commandQueueSize++] = bodyCmd;
    } else {
        printf("Warning: Command queue is full. Ignoring new command.\n");
    }
    munmap(controlData, sizeof(ControlData));
    close(ctrl_fd);
}

int SimulationController::getBodyIndexByName(const char* name) const {
    for (int i = 0; i < maxBodies; i++) {
        if (bodies[i].active && strcmp(bodies[i].name, name) == 0) {
            return i;
        }
    }
    return -1;
}

int SimulationController::getActiveBodyCount() const {
    int count = 0;
    for (int i = 0; i < maxBodies; i++) {
        if (bodies[i].active) {
            count++;
        }
    }
    return count;
}

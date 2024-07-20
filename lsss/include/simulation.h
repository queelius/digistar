#ifndef SIMULATION_H
#define SIMULATION_H

#include <cuda_runtime.h>
#include <pthread.h>
#include <semaphore.h>
#include <optional>
#include "control_data.h"

struct Body {
    double3 position;
    double3 velocity;
    float3 color;
    double mass;
    double radius;
    char name[32];
    int parentIndex;
    bool active;
};

class Simulation {
public:
    Simulation(int maxBodies, double timeStep = 0.1, double grav = 6.67430e-11);
    ~Simulation();

    void initializeSharedMemory();
    void allocateDeviceMemory();
    void startSimulationThread();
    void updateBodies();

    int getBodyIndexByName(const char* name) const;
    Body* getBodies() const;
    int getActiveBodyCount() const;
    void addBody(const Body& body);
    void removeBody(const char* name);
    void updateBody(int index, const BodyCommand& command);

private:
    Body* bodies;
    Body* d_bodies;
    int activeBodyCount;
    int maxBodies;
    double timeStep;
    double grav;
    pthread_t simulationThread;
    ControlData* controlData;

    static void* simulationThreadFunc(void* arg);

    void initializeBodies();
    void updateDeviceMemory();
    void handleControlCommands();
};

__global__ void updateBodiesKernel(Body* bodies, int n, double dt, double grav);

#endif // SIMULATION_H

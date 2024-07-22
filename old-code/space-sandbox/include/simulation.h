#ifndef SIMULATION_H
#define SIMULATION_H

#include <pthread.h>
#include "common.h"
#include "command_queue.h"

class Simulation {
public:
    Simulation(
        int maxBodies = 1000,
        double timeStep = 1,        
        double grav = 6.67430e-11,
        double coloumbs = 8.9875517873681764e9);

    ~Simulation();

    void initializeSharedMemory();
    void allocateDeviceMemory();
    void startSimulationThread();
    void updateBodies();
    Body* getBodies() const;

    int getBodyIndexByName(const char* name) const;

    // Command interface

    // Queue a command to be handled by the simulation thread
    void queueCommand(const Command& cmd);

    // Handle command in the simulation thread
    // We can specify the maximum number of commands to handle at once
    // If it's -1, we handle all commands in the queue. This is useful
    // for handling all commands in the queue before starting the simulation
    // or other batch operations that may be done at once
    void handleCommands(const Command& cmd, int maxCommands = -1);

    // Body command handlers
    void addBody(const BodyCommand& cmd);
    void removeBody(int index);
    void updateBody(int index, const BodyCommand& cmd);

    // Simulation state mutators
    void pause();
    void resume();
    void setGrav(double grav);
    void setColoumb(double coloumb);
    void setTimeStep(double timeStep);

    // Simulation state getters
    int getActiveBodyCount() const;
    double getTimeStep() const;
    double getGrav() const;
    double getColoumb() const;
    bool getIsPaused() const;
    double getElapsedTime() const;
    double getSimulationTime() const;
    
private:
    Body* bodies;
    Body* d_bodies;
    int maxBodies;
    double timeStep;
    double grav;
    double coloumb;
    bool isPaused;
    pthread_t simulationThread;
    CommandQueue cmdQueue;

    static void* simulationThreadFunc(void* arg);

    void initializeBodies();
    void updateDeviceMemory();


};

#endif // SIMULATION_H

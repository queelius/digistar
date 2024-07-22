#include "simulation.h"
#include <unistd.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <cstring>
#include "sim_cuda_core.h"
#include "command_queue.h"

#define BODY_SHM_NAME "/bodies"
#define CTRL_SHM_NAME "/control_data"

Simulation::Simulation(int maxBodies, int cmdQueueSize, double timeStep, double grav, double coloumb)
    : maxBodies(maxBodies), timeStep(timeStep), grav(grav), coloumb(coloumb) 
{
    bodies = new Body[maxBodies];
    cmdQueue = new CommandQueue(cmdQueueSize);
    
    initializeSharedMemory();
    allocateDeviceMemory();
}

Simulation::~Simulation()
{
    delete[] bodies;
    cudaFree(d_bodies);
}

void Simulation::initializeSharedMemory()
{
    int body_fd = shm_open(BODY_SHM_NAME, O_CREAT | O_RDWR, 0666);
    if (body_fd == -1)
    {
        perror("shm_open(body)");
        exit(1);
    }
    if (ftruncate(body_fd, maxBodies * sizeof(Body)) == -1)
    {
        perror("ftruncate(body)");
        exit(1);
    }
    bodies = (Body *)mmap(NULL, maxBodies * sizeof(Body), PROT_READ | PROT_WRITE, MAP_SHARED, body_fd, 0);
    if (bodies == MAP_FAILED)
    {
        perror("mmap(body)");
        exit(1);
    }
    for (int i = 0; i < maxBodies; i++)
    {
        bodies[i].active = false;
    }

}

void Simulation::allocateDeviceMemory()
{
    cudaMalloc(&d_bodies, maxBodies * sizeof(Body));
}

void Simulation::startSimulationThread()
{
    pthread_create(&simulationThread, NULL, simulationThreadFunc, this);
}

void *Simulation::simulationThreadFunc(void *arg)
{
    Simulation *sim = static_cast<Simulation *>(arg);
    int blockSize = 256;
    int numBlocks = (sim->maxBodies + blockSize - 1) / blockSize;
    while (true)
    {
        sim->handleControlCommands();
        if (!sim->isPaused)
        {
            updateGravityFieldKernel<<<numBlocks, blockSize>>>(
                sim->d_bodies, sim->maxBodies, sim->grav);

            updateElectricFieldKernel<<<numBlocks, blockSize>>>(
                sim->d_bodies, sim->maxBodies, sim->coloumb);

            integrateKernel<<<numBlocks, blockSize>>>(
                sim->d_bodies, sim->maxBodies, sim->timeStep);

            cudaDeviceSynchronize();
            cudaMemcpy(
                sim->bodies,
                sim->d_bodies,
                sim->maxBodies * sizeof(Body),
                cudaMemcpyDeviceToHost);
        }
        usleep(1000);
    }
    return nullptr;
}

void Simulation::updateBodies()
{
    // The update logic, if required outside of the kernel, can go here
}

int Simulation::getBodyIndexByName(const char *name) const
{
    for (int i = 0; i < maxBodies; i++)
    {
        if (bodies[i].active && strcmp(bodies[i].name, name) == 0)
        {
            return i;
        }
    }
    return -1;
}

Body *Simulation::getBodies() const
{
    return bodies;
}

int Simulation::getBodyCount() const
{
    int count = 0;
    for (int i = 0; i < maxBodies; i++) {
        count += bodies[i].active;
    }
    return count;
}

void Simulation::removeBody(int index)
{
    if (index < 0 || index >= maxBodies || !bodies[index].active)
    {
        printf("Warning: Invalid body index %d. Cannot remove.\n", index);
        return;
    }
    bodies[index].active = false;
    updateDeviceMemory();
}



int Simulation::addBody(const BodyCommand &command)
{
    for (int i = 0; i < maxBodies; i++)
    {
        if (!bodies[i].active)
        {

            if (command.position.has_value())
                bodies[index].position = command.position.value();
            if (command.velocity.has_value())
                bodies[index].velocity = command.velocity.value();
            if (command.mass.has_value())
                bodies[index].mass = command.mass.value();
            if (command.radius.has_value())
                bodies[index].radius = command.radius.value();
            if (command.color.has_value())
                bodies[index].color = command.color.value();

            bodies[i] = command;
            bodies[i].active = true;
            activeBodyCount++;
            updateDeviceMemory();
            return i;
        }
    }
    return -1;
}
            Body newBody;
            strcpy(newBody.name, cmd.name);
            newBody.position = cmd.position.value_or(make_double3(0.0, 0.0, 0.0));
            newBody.velocity = cmd.velocity.value_or(make_double3(0.0, 0.0, 0.0));
            newBody.force = make_double3(0.0, 0.0, 0.0);
            newBody.mass = cmd.mass.value_or(1.0);
            newBody.charge = cmd.charge.value_or(0.0);
            newBody.radius = cmd.radius.value_or(1.0);
            newBody.color = cmd.color.value_or(make_float3(0.5, 0.5, 0.5));
            newBody.active = true;


void Simulation::updateBody(int index, const BodyCommand &command)
{
    if (command.position.has_value())
        bodies[index].position = command.position.value();
    if (command.velocity.has_value())
        bodies[index].velocity = command.velocity.value();
    if (command.mass.has_value())
        bodies[index].mass = command.mass.value();
    if (command.radius.has_value())
        bodies[index].radius = command.radius.value();
    if (command.color.has_value())
        bodies[index].color = command.color.value();
    updateDeviceMemory();
}

void Simulation::updateDeviceMemory()
{
    cudaMemcpy(d_bodies, bodies, maxBodies * sizeof(Body), cudaMemcpyHostToDevice);
}

void Simulation::handleControlCommands()
{
    for (int i = 0; i < controlData->commandQueueSize; i++)
    {
        const BodyCommand &cmd = controlData->commandQueue[i];
        switch (cmd.command)
        {
        case ADD_BODY:
        {
            // make sure body by the same name doesn't already exist
            if (getBodyIndexByName(cmd.name) != -1)
            {
                printf("Warning: Body with name %s already exists. Ignoring new body.\n", cmd.name);
                break;
            }

            addBody(newBody);
            break;
        }
        case REMOVE_BODY:
            int index = getBodyIndexByName(cmd.name);
            if (index == -1) {
                printf("Warning: Body with name %s not found. Cannot remove.\n", cmd.name);
                break;
            }
            removeBody(index);
            break;
        case UPDATE_BODY:
        {
            int index = getBodyIndexByName(cmd.name);
            if (index == -1) {
                printf("Warning: Body with name %s not found. Cannot update.\n", cmd.name);
                break;
            }
            updateBody(index, cmd);
            break;
        }
        default:
            break;
        }
    }
    controlData->commandQueueSize = 0; // Clear the command queue after processing
}

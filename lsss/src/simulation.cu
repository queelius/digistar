#include "simulation.h"
#include <unistd.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <cstring>

#define SHM_NAME "/bodies"
#define CTRL_SHM_NAME "/control_data"

Simulation::Simulation(int maxBodies, double timeStep, double grav)
    : maxBodies(maxBodies), timeStep(timeStep), grav(grav), activeBodyCount(0)
{
    bodies = new Body[maxBodies];
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
    int shm_fd = shm_open(SHM_NAME, O_CREAT | O_RDWR, 0666);
    if (shm_fd == -1)
    {
        perror("shm_open");
        exit(1);
    }
    if (ftruncate(shm_fd, maxBodies * sizeof(Body)) == -1)
    {
        perror("ftruncate");
        exit(1);
    }
    bodies = (Body *)mmap(NULL, maxBodies * sizeof(Body), PROT_READ | PROT_WRITE, MAP_SHARED, shm_fd, 0);
    if (bodies == MAP_FAILED)
    {
        perror("mmap");
        exit(1);
    }
    for (int i = 0; i < maxBodies; i++)
    {
        bodies[i].active = false;
    }

    int ctrl_fd = shm_open(CTRL_SHM_NAME, O_CREAT | O_RDWR, 0666);
    if (ctrl_fd == -1)
    {
        perror("shm_open (control)");
        exit(1);
    }
    if (ftruncate(ctrl_fd, sizeof(ControlData)) == -1)
    {
        perror("ftruncate (control)");
        exit(1);
    }
    controlData = (ControlData *)mmap(NULL, sizeof(ControlData), PROT_READ | PROT_WRITE, MAP_SHARED, ctrl_fd, 0);
    if (controlData == MAP_FAILED)
    {
        perror("mmap (control)");
        exit(1);
    }
    controlData->commandQueueSize = 0;
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
        sem_wait(sim->semaphore); // Lock the semaphore
        sim->handleControlCommands();
        updateBodiesKernel<<<numBlocks, blockSize>>>(sim->d_bodies, sim->maxBodies, sim->timeStep, sim->grav);
        cudaDeviceSynchronize();
        cudaMemcpy(sim->bodies, sim->d_bodies, sim->maxBodies * sizeof(Body), cudaMemcpyDeviceToHost);
        sem_post(sim->semaphore); // Unlock the semaphore
        usleep(1000);
    }
    return nullptr;
}

__global__ void updateBodiesKernel(Body *bodies, int n,
                                   double dt, double grav)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n && bodies[i].active)
    {
        double3 force = {0.0, 0.0, 0.0};
        for (int j = 0; j < n; j++)
        {
            if (i != j && bodies[j].active)
            {
                double3 r;
                r.x = bodies[j].position.x - bodies[i].position.x;
                r.y = bodies[j].position.y - bodies[i].position.y;
                r.z = bodies[j].position.z - bodies[i].position.z;

                double distSqr = r.x * r.x + r.y * r.y + r.z * r.z;
                double F = grav * bodies[i].mass * bodies[j].mass *
                           rsqrt(distSqr * distSqr * distSqr);
                force.x += F * r.x;
                force.y += F * r.y;
                force.z += F * r.z;
            }
        }

        bodies[i].velocity.x += dt * force.x / bodies[i].mass;
        bodies[i].velocity.y += dt * force.y / bodies[i].mass;
        bodies[i].velocity.z += dt * force.z / bodies[i].mass;

        bodies[i].position.x += dt * bodies[i].velocity.x;
        bodies[i].position.y += dt * bodies[i].velocity.y;
        bodies[i].position.z += dt * bodies[i].velocity.z;
    }
}

void Simulation::updateBodies()
{
    sem_wait(semaphore); // Lock the semaphore
    // The update logic, if required outside of the kernel, can go here
    sem_post(semaphore); // Unlock the semaphore
}

int Simulation::getBodyIndexByName(const char *name) const
{
    for (int i = 0; i < activeBodyCount; i++)
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

int Simulation::getActiveBodyCount() const
{
    return activeBodyCount;
}

void Simulation::addBody(const Body &body)
{
    if (activeBodyCount >= maxBodies)
    {
        printf("Warning: Maximum number of bodies reached. Ignoring new body.\n");
        return;
    }
    bodies[activeBodyCount] = body;
    bodies[activeBodyCount].active = true;
    activeBodyCount++;
    updateDeviceMemory();
}

void Simulation::removeBody(const char *name)
{
    int index = getBodyIndexByName(name);
    if (index != -1)
    {
        bodies[index].active = false;
        updateDeviceMemory();
    }
    else
    {
        printf("Warning: Body with name %s not found. Cannot remove.\n", name);
    }
}

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
    bodies[index].parentIndex = command.parentIndex;
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
            Body newBody;
            strcpy(newBody.name, cmd.name);
            newBody.position = cmd.position.value_or(make_double3(0.0, 0.0, 0.0));
            newBody.velocity = cmd.velocity.value_or(make_double3(0.0, 0.0, 0.0));
            newBody.mass = cmd.mass.value_or(1.0);
            newBody.radius = cmd.radius.value_or(1.0);
            newBody.color = cmd.color.value_or(make_float3(0.5, 0.5, 0.5));
            newBody.parentIndex = cmd.parentIndex;
            newBody.active = true;
            addBody(newBody);
            break;
        }
        case REMOVE_BODY:
            removeBody(cmd.name);
            break;
        case UPDATE_BODY:
        {
            int index = getBodyIndexByName(cmd.name);
            if (index != -1)
            {
                updateBody(index, cmd);
            }
            else
            {
                printf("Warning: Body with name %s not found. Cannot update.\n", cmd.name);
            }
            break;
        }
        default:
            break;
        }
    }
    controlData->commandQueueSize = 0; // Clear the command queue after processing
}

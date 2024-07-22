Simulation::Simulation(int maxBodies, double timeStep) 
    : maxBodies(maxBodies), timeStep(timeStep), activeBodyCount(0) {
    bodies = new Body[maxBodies];
    initializeSharedMemory();
    allocateDeviceMemory();
}

Simulation::~Simulation() {
    delete[] bodies;
    cudaFree(d_bodies);
}

void Simulation::initializeSharedMemory() {
    // Implementation
}

void Simulation::allocateDeviceMemory() {
    cudaMalloc(&d_bodies, maxBodies * sizeof(Body));
}

void Simulation::startSimulationThread() {
    pthread_create(&simulationThread, NULL, simulationThreadFunc, this);
}

void* Simulation::simulationThreadFunc(void* arg) {
    Simulation* sim = static_cast<Simulation*>(arg);
    int blockSize = 256;
    int numBlocks = (sim->maxBodies + blockSize - 1) / blockSize;
    while (true) {
        updateBodiesKernel<<<numBlocks, blockSize>>>(sim->d_bodies, sim->maxBodies);
        cudaDeviceSynchronize();
        cudaMemcpy(sim->bodies, sim->d_bodies, sim->maxBodies * sizeof(Body), cudaMemcpyDeviceToHost);
        usleep(1000);
    }
    return nullptr;
}

__global__ void Simulation::updateBodiesKernel(Body* bodies, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n && bodies[i].active) {
        double3 force = {0.0, 0.0, 0.0};
        for (int j = 0; j < n; j++) {
            if (i != j && bodies[j].active) {
                double3 r;
                r.x = bodies[j].position.x - bodies[i].position.x;
                r.y = bodies[j].position.y - bodies[i].position.y;
                r.z = bodies[j].position.z - bodies[i].position.z;

                double distSqr = r.x * r.x + r.y * r.y + r.z * r.z;
                double F = G * bodies[i].mass * bodies[j].mass *
                    rsqrt(distSqr * distSqr * distSqr);
                force.x += F * r.x;
                force.y += F * r.y;
                force.z += F * r.z;
            }
        }

        bodies[i].velocity.x += DT * force.x / bodies[i].mass;
        bodies[i].velocity.y += DT * force.y / bodies[i].mass;
        bodies[i].velocity.z += DT * force.z / bodies[i].mass;

        bodies[i].position.x += DT * bodies[i].velocity.x;
        bodies[i].position.y += DT * bodies[i].velocity.y;
        bodies[i].position.z += DT * bodies[i].velocity.z;
    }
}

void Simulation::updateBodies() {
    // Update logic if required outside of kernel
}

int Simulation::getBodyIndexByName(const char* name) const {
    for (int i = 0; i < activeBodyCount; i++) {
        if (bodies[i].active && strcmp(bodies[i].name, name) == 0) {
            return i;
        }
    }
    return -1;
}

Body* Simulation::getBodies() const {
    return bodies;
}

int Simulation::getActiveBodyCount() const {
    return activeBodyCount;
}

void Simulation::addBody(const Body& body) {
    if (activeBodyCount >= maxBodies) {
        printf("Warning: Maximum number of bodies reached. Ignoring new body.\n");
        return;
    }
    bodies[activeBodyCount] = body;
    bodies[activeBodyCount].active = true;
    activeBodyCount++;
    updateDeviceMemory();
}

void Simulation::updateBody(int index, const BodyParams& params) {
    if (params.position) bodies[index].position = *params.position;
    if (params.velocity) bodies[index].velocity = *params.velocity;
    if (params.mass) bodies[index].mass = *params.mass;
    if (params.radius) bodies[index].radius = *params.radius;
    if (params.color) bodies[index].color = *params.color;
    bodies[index].parentIndex = params.parentIndex;
    updateDeviceMemory();
}

void Simulation::updateDeviceMemory() {
    cudaMemcpy(d_bodies, bodies, maxBodies * sizeof(Body), cudaMemcpyHostToDevice);
}

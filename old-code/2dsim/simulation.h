class Simulation {
public:
    // Constructor
    Simulation(int maxBodies, double timeStep);

    // Destructor
    ~Simulation();

    // Initialization functions
    void initializeSharedMemory();
    void allocateDeviceMemory();
    void startSimulationThread();

    // Simulation update function
    void updateBodies();

    // Accessor functions
    int getBodyIndexByName(const char* name) const;
    Body* getBodies() const;
    int getActiveBodyCount() const;
    void addBody(const Body& body);
    void updateBody(int index, const BodyParams& params);

private:
    // Data members
    Body* bodies;
    Body* d_bodies;
    int activeBodyCount;
    int maxBodies;
    double timeStep;
    pthread_t simulationThread;

    // CUDA kernel function
    static __global__ void updateBodiesKernel(Body* bodies, int n);
    
    // Simulation thread function
    static void* simulationThreadFunc(void* arg);

    // Helper functions
    void initializeBodies();
    void updateDeviceMemory();
};

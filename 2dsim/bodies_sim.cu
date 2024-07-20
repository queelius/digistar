#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <pthread.h>
#include <unistd.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <math.h>

#define DEBUG true
#define N 3  // Number of bodies: Sun, Earth, Moon
#define G 6.67430e-11  // Gravitational constant in m³ kg⁻¹ s⁻²
#define DT 100  // Time step in seconds

// Define the Body structure
struct Body {
    double3 position;
    double3 velocity;
    float3 color;
    double mass;
    double radius;
};

// Define the Octree node structure
struct OctreeNode {
    double3 center;    // Center of this node
    double size;       // Size of the node
    double3 massCenter; // Center of mass of the bodies in this node
    double mass;       // Total mass of the bodies in this node
    int bodyIndex; // Index of the body in this node (-1 if empty)
    OctreeNode* children[8]; // Pointers to child nodes

    __device__ __host__ OctreeNode(double3 center, double size)
        : center(center), size(size), massCenter(make_double3(0, 0, 0)), mass(0), bodyIndex(-1) {
        for (int i = 0; i < 8; ++i) {
            children[i] = nullptr;
        }
    }

    __device__ __host__ bool isLeaf() const {
        for (int i = 0; i < 8; ++i) {
            if (children[i] != nullptr) {
                return false;
            }
        }
        return true;
    }
};

// Globals
Body* bodies;
Body* d_bodies;
OctreeNode* d_nodes;
int* d_bodyIndices;
int blockSize = 256;  // Adjust block size as needed

// Function prototypes
__device__ void insertBody(OctreeNode* node, int bodyIndex, const Body* bodies);
__device__ int getChildIndex(const OctreeNode* node, const double3& pos);
__global__ void constructTree(OctreeNode* nodes, Body* bodies, int numBodies);
__global__ void updateBodies(Body* bodies, int n);
__global__ void calculateForces(Body* bodies, OctreeNode* nodes, int numBodies, double theta);
void* simulationThread(void* arg);

// Main function
int main() {
    // Use shm_open to create a shared memory segment
    int shm_fd = shm_open("/bodies", O_CREAT | O_RDWR, 0666);
    if (shm_fd == -1) {
        perror("shm_open");
        return 1;
    }
    if (ftruncate(shm_fd, N * sizeof(Body)) == -1) {
        perror("ftruncate");
        return 1;
    }
    bodies = (Body*)mmap(NULL, N * sizeof(Body), PROT_READ | PROT_WRITE, MAP_SHARED, shm_fd, 0);
    if (bodies == MAP_FAILED) {
        perror("mmap");
        return 1;
    }

    cudaMalloc(&d_bodies, N * sizeof(Body));
    cudaMalloc(&d_nodes, N * sizeof(OctreeNode));
    cudaMalloc(&d_bodyIndices, N * sizeof(int));

    populate_sol();

    cudaMemcpy(d_bodies, bodies, N * sizeof(Body), cudaMemcpyHostToDevice);

    pthread_t thread;
    pthread_create(&thread, NULL, simulationThread, NULL);

    printf("Simulation running...\n");

    // Keep the main thread alive to allow queries
    while (true) {
        if (DEBUG) {
            for (int i = 0; i < N; i++) {
                printf("Body %d\n", i);
                printf("\tPosition: (%f, %f, %f)\n", bodies[i].position.x, bodies[i].position.y, bodies[i].position.z);
                printf("\tVelocity: (%f, %f, %f)\n", bodies[i].velocity.x, bodies[i].velocity.y, bodies[i].velocity.z);
                printf("\tMass: %f kg\n", bodies[i].mass);
                printf("\tRadius: %f m\n", bodies[i].radius);
                printf("\n");                
            }
        }
        sleep(1);       
    }

    return 0;
}

// Insert a body into the octree
__device__ void insertBody(OctreeNode* node, int bodyIndex, const Body* bodies) {
    if (node->isLeaf() && node->bodyIndex == -1) {
        node->bodyIndex = bodyIndex;
    } else {
        if (node->isLeaf()) {
            int oldBodyIndex = node->bodyIndex;
            node->bodyIndex = -1;

            for (int i = 0; i < 8; ++i) {
                double3 offset = make_double3(
                    node->size / 4 * ((i & 1) ? 1 : -1),
                    node->size / 4 * ((i & 2) ? 1 : -1),
                    node->size / 4 * ((i & 4) ? 1 : -1)
                );
                double3 childCenter = make_double3(
                    node->center.x + offset.x,
                    node->center.y + offset.y,
                    node->center.z + offset.z
                );
                node->children[i] = new OctreeNode(childCenter, node->size / 2);
            }

            insertBody(node->children[getChildIndex(node, bodies[oldBodyIndex].position)], oldBodyIndex, bodies);
        }
        insertBody(node->children[getChildIndex(node, bodies[bodyIndex].position)], bodyIndex, bodies);
    }
}

// Get the index of the child node that contains the given position
__device__ int getChildIndex(const OctreeNode* node, const double3& pos) {
    int index = 0;
    if (pos.x > node->center.x) index |= 1;
    if (pos.y > node->center.y) index |= 2;
    if (pos.z > node->center.z) index |= 4;
    return index;
}

// Kernel to construct the tree
__global__ void constructTree(OctreeNode* nodes, Body* bodies, int numBodies) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numBodies) {
        insertBody(&nodes[0], idx, bodies);
    }
}

// Kernel to update the bodies
__global__ void updateBodies(Body* bodies, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        double3 force = make_double3(0.0, 0.0, 0.0);
        for (int j = 0; j < n; j++) {
            if (i != j) {
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

// Kernel to calculate forces using the octree
__global__ void calculateForces(Body* bodies, OctreeNode* nodes, int numBodies, double theta) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < numBodies) {
        double3 force = make_double3(0.0, 0.0, 0.0);
        // Traverse the octree and calculate the force on the body
        // Use an appropriate theta value to determine when to approximate
        // This part needs to be filled in based on the specifics of your octree structure
        bodies[i].velocity.x += DT * force.x / bodies[i].mass;
        bodies[i].velocity.y += DT * force.y / bodies[i].mass;
        bodies[i].velocity.z += DT * force.z / bodies[i].mass;
    }
}

// Simulation thread function
void* simulationThread(void* arg) {
    int numBlocks = (N + blockSize - 1) / blockSize;

    while (true) {
        // Construct the tree on the GPU
        constructTree<<<numBlocks, blockSize>>>(d_nodes, d_bodies, N);
        cudaDeviceSynchronize();

        // Calculate forces using the tree on the GPU
        calculateForces<<<numBlocks, blockSize>>>(d_bodies, d_nodes, N, 0.5);
        cudaDeviceSynchronize();

        // Update bodies
        updateBodies<<<numBlocks, blockSize>>>(d_bodies, N);
        cudaDeviceSynchronize();

        // Copy data back to the host
        cudaMemcpy(bodies, d_bodies, N * sizeof(Body), cudaMemcpyDeviceToHost);

        usleep(1000);  // Sleep for a short period to limit update rate
    }
}

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <pthread.h>
#include <unistd.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <math.h>

#define N 3     // Number of bodies: Sun, Earth, Moon
#define G 6.7e-9 // G = 6.67430e-11 but we use diff units (see units.md)
#define DT 6e-5  // Time step (DT is in TU = 1e6 s, 1 min in TU = 6e-5)

struct Body {
    float3 position;
    float3 velocity;
    float mass;
};

Body* bodies;
Body* d_bodies;

__global__ void updateBodies(Body* bodies, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float3 force = {0.0f, 0.0f, 0.0f};
        for (int j = 0; j < n; j++) {
            if (i != j) {
                float3 r;
                r.x = bodies[j].position.x - bodies[i].position.x;
                r.y = bodies[j].position.y - bodies[i].position.y;
                r.z = bodies[j].position.z - bodies[i].position.z;

                float distSqr = r.x * r.x + r.y * r.y + r.z * r.z + 1e-10f;
                float dist = sqrtf(distSqr);
                float invDist = 1.0f / dist;
                float invDist3 = invDist * invDist * invDist;

                force.x += G * bodies[j].mass * invDist3 * r.x;
                force.y += G * bodies[j].mass * invDist3 * r.y;
                force.z += G * bodies[j].mass * invDist3 * r.z;
            }
        }

        bodies[i].velocity.x += DT * force.x / bodies[i].mass;
        bodies[i].velocity.y += DT * force.y / bodies[i].mass;
        bodies[i].velocity.z += DT * force.z / bodies[i].mass;

        bodies[i].position.x += DT * bodies[i].velocity.x;
        bodies[i].position.y += DT * bodies[i].velocity.y;
        bodies[i].position.z += DT * bodies[i].velocity.z;

        // Debugging output
        //printf("Body %d: Position (%f, %f, %f) Velocity (%f, %f, %f) Force (%f, %f, %f)\n",
        //       i, bodies[i].position.x, bodies[i].position.y, bodies[i].position.z,
        //       bodies[i].velocity.x, bodies[i].velocity.y, bodies[i].velocity.z,
        //       force.x, force.y, force.z);
    }
}

void* simulationThread(void* arg) {
    int blockSize = 3;  // Since we only have 3 bodies
    int numBlocks = 1;  // We can fit all bodies in one block

    while (true) {
        updateBodies<<<numBlocks, blockSize>>>(d_bodies, N);
        cudaDeviceSynchronize();
        cudaMemcpy(bodies, d_bodies, N * sizeof(Body), cudaMemcpyDeviceToHost);
        usleep(1);  // Sleep for a short period to limit update rate
    }
}

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

    // Initialize Sun, Earth, and Moon
    // Sun
    bodies[0].position = {0.0f, 0.0f, 0.0f};  // Center of the system
    bodies[0].velocity = {0.0f, 0.0f, 0.0f};  // Stationary
    bodies[0].mass = 1.989e10;  // 1 Solar mass in MU

    // Earth
    bodies[1].position = {1.0f, 0.0f, 0.0f};  // 1 AU from Sun on x-axis
    bodies[1].velocity = {0.0f, 3.651e-2f, 0.0f};  // Orbital velocity
    bodies[1].mass = 5.972e4;  // Earth mass in MU

    // Moon
    bodies[2].position = {1.00257f, 0.0f, 0.0f};  // About 384,400 km from Earth
    bodies[2].velocity = {0.0f, 4.898e-2f, 0.0f};  // Orbital velocity around Earth
    bodies[2].mass = 7.34e2;  // Moon mass in MU

    cudaMemcpy(d_bodies, bodies, N * sizeof(Body), cudaMemcpyHostToDevice);

    pthread_t thread;
    pthread_create(&thread, NULL, simulationThread, NULL);

    printf("Simulation running...\n");

    // Keep the main thread alive to allow queries
    while (true) {
        printf("Sun  position: (%f, %f, %f)\n", bodies[0].position.x, bodies[0].position.y, bodies[0].position.z);
        printf("Earth position: (%f, %f, %f)\n", bodies[1].position.x, bodies[1].position.y, bodies[1].position.z);
        printf("Moon  position: (%f, %f, %f)\n", bodies[2].position.x, bodies[2].position.y, bodies[2].position.z);
        sleep(1);
    }

    return 0;
}
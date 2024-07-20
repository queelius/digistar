#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <pthread.h>
#include <unistd.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <math.h>
#include <fstream>
#include <dirent.h>
#include "json.hpp"

using json = nlohmann::json;

#define DEBUG true
#define N 1000     // max number of bodies
#define G 6.67430e-11 // Gravitational constant in m³ kg⁻¹ s⁻²
#define DT 1  // Time step in seconds

struct Body {
    double3 position;
    double3 velocity;
    float3 color;
    double mass;
    double radius;
    char name[50];
    bool active;
    int parentIndex; // -1 if no parent
};

Body* bodies;
Body* d_bodies;
int activeBodyCount = 0;

__global__ void updateBodies(Body* bodies, int n) {
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

void* simulationThread(void* arg) {
    int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize;

    while (true) {
        updateBodies<<<numBlocks, blockSize>>>(d_bodies, N);
        cudaDeviceSynchronize();
        cudaMemcpy(bodies, d_bodies, N * sizeof(Body), cudaMemcpyDeviceToHost);
        usleep(1000);
    }
}

void addBody(const char* name,
             double3 position,
             double3 velocity,
             double mass,
             double radius,
             float3 color,
             int parentIndex = -1) {

    if (activeBodyCount >= N) {
        printf("Warning: Maximum number of bodies reached. Ignoring new body.\n");
        return;
    }

    int i = activeBodyCount;
    strcpy(bodies[i].name, name);
    bodies[i].position = position;
    bodies[i].velocity = velocity;
    bodies[i].mass = mass;
    bodies[i].radius = radius;
    bodies[i].color = color;
    bodies[i].active = true;
    bodies[i].parentIndex = parentIndex;
    ++activeBodyCount;
}

int get_body_by_name(const char* name) {
    for (int i = 0; i < N; i++) {
        if (bodies[i].active && strcmp(bodies[i].name, name) == 0) {
            return i;
        }
    }
    return -1;
}

void loadBodiesFromJson(
    const std::string& filename,
    double3 position_offset = {0.0, 0.0, 0.0},
    double3 velocity_offset = {0.0, 0.0, 0.0}) {

    std::ifstream file(filename);
    json j;
    file >> j;

    double3 parPos = position_offset;
    double3 parVel = velocity_offset;
    int parIdx = -1;

    if (j.contains("parent")) {
        const auto& par = j["parent"];
        const char* parentName = par["name"].get<std::string>().c_str();
        // check to make sure parentName is not already in bodies. if so,
        // we use the parent's position and velocity as the offset and we
        // do not touch the parent
        parIdx = get_body_by_name(parentName);
        if (parIdx != -1) {
            parPos = bodies[parIdx].position;
            parVel = bodies[parIdx].velocity;
        }
        else {
            // update parent position and velocity
            double3 relPos = make_double3(par["position"][0].get<double>(),
                                          par["position"][1].get<double>(),
                                          par["position"][2].get<double>());
            double3 relVel = make_double3(par["velocity"][0].get<double>(),
                                          par["velocity"][1].get<double>(),
                                          par["velocity"][2].get<double>());
            addBody(parentName,
                    make_double3(parPos.x + relPos.x,
                                 parPos.y + relPos.y,
                                 parPos.z + relPos.z),
                    make_double3(parVel.x + relVel.x,
                                 parVel.y + relVel.y,
                                 parVel.z + relVel.z),
                    par["mass"].get<double>(),
                    par["radius"].get<double>(),
                    make_float3(par["color"][0].get<float>(),
                                par["color"][1].get<float>(),
                                par["color"][2].get<float>()));
            parIdx = activeBodyCount - 1;
        }
    }

    for (const auto& body : j["children"]) {
        const char* childName = body["name"].get<std::string>().c_str();
        int childIdx = get_body_by_name(childName);
        // warn if child already exists
        if (childIdx != -1) {
            printf("Warning: Body with name %s already exists. Ignoring new body.\n", childName);
            continue;
        }
        addBody(childName,
                make_double3(parPos.x + body["position"][0].get<double>(),
                             parPos.y + body["position"][1].get<double>(),
                             parPos.z + body["position"][2].get<double>()),
                make_double3(parVel.x + body["velocity"][0].get<double>(),
                             parVel.y + body["velocity"][1].get<double>(),
                             parVel.z + body["velocity"][2].get<double>()),
                body["mass"].get<double>(),
                body["radius"].get<double>(),
                make_float3(body["color"][0].get<float>(),
                            body["color"][1].get<float>(),
                            body["color"][2].get<float>()),
                parIdx);
    }

    // Update device memory
    cudaMemcpy(d_bodies, bodies, N * sizeof(Body), cudaMemcpyHostToDevice);
}

void loadBodiesFromJsonDir(const std::string& dirPath) {
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

    for (int i = 0; i < N; i++) {
        bodies[i].active = false;
    }

    cudaMalloc(&d_bodies, N * sizeof(Body));
    loadBodiesFromJsonDir("./solar_system");
    //loadBodiesFromJson("./solar_system/main_planets.json");
    //loadBodiesFromJson("./solar_system/jupiter.json");
    cudaMemcpy(d_bodies, bodies, N * sizeof(Body), cudaMemcpyHostToDevice);

    pthread_t thread;
    pthread_create(&thread, NULL, simulationThread, NULL);

    printf("[simulation]: running with %d bodies\n", activeBodyCount);

    // Keep the main thread alive to allow queries
    while (true) {
        if (DEBUG) {
            for (int i = 0; i < N; i++) {
                if (!bodies[i].active) {
                    continue;
                }
                // make sure to show in scientific notation
                printf("Name: %s\n", bodies[i].name);
                printf("\tPosition: (%f, %f, %f)\n", bodies[i].position.x, bodies[i].position.y, bodies[i].position.z);
                printf("\tVelocity: (%f, %f, %f)\n", bodies[i].velocity.x, bodies[i].velocity.y, bodies[i].velocity.z);
                printf("\tMass: %f kg\n", bodies[i].mass);
                printf("\tRadius: %f m\n", bodies[i].radius);
                printf("\n");                
            }
        }
        sleep(10);       
    }

    return 0;
}
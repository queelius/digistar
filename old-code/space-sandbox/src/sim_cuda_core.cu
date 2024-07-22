#include "sim_cuda_core.h"
#include <unistd.h>
#include <cstring>

__global__ void updateGravityFieldKernel(Body *bodies, int n, double grav)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n && bodies[i].active)
    {
        for (int j = 0; j < n; j++)
        {
            if (i != j && bodies[j].active)
            {
                double3 r;
                r.x = bodies[j].position.x - bodies[i].position.x;
                r.y = bodies[j].position.y - bodies[i].position.y;
                r.z = bodies[j].position.z - bodies[i].position.z;

                double distSqr = r.x * r.x + r.y * r.y + r.z * r.z;
                double dist = sqrt(distSqr);
                double invDist = 1.0 / dist;
                double invDist3 = invDist * invDist * invDist;
                double F = grav * bodies[i].mass * bodies[j].mass * invDist3;
                bodies[i].force.x += F * r.x;
                bodies[i].force.y += F * r.y;
                bodies[i].force.z += F * r.z;
            }
        }
    }
}

__global__ void updateElectricFieldKernel(Body *bodies, int n, double coloumb)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n && bodies[i].active)
    {
        for (int j = 0; j < n; j++)
        {
            if (i != j && bodies[j].active)
            {
                double3 r;
                r.x = bodies[j].position.x - bodies[i].position.x;
                r.y = bodies[j].position.y - bodies[i].position.y;
                r.z = bodies[j].position.z - bodies[i].position.z;

                double distSqr = r.x * r.x + r.y * r.y + r.z * r.z;
                double dist = sqrt(distSqr);
                double invDist = 1.0 / dist;
                double invDist3 = invDist * invDist * invDist;
                double F = coloumb * bodies[i].charge * bodies[j].charge * invDist3;
                bodies[i].force.x += F * r.x;
                bodies[i].force.y += F * r.y;
                bodies[i].force.z += F * r.z;
            }
        }
    }
}

__global__ void intergateKernel(
    Body *bodies,
    int n,
    double timeStep)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n && bodies[i].active)
    {
        bodies[i].velocity.x += timeStep * bodies[i].force.x / bodies[i].mass;
        bodies[i].velocity.y += timeStep * bodies[i].force.y / bodies[i].mass;
        bodies[i].velocity.z += timeStep * bodies[i].force.z / bodies[i].mass;

        bodies[i].position.x += timeStep * bodies[i].velocity.x;
        bodies[i].position.y += timeStep * bodies[i].velocity.y;
        bodies[i].position.z += timeStep * bodies[i].velocity.z;

        bodies[i].force.x = 0;
        bodies[i].force.y = 0;
        bodies[i].force.z = 0;
    }
}

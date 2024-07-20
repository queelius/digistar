#include "common.h"

__global__ void updateGravityFieldKernel(
    Body *bodies,
    int n,
    double gravitationalConstant);

__global__ void updateElectricFieldKernel(
    Body *bodies,
    int n,
    double coloumbsConstant);

__global__ void intergateKernel(
    Body *bodies,
    int n,
    double timeStep);
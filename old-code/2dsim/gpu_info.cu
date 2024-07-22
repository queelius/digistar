#include <cuda_runtime.h>
#include <stdio.h>

// Function declaration
int _ConvertSMVer2Cores(int major, int minor);

int main() {
    int deviceCount;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    if (err != cudaSuccess) {
        printf("cudaGetDeviceCount error: %s\n", cudaGetErrorString(err));
        return 1;
    }

    for (int device = 0; device < deviceCount; ++device) {
        cudaDeviceProp deviceProp;
        err = cudaGetDeviceProperties(&deviceProp, device);
        if (err != cudaSuccess) {
            printf("cudaGetDeviceProperties error: %s\n", cudaGetErrorString(err));
            return 1;
        }

        printf("Device %d: %s\n", device, deviceProp.name);
        printf("  Compute capability: %d.%d\n", deviceProp.major, deviceProp.minor);
        printf("  Total global memory: %lu bytes\n", deviceProp.totalGlobalMem);
        printf("  Number of multiprocessors: %d\n", deviceProp.multiProcessorCount);
        printf("  Number of CUDA cores: %d\n", deviceProp.multiProcessorCount * _ConvertSMVer2Cores(deviceProp.major, deviceProp.minor));
        printf("  Total constant memory: %lu bytes\n", deviceProp.totalConstMem);
        printf("  Total shared memory per block: %lu bytes\n", deviceProp.sharedMemPerBlock);
        printf("  Total registers per block: %d\n", deviceProp.regsPerBlock);
        printf("  Warp size: %d\n", deviceProp.warpSize);
        printf("  Maximum number of threads per multiprocessor: %d\n", deviceProp.maxThreadsPerMultiProcessor);
        printf("  Maximum number of threads per block: %d\n", deviceProp.maxThreadsPerBlock);
        printf("  Maximum sizes of each dimension of a block: %d x %d x %d\n",
               deviceProp.maxThreadsDim[0],
               deviceProp.maxThreadsDim[1],
               deviceProp.maxThreadsDim[2]);
        printf("  Maximum sizes of each dimension of a grid: %d x %d x %d\n",
               deviceProp.maxGridSize[0],
               deviceProp.maxGridSize[1],
               deviceProp.maxGridSize[2]);
        printf("  Clock rate: %d kHz\n", deviceProp.clockRate);
        printf("  Memory clock rate: %d kHz\n", deviceProp.memoryClockRate);
        printf("  Memory bus width: %d bits\n", deviceProp.memoryBusWidth);
        printf("  L2 cache size: %d bytes\n", deviceProp.l2CacheSize);
    }

    return 0;
}

int _ConvertSMVer2Cores(int major, int minor) {
    typedef struct {
        int SM;
        int Cores;
    } sSMtoCores;

    sSMtoCores nGpuArchCoresPerSM[] = {
        { 0x30, 192 }, // Kepler Generation
        { 0x32, 192 }, // Kepler Generation
        { 0x35, 192 }, // Kepler Generation
        { 0x37, 192 }, // Kepler Generation
        { 0x50, 128 }, // Maxwell Generation
        { 0x52, 128 }, // Maxwell Generation
        { 0x53, 128 }, // Maxwell Generation
        { 0x60, 64  }, // Pascal Generation
        { 0x61, 128 }, // Pascal Generation
        { 0x62, 128 }, // Pascal Generation
        { 0x70, 64  }, // Volta Generation
        { 0x72, 64  }, // Volta Generation
        { 0x75, 64  }, // Turing Generation
        { 0x80, 64  }, // Ampere Generation
        { 0x86, 128 }, // Ampere Generation
        { -1, -1 }
    };

    int index = 0;
    while (nGpuArchCoresPerSM[index].SM != -1) {
        if (nGpuArchCoresPerSM[index].SM == ((major << 4) + minor)) {
            return nGpuArchCoresPerSM[index].Cores;
        }
        index++;
    }
    printf("MapSMtoCores undefined SMversion %d.%d!\n", major, minor);
    return -1;
}

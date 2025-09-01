#include "backend_interface.h"
#include "cpu_backend_simple.h"
#include "cpu_backend_reference.h"
#include "cpu_backend_openmp.h"
#ifdef CUDA_ENABLED
#include "CUDABackend.h"
#endif

#include <memory>
#include <thread>

namespace digistar {

std::unique_ptr<IBackend> BackendFactory::create(Type type, const SimulationConfig& config) {
    switch (type) {
        case Type::CPU:
            // Use simple backend for basic CPU
            return std::make_unique<CpuBackendSimple>();
            
        case Type::CPU_SIMD:
            // Use OpenMP backend for SIMD/parallel CPU
            return std::make_unique<CpuBackendOpenMP>();
            
#ifdef CUDA_ENABLED
        case Type::CUDA:
            return std::make_unique<CUDABackend>();
#endif
            
        case Type::OpenCL:
            // Not implemented yet, fall back to CPU
            return std::make_unique<CpuBackendSimple>();
            
        case Type::Distributed:
            // Not implemented yet, fall back to CPU
            return std::make_unique<CpuBackendSimple>();
            
        default:
            // Default to simple CPU backend
            return std::make_unique<CpuBackendSimple>();
    }
}

bool BackendFactory::isSupported(Type type) {
    switch (type) {
        case Type::CPU:
        case Type::CPU_SIMD:
            return true;
            
#ifdef CUDA_ENABLED
        case Type::CUDA:
            // Would need to check for CUDA runtime here
            return true;
#else
        case Type::CUDA:
            return false;
#endif
            
        case Type::OpenCL:
        case Type::Distributed:
            return false;  // Not implemented yet
            
        default:
            return false;
    }
}

std::string BackendFactory::getDescription(Type type) {
    switch (type) {
        case Type::CPU:
            return "Simple CPU (single-threaded)";
        case Type::CPU_SIMD:
            return "CPU with OpenMP (multi-threaded)";
        case Type::CUDA:
            return "NVIDIA CUDA GPU";
        case Type::OpenCL:
            return "OpenCL (not implemented)";
        case Type::Distributed:
            return "Distributed (not implemented)";
        default:
            return "Unknown";
    }
}

int BackendFactory::getNumCPUCores() {
    return std::thread::hardware_concurrency();
}

} // namespace digistar
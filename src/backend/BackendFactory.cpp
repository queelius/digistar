#include "ISimulationBackend.h"
#include <memory>
#include <iostream>
#include <thread>

// Include backend implementations
#include "SimpleBackend_v3.cpp"
// TODO: Update these to new interface
// #include "SSE2Backend.cpp"
// #include "AVX2Backend.cpp"

// Platform detection
#ifdef __AVX2__
    #define HAS_AVX2 1
#else
    #define HAS_AVX2 0
#endif

#ifdef __SSE2__
    #define HAS_SSE2 1
#else  
    #define HAS_SSE2 1  // All x86-64 has SSE2
#endif

// Check for CUDA at runtime
bool BackendFactory::hasCUDA() {
    // TODO: Implement actual CUDA detection
    // For now, return false since CUDA backend isn't implemented
    return false;
}

// Check for AVX2 at runtime
bool BackendFactory::hasAVX2() {
    #ifdef _MSC_VER
        int cpuinfo[4];
        __cpuidex(cpuinfo, 7, 0);
        return (cpuinfo[1] & (1 << 5)) != 0;  // AVX2 bit
    #else
        __builtin_cpu_init();
        return __builtin_cpu_supports("avx2");
    #endif
}

// Get number of CPU cores
int BackendFactory::getNumCPUCores() {
    return std::thread::hardware_concurrency();
}

// Get best backend for given requirements
BackendType BackendFactory::recommendBackend(
    size_t num_particles,
    ForceAlgorithm algorithm
) {
    // For large particle counts, strongly prefer GPU
    if (num_particles > 100000 && hasCUDA()) {
        return BackendType::CUDA;
    }
    
    // For medium counts, use AVX2 if available
    if (num_particles > 10000 && hasAVX2()) {
        return BackendType::AVX2;
    }
    
    // For small counts or if nothing else available
    return BackendType::CPU;
}

// Create the appropriate backend
std::unique_ptr<ISimulationBackend> BackendFactory::create(
    BackendType type,
    ForceAlgorithm algorithm,
    size_t target_particles
) {
    // Override with specific type if requested
    if (type == BackendType::CUDA) {
        if (hasCUDA()) {
            std::cout << "Creating CUDA backend..." << std::endl;
            // TODO: return std::make_unique<CUDABackend>();
            std::cerr << "CUDA backend not yet implemented!" << std::endl;
            type = BackendType::AUTO;
        } else {
            std::cerr << "CUDA requested but not available!" << std::endl;
            type = BackendType::AUTO;
        }
    }
    
    if (type == BackendType::AVX2) {
        if (hasAVX2()) {
            std::cout << "Creating AVX2 backend..." << std::endl;
            // TODO: Update AVX2Backend to new interface
            // return std::make_unique<AVX2Backend>();
            std::cout << "AVX2 backend not yet updated, using Simple backend" << std::endl;
            return std::make_unique<SimpleBackend>();
        } else {
            std::cerr << "AVX2 requested but not supported!" << std::endl;
            type = BackendType::AUTO;
        }
    }
    
    if (type == BackendType::SSE2) {
        std::cout << "Creating SSE2 backend..." << std::endl;
        // TODO: Update SSE2Backend to new interface
        // return std::make_unique<SSE2Backend>();
        std::cout << "SSE2 backend not yet updated, using Simple backend" << std::endl;
        return std::make_unique<SimpleBackend>();
    }
    
    if (type == BackendType::CPU) {
        std::cout << "Creating Simple CPU backend..." << std::endl;
        return std::make_unique<SimpleBackend>();
    }
    
    // AUTO mode - pick best available
    if (type == BackendType::AUTO) {
        // For large particle counts with Barnes-Hut, AVX2 is great
        if (algorithm == ForceAlgorithm::BARNES_HUT && hasAVX2()) {
            std::cout << "AUTO: Selected AVX2 backend for Barnes-Hut" << std::endl;
            // TODO: return std::make_unique<AVX2Backend>();
            return std::make_unique<SimpleBackend>();
        }
        
        // For brute force with medium counts, AVX2 helps a lot
        if (algorithm == ForceAlgorithm::BRUTE_FORCE && target_particles < 10000 && hasAVX2()) {
            std::cout << "AUTO: Selected AVX2 backend for " 
                      << target_particles << " particles" << std::endl;
            // TODO: return std::make_unique<AVX2Backend>();
            return std::make_unique<SimpleBackend>();
        }
        
        // Default fallback
        std::cout << "AUTO: Selected Simple CPU backend (default)" << std::endl;
        return std::make_unique<SimpleBackend>();
    }
    
    // Should never reach here
    return std::make_unique<SimpleBackend>();
}
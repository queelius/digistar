#include "event_system.h"

#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <cstring>
#include <stdexcept>
#include <iostream>

namespace digistar {

namespace {
    /**
     * Round up to nearest power of 2
     */
    size_t round_up_to_power_of_2(size_t n) {
        if (n == 0) return 1;
        n--;
        n |= n >> 1;
        n |= n >> 2;
        n |= n >> 4;
        n |= n >> 8;
        n |= n >> 16;
        if constexpr (sizeof(size_t) > 4) {
            n |= n >> 32;
        }
        return n + 1;
    }
}

SharedMemoryEventSystem::SharedMemoryEventSystem(const std::string& shm_name, bool create_new)
    : shm_name_(shm_name), shm_fd_(-1), buffer_(nullptr), 
      buffer_size_(sizeof(EventRingBuffer)), is_owner_(create_new) {
    
    try {
        if (create_new) {
            create_shared_memory();
        } else {
            attach_shared_memory();
        }
    } catch (...) {
        cleanup();
        throw;
    }
}

SharedMemoryEventSystem::~SharedMemoryEventSystem() {
    cleanup();
}

void SharedMemoryEventSystem::create_shared_memory() {
    // Create shared memory segment
    shm_fd_ = shm_open(shm_name_.c_str(), O_CREAT | O_EXCL | O_RDWR, S_IRUSR | S_IWUSR);
    if (shm_fd_ == -1) {
        if (errno == EEXIST) {
            throw std::runtime_error("Shared memory segment already exists: " + shm_name_);
        }
        throw std::runtime_error("Failed to create shared memory: " + std::string(strerror(errno)));
    }
    
    // Set the size
    if (ftruncate(shm_fd_, buffer_size_) == -1) {
        throw std::runtime_error("Failed to set shared memory size: " + std::string(strerror(errno)));
    }
    
    // Map the memory
    void* ptr = mmap(nullptr, buffer_size_, PROT_READ | PROT_WRITE, MAP_SHARED, shm_fd_, 0);
    if (ptr == MAP_FAILED) {
        throw std::runtime_error("Failed to map shared memory: " + std::string(strerror(errno)));
    }
    
    // Initialize the buffer using placement new
    buffer_ = new(ptr) EventRingBuffer();
    
    std::cout << "Created shared memory event system: " << shm_name_ 
              << " (size: " << buffer_size_ << " bytes)" << std::endl;
}

void SharedMemoryEventSystem::attach_shared_memory() {
    // Open existing shared memory segment
    shm_fd_ = shm_open(shm_name_.c_str(), O_RDWR, 0);
    if (shm_fd_ == -1) {
        throw std::runtime_error("Failed to open shared memory: " + std::string(strerror(errno)));
    }
    
    // Get the size
    struct stat sb;
    if (fstat(shm_fd_, &sb) == -1) {
        throw std::runtime_error("Failed to get shared memory size: " + std::string(strerror(errno)));
    }
    
    if (static_cast<size_t>(sb.st_size) != buffer_size_) {
        throw std::runtime_error("Shared memory size mismatch");
    }
    
    // Map the memory
    void* ptr = mmap(nullptr, buffer_size_, PROT_READ | PROT_WRITE, MAP_SHARED, shm_fd_, 0);
    if (ptr == MAP_FAILED) {
        throw std::runtime_error("Failed to map shared memory: " + std::string(strerror(errno)));
    }
    
    buffer_ = static_cast<EventRingBuffer*>(ptr);
    
    // Validate the buffer
    if (!buffer_->is_valid()) {
        throw std::runtime_error("Invalid shared memory buffer");
    }
    
    std::cout << "Attached to shared memory event system: " << shm_name_ << std::endl;
}

void SharedMemoryEventSystem::cleanup() {
    if (buffer_ && buffer_ != MAP_FAILED) {
        munmap(buffer_, buffer_size_);
        buffer_ = nullptr;
    }
    
    if (shm_fd_ != -1) {
        close(shm_fd_);
        shm_fd_ = -1;
    }
    
    // Only unlink if we created the segment
    if (is_owner_ && !shm_name_.empty()) {
        shm_unlink(shm_name_.c_str());
    }
}

bool SharedMemoryEventSystem::is_valid() const {
    return buffer_ != nullptr && buffer_->is_valid();
}

SharedMemoryEventSystem::Stats SharedMemoryEventSystem::get_stats() const {
    if (!is_valid()) {
        return {0, 0, 0, 0.0f, 0};
    }
    
    return {
        buffer_->total_events.load(std::memory_order_relaxed),
        buffer_->dropped_events.load(std::memory_order_relaxed),
        buffer_->num_consumers.load(std::memory_order_relaxed),
        buffer_->get_utilization(),
        buffer_size_
    };
}

const char* event_system_error_string(EventSystemError error) {
    switch (error) {
        case EventSystemError::SUCCESS:
            return "Success";
        case EventSystemError::BUFFER_FULL:
            return "Buffer full - events were dropped";
        case EventSystemError::INVALID_CONSUMER:
            return "Invalid consumer ID";
        case EventSystemError::NO_EVENTS:
            return "No events available";
        case EventSystemError::SHARED_MEMORY_ERROR:
            return "Shared memory error";
        case EventSystemError::INVALID_BUFFER:
            return "Invalid buffer state";
        case EventSystemError::TOO_MANY_CONSUMERS:
            return "Too many consumers registered";
        default:
            return "Unknown error";
    }
}

} // namespace digistar
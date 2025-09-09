#pragma once

#include <vector>
#include <thread>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <functional>
#include <future>
#include <atomic>
#include <memory>

namespace digistar {
namespace dsl {

class ThreadPool {
private:
    // Worker threads
    std::vector<std::thread> workers;
    
    // Task queue
    std::queue<std::function<void()>> tasks;
    
    // Synchronization
    mutable std::mutex queue_mutex;
    std::condition_variable condition;
    std::atomic<bool> stop{false};
    std::atomic<bool> paused{false};
    
    // Statistics
    std::atomic<size_t> tasks_completed{0};
    std::atomic<size_t> tasks_pending{0};
    std::atomic<size_t> active_workers{0};
    
    // Worker function
    void worker_thread() {
        while (true) {
            std::function<void()> task;
            
            {
                std::unique_lock<std::mutex> lock(queue_mutex);
                
                // Wait for task or stop signal
                condition.wait(lock, [this] { 
                    return stop || (!paused && !tasks.empty()); 
                });
                
                if (stop && tasks.empty()) {
                    return;
                }
                
                if (!tasks.empty() && !paused) {
                    task = std::move(tasks.front());
                    tasks.pop();
                    tasks_pending--;
                    active_workers++;
                }
            }
            
            // Execute task outside of lock
            if (task) {
                task();
                tasks_completed++;
                active_workers--;
            }
        }
    }
    
public:
    // Constructor with configurable thread count
    // 0 = auto-detect (hardware_concurrency - 1)
    // 1 = single thread (good for CPU-only systems)
    // N = use N threads
    explicit ThreadPool(size_t num_threads = 0) {
        if (num_threads == 0) {
            // Auto-detect: use all cores minus one (leave one for main thread)
            num_threads = std::max(1u, std::thread::hardware_concurrency() - 1);
        }
        
        workers.reserve(num_threads);
        for (size_t i = 0; i < num_threads; ++i) {
            workers.emplace_back(&ThreadPool::worker_thread, this);
        }
    }
    
    // Destructor - wait for all tasks to complete
    ~ThreadPool() {
        shutdown();
    }
    
    // Submit a task and get a future for the result
    template<typename F, typename... Args>
    auto submit(F&& f, Args&&... args) -> std::future<decltype(f(args...))> {
        using return_type = decltype(f(args...));
        
        auto task = std::make_shared<std::packaged_task<return_type()>>(
            std::bind(std::forward<F>(f), std::forward<Args>(args)...)
        );
        
        std::future<return_type> result = task->get_future();
        
        {
            std::unique_lock<std::mutex> lock(queue_mutex);
            
            if (stop) {
                throw std::runtime_error("Cannot submit task to stopped thread pool");
            }
            
            tasks.emplace([task]() { (*task)(); });
            tasks_pending++;
        }
        
        condition.notify_one();
        return result;
    }
    
    // Submit a fire-and-forget task (no future)
    void submit_detached(std::function<void()> task) {
        {
            std::unique_lock<std::mutex> lock(queue_mutex);
            
            if (stop) {
                return;  // Silently drop if stopped
            }
            
            tasks.emplace(std::move(task));
            tasks_pending++;
        }
        
        condition.notify_one();
    }
    
    // Batch submit multiple tasks
    template<typename F>
    std::vector<std::future<void>> submit_batch(const std::vector<F>& batch_tasks) {
        std::vector<std::future<void>> futures;
        futures.reserve(batch_tasks.size());
        
        for (const auto& task : batch_tasks) {
            futures.push_back(submit(task));
        }
        
        return futures;
    }
    
    // Pause/resume execution
    void pause() {
        paused = true;
    }
    
    void resume() {
        paused = false;
        condition.notify_all();
    }
    
    // Wait for all pending tasks to complete
    void wait_all() {
        while (tasks_pending > 0 || active_workers > 0) {
            std::this_thread::yield();
        }
    }
    
    // Resize the thread pool
    void resize(size_t new_size) {
        if (new_size == workers.size()) {
            return;
        }
        
        if (new_size < workers.size()) {
            // Shrink: need to stop some workers
            size_t to_remove = workers.size() - new_size;
            
            // Signal workers to stop
            for (size_t i = 0; i < to_remove; ++i) {
                submit_detached([]() {
                    // Dummy task to wake up worker
                });
            }
            
            // Wait for workers to finish
            for (size_t i = workers.size() - to_remove; i < workers.size(); ++i) {
                if (workers[i].joinable()) {
                    workers[i].join();
                }
            }
            
            workers.resize(new_size);
        } else {
            // Grow: add more workers
            size_t old_size = workers.size();
            workers.reserve(new_size);
            
            for (size_t i = old_size; i < new_size; ++i) {
                workers.emplace_back(&ThreadPool::worker_thread, this);
            }
        }
    }
    
    // Shutdown the thread pool
    void shutdown() {
        {
            std::unique_lock<std::mutex> lock(queue_mutex);
            if (stop) return;  // Already stopped
            stop = true;
        }
        
        condition.notify_all();
        
        for (std::thread& worker : workers) {
            if (worker.joinable()) {
                worker.join();
            }
        }
        
        workers.clear();
    }
    
    // Get thread pool statistics
    size_t size() const { return workers.size(); }
    size_t pending_tasks() const { return tasks_pending.load(); }
    size_t completed_tasks() const { return tasks_completed.load(); }
    size_t active_threads() const { return active_workers.load(); }
    bool is_paused() const { return paused.load(); }
    bool is_stopped() const { return stop.load(); }
    
    // Clear pending tasks (but don't stop running ones)
    void clear_pending() {
        std::unique_lock<std::mutex> lock(queue_mutex);
        std::queue<std::function<void()>> empty;
        std::swap(tasks, empty);
        tasks_pending = 0;
    }
};

// Global thread pool for script execution (singleton pattern)
class ScriptExecutor {
private:
    static std::unique_ptr<ThreadPool> pool;
    static std::mutex init_mutex;
    
public:
    // Initialize with specified thread count
    static void initialize(size_t num_threads = 0) {
        std::lock_guard<std::mutex> lock(init_mutex);
        if (!pool) {
            pool = std::make_unique<ThreadPool>(num_threads);
        }
    }
    
    // Get the thread pool (lazy initialization with auto-detect)
    static ThreadPool& get() {
        std::lock_guard<std::mutex> lock(init_mutex);
        if (!pool) {
            pool = std::make_unique<ThreadPool>(0);  // Auto-detect
        }
        return *pool;
    }
    
    // Reconfigure thread count
    static void set_thread_count(size_t num_threads) {
        get().resize(num_threads);
    }
    
    // Get current thread count
    static size_t get_thread_count() {
        return get().size();
    }
    
    // Shutdown
    static void shutdown() {
        std::lock_guard<std::mutex> lock(init_mutex);
        if (pool) {
            pool->shutdown();
            pool.reset();
        }
    }
};

} // namespace dsl
} // namespace digistar
#include "thread_pool.h"

namespace digistar {
namespace dsl {

// Static member definitions
std::unique_ptr<ThreadPool> ScriptExecutor::pool = nullptr;
std::mutex ScriptExecutor::init_mutex;

} // namespace dsl
} // namespace digistar
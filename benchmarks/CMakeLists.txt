# Benchmarks CMakeLists.txt

add_executable(benchmark_backends benchmark_backends.cpp)
target_link_libraries(benchmark_backends 
    digistar_backend
    digistar_core
    ${FFTW3_LIBRARIES}
    Threads::Threads
)
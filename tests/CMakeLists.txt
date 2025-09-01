# Tests CMakeLists.txt

# Unit tests
add_executable(test_pools unit/test_pools.cpp)
target_link_libraries(test_pools digistar_core Threads::Threads)

add_executable(test_dsl test_dsl.cpp)
target_link_libraries(test_dsl digistar_dsl digistar_core Threads::Threads)

add_executable(test_command_queue test_command_queue.cpp)
target_link_libraries(test_command_queue digistar_dsl digistar_core Threads::Threads)

add_executable(test_algorithms unit/test_algorithms.cpp)
target_link_libraries(test_algorithms digistar_backend digistar_core Threads::Threads)

# Integration tests
add_executable(test_convergence integration/test_convergence.cpp)
target_link_libraries(test_convergence digistar_backend digistar_core Threads::Threads)

# Validation tests
add_executable(test_accuracy validation/test_accuracy.cpp)
target_link_libraries(test_accuracy digistar_backend digistar_core Threads::Threads)

add_executable(test_solar_system validation/test_solar_system.cpp)
target_link_libraries(test_solar_system 
    digistar_simulation
    digistar_backend
    digistar_core
    ${FFTW3_LIBRARIES}
    Threads::Threads
)

# Add tests to CTest
add_test(NAME pools COMMAND test_pools)
add_test(NAME dsl COMMAND test_dsl)
add_test(NAME command_queue COMMAND test_command_queue)
add_test(NAME algorithms COMMAND test_algorithms)
add_test(NAME convergence COMMAND test_convergence)
add_test(NAME accuracy COMMAND test_accuracy)
add_test(NAME solar_system COMMAND test_solar_system)

# Custom target to run all tests
add_custom_target(run_tests
    COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure
    DEPENDS test_pools test_dsl test_command_queue 
            test_algorithms test_convergence test_accuracy
            test_solar_system
)
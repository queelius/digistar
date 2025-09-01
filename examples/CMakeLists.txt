# Examples CMakeLists.txt

add_executable(solar_system solar_system.cpp)
target_link_libraries(solar_system 
    digistar_simulation
    digistar_backend
    digistar_rendering
    digistar_core
    ${FFTW3_LIBRARIES}
    Threads::Threads
)

add_executable(solar_system_simple solar_system_simple.cpp)
target_link_libraries(solar_system_simple 
    digistar_simulation
    digistar_backend
    digistar_rendering
    digistar_core
    ${FFTW3_LIBRARIES}
    Threads::Threads
)

add_executable(million_particles million_particles.cpp)
target_link_libraries(million_particles 
    digistar_simulation
    digistar_backend
    digistar_rendering
    digistar_core
    ${FFTW3_LIBRARIES}
    Threads::Threads
)
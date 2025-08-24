# DigiStar Makefile
CXX = g++
NVCC = nvcc
CXXFLAGS = -std=c++17 -O3 -march=native -fopenmp -Wall
NVCCFLAGS = -std=c++17 -O3 -arch=sm_60
LDFLAGS = -lm -fopenmp -lfftw3f -pthread

# Directories
SRC_DIR = src
BUILD_DIR = build
BIN_DIR = $(BUILD_DIR)/bin
OBJ_DIR = $(BUILD_DIR)/obj
TEST_DIR = tests
BENCH_DIR = benchmarks
EXAMPLE_DIR = examples
TOOL_DIR = tools

# Create directories
$(shell mkdir -p $(BIN_DIR) $(OBJ_DIR))

# Main simulation objects
SIMULATION_OBJS = $(OBJ_DIR)/main.o \
                  $(OBJ_DIR)/simulation.o \
                  $(OBJ_DIR)/cpu_backend_reference.o \
                  $(OBJ_DIR)/cpu_backend_openmp.o \
                  $(OBJ_DIR)/ascii_renderer.o

# DSL and Command Queue objects
DSL_OBJS = $(OBJ_DIR)/sexpr.o \
           $(OBJ_DIR)/evaluator.o \
           $(OBJ_DIR)/command.o \
           $(OBJ_DIR)/core_simulation.o

# Backend objects (old - for compatibility)
BACKEND_OBJS = $(OBJ_DIR)/SimpleBackend.o \
               $(OBJ_DIR)/BackendFactory.o \
               $(OBJ_DIR)/SSE2Backend.o \
               $(OBJ_DIR)/AVX2Backend.o

# Default target
all: digistar backends tests benchmarks examples tools

# Main simulation executable
digistar: $(BIN_DIR)/digistar

$(BIN_DIR)/digistar: $(SIMULATION_OBJS)
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDFLAGS)

# Simulation objects compilation
$(OBJ_DIR)/main.o: $(SRC_DIR)/main.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(OBJ_DIR)/simulation.o: $(SRC_DIR)/simulation/simulation.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(OBJ_DIR)/cpu_backend_reference.o: $(SRC_DIR)/backend/cpu_backend_reference.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(OBJ_DIR)/cpu_backend_openmp.o: $(SRC_DIR)/backend/cpu_backend_openmp.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(OBJ_DIR)/ascii_renderer.o: $(SRC_DIR)/visualization/ascii_renderer.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Backends (old - for compatibility)
backends: $(BACKEND_OBJS)

$(OBJ_DIR)/SimpleBackend.o: $(SRC_DIR)/backend/SimpleBackend.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(OBJ_DIR)/BackendFactory.o: $(SRC_DIR)/backend/BackendFactory.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(OBJ_DIR)/SSE2Backend.o: $(SRC_DIR)/backend/SSE2Backend.cpp
	$(CXX) $(CXXFLAGS) -msse2 -c $< -o $@

$(OBJ_DIR)/AVX2Backend.o: $(SRC_DIR)/backend/AVX2Backend.cpp
	$(CXX) $(CXXFLAGS) -mavx2 -c $< -o $@

# DSL objects compilation
$(OBJ_DIR)/sexpr.o: $(SRC_DIR)/dsl/sexpr.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(OBJ_DIR)/evaluator.o: $(SRC_DIR)/dsl/evaluator.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(OBJ_DIR)/command.o: $(SRC_DIR)/dsl/command.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(OBJ_DIR)/core_simulation.o: $(SRC_DIR)/core/simulation.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Tests
tests: $(BIN_DIR)/test_algorithms $(BIN_DIR)/test_convergence $(BIN_DIR)/test_accuracy $(BIN_DIR)/test_dsl $(BIN_DIR)/test_command_queue

$(BIN_DIR)/test_algorithms: $(TEST_DIR)/unit/test_algorithms.cpp $(BACKEND_OBJS)
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDFLAGS)

$(BIN_DIR)/test_convergence: $(TEST_DIR)/integration/test_convergence.cpp $(BACKEND_OBJS)
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDFLAGS)

$(BIN_DIR)/test_accuracy: $(TEST_DIR)/validation/test_accuracy.cpp $(BACKEND_OBJS)
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDFLAGS)

$(BIN_DIR)/test_dsl: $(TEST_DIR)/test_dsl.cpp $(DSL_OBJS)
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDFLAGS) -lgtest -lgtest_main

$(BIN_DIR)/test_command_queue: $(TEST_DIR)/test_command_queue.cpp $(DSL_OBJS)
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDFLAGS) -lgtest -lgtest_main

# Benchmarks
benchmarks: $(BIN_DIR)/benchmark_backends $(BIN_DIR)/benchmark_million

$(BIN_DIR)/benchmark_backends: $(BENCH_DIR)/benchmark_backends.cpp $(BACKEND_OBJS)
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDFLAGS)

$(BIN_DIR)/benchmark_million: $(BENCH_DIR)/benchmark_million.cpp $(BACKEND_OBJS)
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDFLAGS)

# Examples
examples: $(BIN_DIR)/solar_system $(BIN_DIR)/million_particles

$(BIN_DIR)/solar_system: $(EXAMPLE_DIR)/solar_system.cpp $(BACKEND_OBJS)
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDFLAGS)

$(BIN_DIR)/million_particles: $(EXAMPLE_DIR)/million_particles.cpp $(BACKEND_OBJS)
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDFLAGS)

# Tools
tools: $(BIN_DIR)/backend_comparison

$(BIN_DIR)/backend_comparison: $(TOOL_DIR)/backend_comparison.cpp $(BACKEND_OBJS)
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDFLAGS)

# CUDA backend (optional)
cuda: $(OBJ_DIR)/CUDABackend.o

$(OBJ_DIR)/CUDABackend.o: $(SRC_DIR)/backend/CUDABackend.cu
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

# Run targets for the new simulation
run: $(BIN_DIR)/digistar
	$(BIN_DIR)/digistar --scenario default

run-solar: $(BIN_DIR)/digistar
	$(BIN_DIR)/digistar --scenario solar_system

run-galaxy: $(BIN_DIR)/digistar
	$(BIN_DIR)/digistar --scenario galaxy_collision

run-springs: $(BIN_DIR)/digistar
	$(BIN_DIR)/digistar --scenario spring_network

run-composite: $(BIN_DIR)/digistar
	$(BIN_DIR)/digistar --scenario composite_test

run-benchmark: $(BIN_DIR)/digistar
	$(BIN_DIR)/digistar --scenario stress_test --no-render --particles 50000

# Clean
clean:
	rm -rf $(BUILD_DIR)/*
	rm -f test_* benchmark_* interactive_* backend_compare solar_system

# Phony targets
.PHONY: all digistar backends tests benchmarks examples tools cuda clean run run-solar run-galaxy run-springs run-composite run-benchmark

# Help
help:
	@echo "DigiStar Build System"
	@echo "===================="
	@echo "make all        - Build everything"
	@echo "make backends   - Build backend libraries"
	@echo "make tests      - Build test suite"
	@echo "make benchmarks - Build benchmarks"
	@echo "make examples   - Build example applications"
	@echo "make tools      - Build utility tools"
	@echo "make cuda       - Build CUDA backend (requires NVCC)"
	@echo "make clean      - Clean build artifacts"
	@echo "make help       - Show this help"
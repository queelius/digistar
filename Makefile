# DigiStar - Clean Build System
# Optimized physics simulation achieving 1M particles at 8+ FPS

CXX = g++
CXXFLAGS = -std=c++17 -O3 -march=native -fopenmp -Wall -I.
LDFLAGS = -lSDL2 -lfftw3f -lm -lgomp

# Source files
SRCS = src/main.cpp \
       src/physics/pm_solver.cpp \
       src/physics/collision_backend.cpp

# Object files
OBJS = $(SRCS:.cpp=.o)

# Target executable
TARGET = digistar

# Default target
all: $(TARGET)

$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

# Pattern rule for object files
%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c -o $@ $<

# Run the simulation
run: $(TARGET)
	./$(TARGET)

# Run in headless benchmark mode
benchmark: $(TARGET)
	./$(TARGET) benchmark 1000000

# Clean build artifacts
clean:
	rm -f $(TARGET) $(OBJS)
	rm -f src/*.o src/physics/*.o

# Install (optional)
install: $(TARGET)
	install -m 755 $(TARGET) /usr/local/bin/

.PHONY: all clean run benchmark install
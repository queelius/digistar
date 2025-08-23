# DigiStar Experiments Makefile
# Automatically builds and optionally runs experiments

CXX = g++
CXXFLAGS = -std=c++17 -O3 -fopenmp -Wall -Wextra
LDFLAGS = -lm
BUILD_DIR = build

# Colors for output
RED = \033[0;31m
GREEN = \033[0;32m
YELLOW = \033[1;33m
NC = \033[0m # No Color

# Find all .cpp files recursively
CPP_FILES := $(shell find . -name "*.cpp" -not -path "./build/*" 2>/dev/null)
# Generate executable names (strip .cpp and directory path)
EXECUTABLES := $(patsubst %.cpp,$(BUILD_DIR)/%,$(notdir $(CPP_FILES)))

# Default target: show help
help:
	@echo "DigiStar Experiments Build System"
	@echo "================================="
	@echo ""
	@echo "Usage:"
	@echo "  make all          - Build all experiments"
	@echo "  make clean        - Remove all built files"
	@echo "  make list         - List all available experiments"
	@echo "  make run-all      - Build and run all experiments"
	@echo ""
	@echo "Build & run specific experiment:"
	@echo "  make sparse_grid  - Build and run sparse_grid experiment"
	@echo "  make [name]       - Build and run any experiment by name"
	@echo ""
	@echo "Build only (no run):"
	@echo "  make build/[name] - Build specific experiment without running"
	@echo ""
	@echo "Examples:"
	@echo "  make integration_methods"
	@echo "  make composite_collision"
	@echo "  make black_holes"

# Create build directory
$(BUILD_DIR):
	@mkdir -p $(BUILD_DIR)

# Pattern rule to build any .cpp file
$(BUILD_DIR)/%: */%.cpp | $(BUILD_DIR)
	@echo "$(GREEN)Building $*...$(NC)"
	@if [ "$*" = "million_particle_system" ]; then \
		$(CXX) $(CXXFLAGS) $< -o $@ $(LDFLAGS) -lfftw3f -lfftw3f_threads; \
	elif [ "$*" = "solar_system_fftw" ]; then \
		$(CXX) $(CXXFLAGS) $< -o $@ $(LDFLAGS) -lfftw3f -lfftw3f_threads; \
	else \
		$(CXX) $(CXXFLAGS) $< -o $@ $(LDFLAGS); \
	fi
	@echo "$(GREEN)✓ Built: $@$(NC)"

# Build all experiments
all: $(EXECUTABLES)
	@echo "$(GREEN)All experiments built successfully!$(NC)"

# Clean build directory
clean:
	@echo "$(YELLOW)Cleaning build directory...$(NC)"
	@rm -rf $(BUILD_DIR)
	@echo "$(GREEN)✓ Clean complete$(NC)"

# List all available experiments
list:
	@echo "$(YELLOW)Available experiments:$(NC)"
	@for file in $(CPP_FILES); do \
		name=$$(basename $$file .cpp); \
		dir=$$(dirname $$file | sed 's|^\./||'); \
		echo "  $(GREEN)$$name$(NC) ($$dir/)"; \
	done

# Run all experiments
run-all: all
	@echo "$(YELLOW)Running all experiments...$(NC)"
	@echo "=========================="
	@for exe in $(EXECUTABLES); do \
		if [ -f $$exe ]; then \
			echo ""; \
			echo "$(GREEN)Running $$(basename $$exe)...$(NC)"; \
			echo "-------------------"; \
			$$exe || echo "$(RED)Failed: $$(basename $$exe)$(NC)"; \
			echo ""; \
		fi; \
	done
	@echo "$(GREEN)All experiments completed!$(NC)"

# Generic target to build and run any experiment by name
%: $(BUILD_DIR)/%
	@echo "$(YELLOW)Running $*...$(NC)"
	@echo "===================="
	@$(BUILD_DIR)/$* || (echo "$(RED)Experiment failed!$(NC)" && exit 1)
	@echo ""
	@echo "$(GREEN)✓ $* completed$(NC)"

# Special targets for common experiments (shortcuts)
spatial: sparse_grid spatial_index

collision: composite_collision_v2

integration: integration_methods

# Verbose build (show commands)
verbose: CXXFLAGS += -v
verbose: all

# Debug build
debug: CXXFLAGS = -std=c++17 -g -O0 -fopenmp -Wall -Wextra -DDEBUG
debug: all

# Benchmark mode (maximum optimization)
benchmark: CXXFLAGS = -std=c++17 -O3 -march=native -fopenmp -DNDEBUG
benchmark: all

# Help if no target specified
.DEFAULT_GOAL := help

# Prevent make from deleting intermediate files
.PRECIOUS: $(BUILD_DIR)/%

.PHONY: all clean list run-all help spatial collision integration verbose debug benchmark
#!/bin/bash

# Build script for gravity baseline

echo "Building gravity baseline..."

# Create build directory
mkdir -p build
cd build

# Configure with CMake
cmake ../src -DCMAKE_BUILD_TYPE=Release

# Build
make -j$(nproc)

echo ""
echo "Build complete! Run with:"
echo "  ./build/gravity_baseline"
echo ""
echo "To profile with Nsight Compute:"
echo "  ncu --set full ./build/gravity_baseline"
echo ""
echo "To profile with nvprof:"
echo "  nvprof ./build/gravity_baseline"
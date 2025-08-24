#!/bin/bash

# DigiStar Repository Cleanup Script
# This script reorganizes the repository structure

echo "=== DigiStar Repository Cleanup ==="
echo "This will reorganize files into proper directories."
echo "Press Ctrl+C to abort, or Enter to continue..."
read

# Step 1: Move test files
echo "Moving test files..."

# Unit tests
mv test_quadtree.cpp tests/unit/ 2>/dev/null
mv test_fft_simple.cpp tests/unit/test_fft.cpp 2>/dev/null
mv test_algorithms.cpp tests/unit/ 2>/dev/null

# Integration tests  
mv test_backend_v4.cpp tests/integration/test_backends.cpp 2>/dev/null
mv test_convergence.cpp tests/integration/ 2>/dev/null
mv test_barnes_hut_backend.cpp tests/integration/ 2>/dev/null
mv test_simple_bh.cpp tests/integration/ 2>/dev/null
mv test_bh_robust.cpp tests/integration/ 2>/dev/null
mv test_barnes_hut_fixed.cpp tests/integration/ 2>/dev/null

# Validation tests
mv test_accuracy.cpp tests/validation/ 2>/dev/null
mv test_accuracy_multi.cpp tests/validation/ 2>/dev/null
mv test_earth_moon.cpp tests/validation/ 2>/dev/null
mv test_orbit_simple.cpp tests/validation/ 2>/dev/null
mv test_solar_system.cpp tests/validation/ 2>/dev/null
mv test_pm.cpp tests/validation/ 2>/dev/null
mv test_pm_custom.cpp tests/validation/ 2>/dev/null

# Step 2: Move benchmarks
echo "Moving benchmarks..."
mv benchmark_all.cpp benchmarks/benchmark_backends.cpp 2>/dev/null
mv test_million.cpp benchmarks/benchmark_million.cpp 2>/dev/null
mv test_million_viz.cpp benchmarks/ 2>/dev/null

# Step 3: Move examples
echo "Moving examples..."
mv interactive_solar_system_accurate.cpp examples/solar_system.cpp 2>/dev/null
mv interactive_solar_system.cpp examples/solar_system_simple.cpp 2>/dev/null
mv interactive_2m.cpp examples/million_particles.cpp 2>/dev/null

# Step 4: Move tools
echo "Moving tools..."
mv backend_comparison.cpp tools/ 2>/dev/null

# Step 5: Clean up binaries
echo "Cleaning up binaries..."
rm -f test_* 2>/dev/null
rm -f benchmark_* 2>/dev/null
rm -f interactive_* 2>/dev/null
rm -f backend_compare 2>/dev/null
rm -f solar_system 2>/dev/null
rm -f *.o 2>/dev/null

# Step 6: Create .gitignore if needed
if [ ! -f .gitignore ]; then
    echo "Creating .gitignore..."
    cat > .gitignore << EOF
# Build outputs
build/
*.o
*.a
*.so
*.exe

# Executables
/bin/
/test_*
/benchmark_*

# IDE files
.vscode/
*.swp
*.swo
*~

# System files
.DS_Store
Thumbs.db
EOF
fi

echo ""
echo "=== Cleanup Complete ==="
echo ""
echo "Next steps:"
echo "1. Consolidate backend versions (keep v4 as main)"
echo "2. Update #include paths in all files"
echo "3. Create CMakeLists.txt or Makefile"
echo "4. Update documentation"
echo ""
echo "Files are now organized as:"
echo "  tests/       - All test files"
echo "  benchmarks/  - Performance benchmarks"
echo "  examples/    - Example applications"
echo "  tools/       - Utility tools"
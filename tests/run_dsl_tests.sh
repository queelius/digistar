#!/bin/bash

echo "Building DSL tests..."

# Compile the test
g++ -std=c++17 -O2 \
    test_dsl.cpp \
    ../src/dsl/sexpr.cpp \
    ../src/dsl/evaluator.cpp \
    ../src/physics/pools.cpp \
    -I../src \
    -o test_dsl \
    -Wall -Wextra

if [ $? -eq 0 ]; then
    echo "Build successful. Running tests..."
    echo "================================="
    ./test_dsl
    exit_code=$?
    
    # Clean up
    rm -f test_dsl
    
    exit $exit_code
else
    echo "Build failed!"
    exit 1
fi
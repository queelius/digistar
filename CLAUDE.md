# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Digital Star (digistar) is an ambitious sandbox space simulation game that simulates "big atoms" - fundamental units with complex physics properties that can scale from spaceships to multi-star systems. The project is currently in early design phase with extensive documentation but limited implementation.

## Repository Structure

### Active Development Areas
- `docs/` - Primary design documentation (THIS IS THE MAIN FOCUS)
  - Contains extensive design specs for physics, networking, AI, and game mechanics
  - Design is experimental and ambitious - expect iteration
  
### Legacy Code (Reference Only)
- `old-code/` - Historical implementations for reference
  - `2dsim/` - 2D simulation with CUDA kernels
  - `lsss/` - Large-scale space simulation 
  - `space-sandbox/` - Another space simulation variant
  - `oldest_sim/` - Original C++ physics implementation

## Key Design Concepts

### Big Atoms
The fundamental simulation unit with properties:
- Mass, radius, position, velocity
- Charge, interaction vector
- Rotational state, internal temperature
- Magnetic moment

### Core Systems (Planned)
- **Physics**: GPU-accelerated force calculations, octree spatial indexing
- **Networking**: RESTful API, binary UDP, shared memory IPC (see below)
- **AI**: Python integration, subsumption agent architecture
- **Rendering**: Separate from physics engine (GPU dedicated to physics)

### High-Performance IPC (posix_shm)
The `posix_shm` repository provides zero-overhead shared memory data structures for IPC:
- **Lock-free data structures**: Arrays, queues, ring buffers, object pools
- **Zero-copy performance**: Direct memory access between processes (2-3ns per operation)
- **Dynamic discovery**: Named structures with metadata management via `shm_table`
- **Optimized for simulation**: 170x faster than serialization, 400x lower latency than pipes
- Designed for high-frequency communication between physics engine, rendering, and AI bots

## Development Commands

### Documentation Build
```bash
cd docs
make                    # Concatenate all markdown files into design_document.md
make design_document.pdf # Generate PDF version (requires pandoc)
make clean             # Remove generated files
```

### Legacy CUDA Projects (in old-code/*/):
```bash
mkdir build && cd build
cmake ..
make
```

Note: These are reference implementations - no active development.

## Working with Design Documents

The design documentation in `docs/` follows this structure:
1. `index.md` - Table of contents for all design docs
2. Individual topic files (e.g., `big-atom-fundamentals.md`, `tensor-springs.md`)
3. `design_document.md` - Auto-generated concatenation (via Makefile)

When modifying design docs:
- Edit individual topic files, not `design_document.md`
- Run `make` in docs/ to regenerate the combined document
- Focus on clarity and feasibility - the design is ambitious and needs refinement

## Important Notes

- **Design Phase**: The project is primarily in design/documentation phase
- **No Active Code**: There's no current implementation - focus on design docs
- **GPU Focus**: All physics planned for CUDA GPU acceleration
- **Scale Goal**: Target 10+ million big atoms simultaneously
- **Experimental Physics**: Novel physics models (black holes, warp channels, etc.)

## Python Scripts in Docs

The docs folder contains visualization scripts:
- `energy-fission-prob.py` - Visualizes energy fission probability
- `tensor-stiffness-spring.py` - Visualizes tensor spring stiffness
- `makemake.py` - Helper script for documentation

## Design Philosophy

The project aims to create emergent complexity from simple "big atom" interactions. Key principles:
1. Fundamental properties lead to complex behaviors
2. GPU optimization is critical for scale
3. Players interact with physics, not pre-scripted events
4. Support for both human players and AI agents
5. Distributed server architecture for massive scale

## Key Design Concepts from Python Prototype

The `space-sandbox-sim` Python implementation explores several important concepts:

### Emergent Composite Bodies
- **Automatic Spring Formation**: Springs form automatically between bodies based on distance and relative velocity thresholds
- **Graph Connectivity**: Uses NetworkX to identify connected components in spring networks, forming composite bodies
- **Convex Hulls**: Used primarily for fast intersection testing rather than visualization
  - Provides bounding volumes for collision detection
  - Calculates hull area, centroid, and other geometric properties

### Spring Network Mechanics
- Springs have configurable properties: stiffness, damping, equilibrium distance
- Break conditions: distance threshold and force threshold
- Virtual spring fields automatically connect nearby bodies with low relative velocities
- Spring networks are reconstructed each timestep to identify transient composite bodies

### Composite Body Properties
- Emergent properties calculated from constituent bodies:
  - Center of mass and velocity
  - Angular momentum and velocity
  - Internal vs. rotational energy
  - Moment of inertia
- Forces can be applied to composite as a whole (distributed by mass)
- Torques distributed as linear forces to constituent bodies

These concepts should inform the C++ implementation while maintaining focus on GPU optimization and scale.
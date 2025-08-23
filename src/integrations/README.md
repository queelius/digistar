# Digital Star Integrations

This folder contains various ways to interact with the Digital Star physics simulation engine.

## Architecture

The core simulation runs as a headless physics engine. Integrations connect via:
- **Shared Memory** (fastest, local only) - Zero-copy data transfer
- **WebSockets** (network, real-time) - For remote viewers and agents
- **gRPC** (network, structured) - For robust client-server communication
- **MCP** (Model Context Protocol) - For LLM integration

## Integrations

### 1. Viewers (`/viewers`)
- **ASCII Viewer** - Terminal-based visualization for debugging
- **Stats Dashboard** - Performance metrics and simulation statistics
- **OpenGL Viewer** - 3D visualization client (separate process)
- **Web Viewer** - Browser-based visualization using WebGL

### 2. AI Agents (`/agents`)
- **Reactive Agents** - Simple rule-based behaviors (subsumption architecture)
- **RL Agents** - Reinforcement learning agents
  - DQN (Deep Q-Network)
  - PPO (Proximal Policy Optimization)
  - A3C (Asynchronous Advantage Actor-Critic)
- **LLM Agents** - Language model-driven agents with cognitive cores

### 3. MCP Server (`/mcp`)
Model Context Protocol integration enabling:
- **Dungeon Master Mode** - LLM narrates the space adventure in real-time
- **God Mode** - LLM can create/modify simulation entities
- **Agent Controller** - LLM controls agent behaviors through natural language
- **World Builder** - Procedural generation via LLM prompts

### 4. Shared Memory (`/shm`)
- **POSIX Shared Memory** - Zero-copy data sharing between processes
- **Lock-free Ring Buffers** - For streaming particle data
- **Memory Pools** - Pre-allocated memory for performance

### 5. Network (`/network`)
- **WebSocket Server** - Real-time bidirectional communication
- **REST API** - HTTP endpoints for control and queries
- **gRPC Service** - High-performance RPC with protobuf

## Data Flow

```
┌─────────────────────────────────────────────────────────┐
│                  Digital Star Core Engine                │
│                   (Physics Simulation)                   │
└────────────┬───────────────────────────┬────────────────┘
             │                           │
        Shared Memory              Network (WS/gRPC)
             │                           │
    ┌────────┴────────┐         ┌───────┴────────┐
    │  Local Clients  │         │ Remote Clients  │
    │  - Viewers      │         │  - Web Viewer   │
    │  - AI Training  │         │  - MCP Server   │
    │  - Analytics    │         │  - Multiplayer  │
    └─────────────────┘         └────────────────┘
```

## Integration Interface

All integrations interact through a common interface:

```cpp
class ISimulationInterface {
    // Query
    virtual SimulationState getState() = 0;
    virtual std::vector<Particle> getParticles(BoundingBox region) = 0;
    virtual std::vector<Entity> getEntities(EntityFilter filter) = 0;
    
    // Control
    virtual void spawnEntity(EntityDef def) = 0;
    virtual void applyForce(EntityID id, Vec3 force) = 0;
    virtual void setParameter(std::string param, float value) = 0;
    
    // Events
    virtual void subscribe(EventType type, EventCallback cb) = 0;
    virtual void processCommands(CommandQueue& queue) = 0;
};
```

## Performance Considerations

- **Shared Memory**: ~100ns latency, 100GB/s throughput
- **WebSocket**: ~1ms latency, 100MB/s throughput  
- **gRPC**: ~0.5ms latency, 1GB/s throughput
- **MCP**: ~100ms latency (LLM inference time)

## Future Integrations

- **VR/AR Support** - Immersive space exploration
- **Distributed Simulation** - Multi-node computation
- **Blockchain** - Persistent universe state
- **Scientific Visualization** - ParaView, VTK integration
- **Game Engines** - Unity/Unreal Engine bridges
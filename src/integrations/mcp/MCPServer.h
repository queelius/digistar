#pragma once

#include <string>
#include <memory>
#include <functional>
#include <vector>
#include <map>
#include <json/json.h>  // Will use jsoncpp for JSON parsing
#include "../../backend/ISimulationBackend.h"

// MCP (Model Context Protocol) Server
// Allows LLMs to control the simulation as a "space dungeon master"
// Provides high-level commands and queries that LLMs can use to orchestrate gameplay
class MCPServer {
public:
    // Tool definitions for MCP protocol
    struct Tool {
        std::string name;
        std::string description;
        Json::Value parameters;  // JSON schema for parameters
        std::function<Json::Value(const Json::Value&)> handler;
    };
    
    // Events that LLMs can subscribe to
    struct Event {
        std::string type;
        Json::Value data;
        float timestamp;
    };
    
private:
    std::shared_ptr<ISimulationBackend> backend;
    std::vector<Tool> tools;
    std::vector<Event> event_queue;
    bool running;
    
    // Game state for LLM context
    struct GameState {
        float simulation_time;
        size_t step_count;
        size_t particle_count;
        std::map<std::string, float> global_properties;  // Energy, momentum, etc.
        std::vector<std::string> active_scenarios;
        std::map<size_t, std::string> named_entities;  // Important particles/groups
    } game_state;
    
    // Initialize default tools available to LLMs
    void initializeTools() {
        // Create particle cluster
        tools.push_back({
            "create_cluster",
            "Create a cluster of particles at specified location",
            createParameterSchema({
                {"x", "number", "X coordinate of cluster center"},
                {"y", "number", "Y coordinate of cluster center"},
                {"count", "integer", "Number of particles"},
                {"mass_range", "array", "Min and max mass [min, max]"},
                {"velocity", "array", "Initial velocity [vx, vy]"},
                {"pattern", "string", "Formation pattern: sphere, ring, spiral, random"}
            }),
            [this](const Json::Value& params) { return handleCreateCluster(params); }
        });
        
        // Apply force to region
        tools.push_back({
            "apply_force",
            "Apply a force to all particles in a region",
            createParameterSchema({
                {"x", "number", "X coordinate of force center"},
                {"y", "number", "Y coordinate of force center"},
                {"radius", "number", "Radius of effect"},
                {"fx", "number", "X component of force"},
                {"fy", "number", "Y component of force"},
                {"duration", "number", "Duration in simulation time"}
            }),
            [this](const Json::Value& params) { return handleApplyForce(params); }
        });
        
        // Query particles
        tools.push_back({
            "query_region",
            "Get information about particles in a region",
            createParameterSchema({
                {"x", "number", "X coordinate of query center"},
                {"y", "number", "Y coordinate of query center"},
                {"radius", "number", "Query radius"},
                {"properties", "array", "Properties to return: position, velocity, mass, etc."}
            }),
            [this](const Json::Value& params) { return handleQueryRegion(params); }
        });
        
        // Create black hole
        tools.push_back({
            "create_black_hole",
            "Create a massive gravitational object",
            createParameterSchema({
                {"x", "number", "X coordinate"},
                {"y", "number", "Y coordinate"},
                {"mass", "number", "Mass of black hole"},
                {"name", "string", "Optional name for tracking"}
            }),
            [this](const Json::Value& params) { return handleCreateBlackHole(params); }
        });
        
        // Set scenario
        tools.push_back({
            "set_scenario",
            "Load a predefined scenario or game mode",
            createParameterSchema({
                {"scenario", "string", "Scenario name: galaxy_collision, asteroid_field, planet_formation, battle_royale"},
                {"parameters", "object", "Scenario-specific parameters"}
            }),
            [this](const Json::Value& params) { return handleSetScenario(params); }
        });
        
        // Get simulation stats
        tools.push_back({
            "get_stats",
            "Get current simulation statistics",
            createParameterSchema({}),
            [this](const Json::Value& params) { return handleGetStats(params); }
        });
        
        // Advance simulation
        tools.push_back({
            "advance_time",
            "Advance simulation by specified time",
            createParameterSchema({
                {"time", "number", "Time to advance in simulation units"},
                {"max_steps", "integer", "Maximum steps to take"}
            }),
            [this](const Json::Value& params) { return handleAdvanceTime(params); }
        });
        
        // Create named entity
        tools.push_back({
            "create_entity",
            "Create a named entity for story purposes",
            createParameterSchema({
                {"name", "string", "Entity name for narrative"},
                {"type", "string", "Entity type: ship, planet, asteroid, station"},
                {"x", "number", "X coordinate"},
                {"y", "number", "Y coordinate"},
                {"properties", "object", "Type-specific properties"}
            }),
            [this](const Json::Value& params) { return handleCreateEntity(params); }
        });
    }
    
    // Helper to create JSON schema for parameters
    Json::Value createParameterSchema(const std::map<std::string, std::vector<std::string>>& params) {
        Json::Value schema;
        schema["type"] = "object";
        schema["properties"] = Json::Value(Json::objectValue);
        
        for (const auto& [name, info] : params) {
            Json::Value prop;
            prop["type"] = info[0];  // type
            prop["description"] = info[1];  // description
            if (info.size() > 2) {
                prop["enum"] = Json::Value(Json::arrayValue);
                for (size_t i = 2; i < info.size(); i++) {
                    prop["enum"].append(info[i]);
                }
            }
            schema["properties"][name] = prop;
        }
        
        return schema;
    }
    
    // Tool handlers
    Json::Value handleCreateCluster(const Json::Value& params);
    Json::Value handleApplyForce(const Json::Value& params);
    Json::Value handleQueryRegion(const Json::Value& params);
    Json::Value handleCreateBlackHole(const Json::Value& params);
    Json::Value handleSetScenario(const Json::Value& params);
    Json::Value handleGetStats(const Json::Value& params);
    Json::Value handleAdvanceTime(const Json::Value& params);
    Json::Value handleCreateEntity(const Json::Value& params);
    
    // Event generation for narrative
    void generateEvent(const std::string& type, const Json::Value& data) {
        Event evt;
        evt.type = type;
        evt.data = data;
        evt.timestamp = game_state.simulation_time;
        event_queue.push_back(evt);
        
        // Keep queue size reasonable
        if (event_queue.size() > 100) {
            event_queue.erase(event_queue.begin());
        }
    }
    
public:
    MCPServer(std::shared_ptr<ISimulationBackend> backend) 
        : backend(backend), running(false) {
        initializeTools();
        game_state = {0};
    }
    
    // MCP protocol implementation
    Json::Value listTools() const {
        Json::Value result;
        result["tools"] = Json::Value(Json::arrayValue);
        
        for (const auto& tool : tools) {
            Json::Value t;
            t["name"] = tool.name;
            t["description"] = tool.description;
            t["inputSchema"] = tool.parameters;
            result["tools"].append(t);
        }
        
        return result;
    }
    
    // Execute a tool call from LLM
    Json::Value executeTool(const std::string& tool_name, const Json::Value& arguments) {
        for (const auto& tool : tools) {
            if (tool.name == tool_name) {
                try {
                    return tool.handler(arguments);
                } catch (const std::exception& e) {
                    Json::Value error;
                    error["error"] = true;
                    error["message"] = e.what();
                    return error;
                }
            }
        }
        
        Json::Value error;
        error["error"] = true;
        error["message"] = "Unknown tool: " + tool_name;
        return error;
    }
    
    // Get recent events for narrative context
    Json::Value getRecentEvents(size_t count = 10) const {
        Json::Value result;
        result["events"] = Json::Value(Json::arrayValue);
        
        size_t start = event_queue.size() > count ? event_queue.size() - count : 0;
        for (size_t i = start; i < event_queue.size(); i++) {
            Json::Value evt;
            evt["type"] = event_queue[i].type;
            evt["data"] = event_queue[i].data;
            evt["timestamp"] = event_queue[i].timestamp;
            result["events"].append(evt);
        }
        
        return result;
    }
    
    // Start/stop server
    void start();
    void stop() { running = false; }
    bool isRunning() const { return running; }
    
    // Update game state
    void update(float dt);
};

// Example scenarios that LLMs can orchestrate
namespace Scenarios {
    // Galaxy collision scenario
    class GalaxyCollision {
    public:
        static Json::Value create(const Json::Value& params);
    };
    
    // Asteroid mining scenario
    class AsteroidField {
    public:
        static Json::Value create(const Json::Value& params);
    };
    
    // Planet formation from dust
    class PlanetFormation {
    public:
        static Json::Value create(const Json::Value& params);
    };
    
    // Space battle with fleets
    class BattleRoyale {
    public:
        static Json::Value create(const Json::Value& params);
    };
}
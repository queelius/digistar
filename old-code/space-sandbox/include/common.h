#ifndef COMMON_H
#define COMMON_H

#include <cuda_runtime.h>
#include <optional>
#include <variant>

#define MAX_NAME_LENGTH 32

enum CommandType {
    ADD_BODY,
    UPDATE_BODY,
    REMOVE_BODY,
    PAUSE_SIMULATION,
    RESUME_SIMULATION,
    CHANGE_SIMULATION_PARAMS
};

struct Body {
    double3 position;
    double3 velocity;
    double3 force;
    double3 color;
    double mass;
    double charge;
    double radius;

    // name is unique for each body
    // we can use it to identify a body
    char name[MAX_NAME_LENGTH];

    // active is used to determine if the body is active
    // if it's not active, it won't be updated, and it may be overwritten
    // by a new body (fixed size array. we optimize for the case where
    // bodies are added or removed very frequently, and we don't want to
    // reallocate memory. it's also all suppose to reside in GPU memory,
    // which is also shared with the CPU, and a shared memory structure that
    // other processes can access)
    bool active;
};

struct BodyCommand {
    std::variant<const char*, int> name_or_index;
    std::optional<double3> position;
    std::optional<double3> velocity;
    std::optional<double3> color;
    std::optional<double> force;
    std::optional<double> mass;
    std::optional<double> charge;
    std::optional<double> radius;
};

struct ControlCommand {
    std::optional<double> timeStep;
    std::optional<double> gravitationalConstant;
    std::optional<double> coloumbsConstant;
    std::optional<bool> pause;
    std::optional<bool> resume;
};

struct Command {
    CommandType type;
    union {
        BodyCommand bodyCmd;
        ControlCommand controlCmd;
    };

    Command() {
        std::memset(this, 0, sizeof(Command));
    }
};

#endif // COMMON_H

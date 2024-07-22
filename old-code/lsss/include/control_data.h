#ifndef CONTROL_DATA_H
#define CONTROL_DATA_H

#include <optional>

enum Command {
    NONE,
    PAUSE,
    RESUME,
    UPDATE_PARAMETERS,
    ADD_BODY,
    REMOVE_BODY,
    UPDATE_BODY
};

struct BodyCommand {
    Command command;
    char name[32];
    std::optional<double3> position;
    std::optional<double3> velocity;
    std::optional<double> mass;
    std::optional<double> radius;
    std::optional<float3> color;
    int parentIndex;
};

struct ControlData {
    BodyCommand commandQueue[10]; // Simple fixed-size queue for demonstration
    int commandQueueSize;
};

#endif // CONTROL_DATA_H

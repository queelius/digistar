#ifndef COMMAND_QUEUE_H
#define COMMAND_QUEUE_H

#include "common.h"

struct CommandQueue
{
    CommandQueue(int capacity = 1000, const char * resource_name);
    ~CommandQueue();

    bool isEmpty() const;
    bool isFull() const;

    int pushAll(Command * commands, int count);
    std::pair<int, Command*> popAll();

    void clear();

    std::optional<Command> pop();
    bool push(Command command);

    Command * cmds;
    const char * resource_name;
    int capacity;
    int size;
};

#endif // COMMAND_QUEUE_H

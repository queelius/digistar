#include "src/events/event_system.h"
#include <iostream>

using namespace digistar;

int main() {
    std::cout << "Event size: " << sizeof(Event) << " bytes" << std::endl;
    std::cout << "Event alignment: " << alignof(Event) << " bytes" << std::endl;
    
    Event e;
    std::cout << "Offset of type: " << offsetof(Event, type) << std::endl;
    std::cout << "Offset of flags: " << offsetof(Event, flags) << std::endl;
    std::cout << "Offset of tick: " << offsetof(Event, tick) << std::endl;
    std::cout << "Offset of timestamp: " << offsetof(Event, timestamp) << std::endl;
    std::cout << "Offset of x: " << offsetof(Event, x) << std::endl;
    std::cout << "Offset of y: " << offsetof(Event, y) << std::endl;
    std::cout << "Offset of radius: " << offsetof(Event, radius) << std::endl;
    std::cout << "Offset of primary_id: " << offsetof(Event, primary_id) << std::endl;
    std::cout << "Offset of secondary_id: " << offsetof(Event, secondary_id) << std::endl;
    std::cout << "Offset of magnitude: " << offsetof(Event, magnitude) << std::endl;
    std::cout << "Offset of secondary_value: " << offsetof(Event, secondary_value) << std::endl;
    std::cout << "Offset of data: " << offsetof(Event, data) << std::endl;
    
    std::cout << "\nSize of data union: " << sizeof(e.data) << std::endl;
    
    return 0;
}
#ifndef BUS_H
#define BUS_H

#include <map>
#include "Object.h"
#include "Message.h"

const unsigned BROADCAST = ~0;

class Bus
{
public:
    bool sendTypes(const Message& msg, unsigned types = Types::OBJECT, Object* from = 0)
    {
        for (auto o = _oList.begin(); o != _oList.end(); o++)
        {
            if (o->second->isTypes(types))
                o->second->send(msg, from);
        }
    };

    bool send(const Message& msg, unsigned id = BROADCAST, Object* from = 0)
    {
        if (id == BROADCAST)
        {
            for (auto o = _oList.begin(); o != _oList.end(); o++)
                o->second->send(msg, from);
        }
        else // unicast
        {
            if (!_oList.count())
                return false;
            _oList[id]->send(msg, from);
        }
        return true;
    };

    unsigned getNumObjects() const
    {
        return _oList.size();
    };

    bool detachObject(unsigned id)
    {
        return _oList.erase(id);
    };

    void attachObject(Object* object)
    {
        _oList[object->getId()] = object;
    };

protected:
    std::map<unsigned, Object*> _oList;
};

#endif
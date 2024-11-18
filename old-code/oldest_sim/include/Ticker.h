#ifndef TICKER_H
#define TICKER_H

// make it based on simple timer, or stick with clock() stuff?
// or convert everything to clock() stuff?

#include <list>
#include <ctime>
#include <string>
#include "Object.h"
#include "Events.h"
#include "Message.h"

class Ticker: public Object
{
public:
    Ticker(double time, unsigned id, double ticksPerSec, unsigned maxTicks = 0);
    void update();
    void set(double ticksPerSec, unsigned maxTicks = 0);

protected:
    clock_t _period;
    clock_t _nextTick;
    unsigned _ticks;
    unsigned _maxTicks;
};

Ticker::Ticker(double time, unsigned id, double ticksPerSec, unsigned maxTicks): Object(time, id, Types::TICKER)
{
    set(ticksPerSec, maxTicks);
}

void Ticker::update()
{
    if (_maxTicks && _ticks == _maxTicks)
    {        
        expire();
        alert(Message(Events::EXPIRED));
    }
    else if (_nextTick <= clock())
    {
        ++_ticks;
        _nextTick += _period;
        alert(Message(Events::TICK));
    }
}

void Ticker::set(double ticksPerSec, unsigned maxTicks)
{
    if (ticksPerSec <= 0)
        throw "Error: ticksPerSec must be > 0"; 

    _ticks = 0;
    _maxTicks = maxTicks;
    _period = (clock_t)((double)CLOCKS_PER_SEC / ticksPerSec);
    _nextTick = clock() + _period;

    alert(Message(Events::STATE_CHANGED));
}

#endif
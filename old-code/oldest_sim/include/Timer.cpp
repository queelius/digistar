#include "Timer.h"

Timer::Timer()
{
    stop();
}

bool Timer::isPaused() const
{
    return _paused;
};

bool Timer::isStopped() const
{
    return !_started;
}

bool Timer::isStarted() const
{
    return _started;
}

double Timer::getElapsedPause() const
{
    if (_paused)
        return (clock() - _pauseStart) / (double)CLOCKS_PER_SEC;
    else
        return 0.0;
};

double Timer::getElapsed() const
{
    if (_started)
        return (clock() - _start) / (double)CLOCKS_PER_SEC;
    else
        return _elapsed;
}

bool Timer::pause()
{
    if (!_paused && _started)
    {
        _elapsed = getElapsed();
        _pauseStart = clock();
        _paused = true;            
        return true;
    }
    else
        return false;
}

bool Timer::start()
{
    if (_paused)
    {
        _start += (clock() - _pauseStart); // accumulate pause durations by adjusting start time
        _paused = false;
        return true;
    }
    else if (!_started)
    {
        _start = clock();
        _started = true;
        return true;
    }
    else
        return false;
}

bool Timer::stop()
{
    if (_started)
        return false;
    else
    {
        _started = false;
        _paused = false;
        _elapsed = 0;
        return true;
    }
}

double operator-(const Timer &lhs, Timer &rhs)
{
    return lhs.getElapsed() - rhs.getElapsed();
}

bool operator==(const Timer  &lhs, Timer &rhs)
{
    return lhs.getElapsed() == rhs.getElapsed();
}

bool operator<(const Timer  &lhs, Timer &rhs)
{
    return lhs.getElapsed() < rhs.getElapsed();
}

bool operator>(const Timer  &lhs, Timer &rhs)
{
    return lhs.getElapsed() > rhs.getElapsed();
}

bool operator<=(const Timer  &lhs, Timer &rhs)
{
    return lhs.getElapsed() < rhs.getElapsed() ||
           lhs.getElapsed() == rhs.getElapsed();
}

bool operator>=(const Timer  &lhs, Timer &rhs)
{
    return lhs.getElapsed() > rhs.getElapsed() ||
           lhs.getElapsed() == rhs.getElapsed();
}

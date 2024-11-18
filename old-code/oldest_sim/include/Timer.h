#ifndef TIMER_H
#define TIMER_H

#include <ctime>

class Timer {
public:
            Timer();

    bool    isPaused()          const;
    bool    isStopped()         const;
    bool    isStarted()         const;

    double  getElapsedPause()   const;
    double  getElapsed()        const;

    bool    pause();
    bool    start();
    bool    stop();

protected:
    clock_t _start;
    clock_t _pauseStart;

    double  _elapsed;

    bool    _started;
    bool    _paused;
};

double  operator-   (const Timer &lhs, Timer &rhs);
bool    operator<   (const Timer &lhs, Timer &rhs);
bool    operator>   (const Timer &lhs, Timer &rhs);
bool    operator<=  (const Timer &lhs, Timer &rhs);
bool    operator>=  (const Timer &lhs, Timer &rhs);
bool    operator==  (const Timer &lhs, Timer &rhs);

#endif
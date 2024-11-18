#include "simpleTime.h"

SimpleTime::SimpleTime(GLuint startTime, GLfloat ticksPerSec) {
    this->time = startTime;
    this->ticksPerSec = ticksPerSec;
}

GLuint SimpleTime::get() {
    return time;
}

GLfloat SimpleTime::getSeconds() {
    return time / ticksPerSec;
}

GLfloat SimpleTime::getMinutes() {
    return getSeconds() / 60.0;
}

GLfloat SimpleTime::getHours() {
    return getHours() / 60.0;
}

GLfloat SimpleTime::getTicksPerSec() {
    return ticksPerSec;
}

void SimpleTime::setTicksPerSec(GLfloat ticksPerSec) {
    this->ticksPerSec = ticksPerSec;
}

void SimpleTime::setTime(GLuint time) {
    this->time = time;
}

GLuint SimpleTime::update() {
    ++time;
}

void SimpleTime::reset() {
    time = 0;
}

GLint getTimeDiff(SimpleTime t_start, SimpleTime t_end) {
    if (t_start.ticksPerSec == t_end.ticksPerSec)
        return (GLint)(t_end.time - t_start.time);
    else
        return (GLint)(t_end.time * t_end.ticksPerSec -
                       t_start.time * t_start.ticksPerSec);
}

#ifndef SIMPLE_TIME_H
#define SIMPLE_TIME_H

#include "glut.h"

class SimpleTime {
    friend GLint getTimeDiff(SimpleTime t_start, SimpleTime t_end);

public:
    SimpleTime(GLuint startTime = 0, GLfloat ticksPerSec = 100);

    GLuint get();
    GLfloat getSeconds();
    GLfloat getMinutes();
    GLfloat getHours();

    GLfloat getTicksPerSec();

    void setTicksPerSec(GLfloat ticksPerSec);
    void setTime(GLuint time);

    GLuint update();
    void reset();

private:
    GLuint time;
    GLfloat ticksPerSec;
};

GLint getTimeDiff(SimpleTime t_start, SimpleTime t_end);

#endif
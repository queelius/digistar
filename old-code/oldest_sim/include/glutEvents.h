#ifndef GLUT_EVENTS_H
#define GLUT_EVENTS_H

#include "globals.h"
#include "canvas.h"
#include <list>

extern std::vector<Object*> o;
extern Object *player;
extern vector<Object*>::const_iterator selObject;
extern Canvas canvas;

struct DelayedUpdate {
    Object *objectRef;
    GLfloat *newVelocity;
};

void applyCollisions();
void mouseEvent(int mouseButton, int mouseState, int mouseXPosition, int mouseYPosition);
void keyEvent(unsigned char pressedKey, int mouseXPosition, int mouseYPosition);
void specialKeyEvent(int pressedKey, int mouseXPosition, int mouseYPosition);
void updateEvent(int value);
void refreshEvent();
void resizeEvent(GLsizei w, GLsizei h);

#endif
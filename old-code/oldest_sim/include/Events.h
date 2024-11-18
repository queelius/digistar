#ifndef EVENTS_H
#define EVENTS_H

#include "Globals.h"
#include "OrthoCam.h"
#include "VectorSpace.h"
#include "Planetoid.h"
#include "Utilities.h"
#include <list>
#include <iostream>
using namespace PhysicsTest;

namespace PhysicsTest {

extern std::list<Object*> world;
extern OrthoCam cam;

namespace Events {
    void mouseButtonEvent(int button, int state, int x, int y);
    void keyboardEvent(unsigned char key, int x, int y);
    void specialKeyboardEvent(int key, int x, int y);
    void timerEvent(int value);
    void displayEvent();
    void reshapeEvent(GLsizei w, GLsizei h);
}

}

#endif
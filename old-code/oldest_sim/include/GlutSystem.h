#ifndef GLUT_SYSTEM_H
#define GLUT_SYSTEM_H

#include <cstdlib>
#include "glut.h"
#include "OrthoCam.h"

namespace GlutSystem
{
    OrthoCam     cam;
    double       translateUnit;
    unsigned int timerDelay;     // in milliseconds; lowerbound
    double       scale;          // camera zoom factor
    unsigned int pollInterval;   // joystick polling interval

    void fullScreen(void);

    // repeatMode: GLUT_KEY_REPEAT_OFF, GLUT_KEY_REPEAT_ON, GLUT_KEY_REPEAT_DEFAULT
    void setKeyRepeat(int repeatMode);
    void idleFunc();
    void init(unsigned int timerDelay = 25);
    void joystickFunc(unsigned int buttonMask, int x, int y, int z);
    void visibilityFunc(int state);
    void passiveMotionFunc(int x, int y);

    // Called when the mouse moves within the window while one or more mouse buttons are pressed.
    void motionFunc(int x, int y);
    void entryFunc(int state);

    // Called when a user presses and releases mouse buttons; each press and each release generates a callback.
    //
    // The button parameter is one of GLUT_LEFT_BUTTON, GLUT_MIDDLE_BUTTON, or GLUT_RIGHT_BUTTON
    // The state parameter is either GLUT_UP or GLUT_DOWN indicating whether the callback was due to a release or press respectively.
    // The x and y parameters indicate the window relative coordinates when the mouse button state changed.
    void mouseFunc(int button, int state, int x, int y);
    void specialFunc(int key, int x, int y);
    void specialUpFunc(int key, int x, int y);

    // Called when a user types into the window, each key press generating an ASCII character will generate a keyboard callback.
    // The key parameter is the generated ASCII character.
    // The x and y parameters indicate the mouse location in window relative coordinates when the key was pressed.
    void keyboardFunc(unsigned char key, int x, int y);

    // Called when the keyboard up (key release) is detected
    //static void keyboardUpFunc(unsigned char key, int x, int y);
    void timerFunc(int value);
    void displayFunc();
    void reshapeFunc(int width, int height);
};

#endif
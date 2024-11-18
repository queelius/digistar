#include "GlutSystem.h"

#include <iostream>

void GlutSystem::fullScreen(void)
{
    glutFullScreen();
}

void GlutSystem::setKeyRepeat(int repeatMode)
{
    glutSetKeyRepeat(repeatMode);
}

void GlutSystem::idleFunc()
{
}

void GlutSystem::init(unsigned int timerDelay)
{
    cam.setViewDimensions(2, 2);
    translateUnit = 0.1;
    timerDelay = timerDelay;
    scale = 1.0;
    pollInterval = 25;

    std::cout << "???" << std::endl;

    glutReshapeFunc(reshapeFunc);
    std::cout << "???" << std::endl;

    glutDisplayFunc(displayFunc);
    std::cout << "???" << std::endl;

    glutIdleFunc(idleFunc);
    std::cout << "???" << std::endl;

    glutSpecialFunc(specialFunc);
    std::cout << "???" << std::endl;

    glutMouseFunc(mouseFunc);
    std::cout << "???" << std::endl;

    glutKeyboardFunc(keyboardFunc);
    std::cout << "???" << std::endl;

    glutTimerFunc(timerDelay, timerFunc, 1);
    glutEntryFunc(entryFunc);
    glutMotionFunc(motionFunc);
    glutPassiveMotionFunc(passiveMotionFunc);
    //glutKeyboardUpFunc(keyboardUpFunc);
    glutJoystickFunc(joystickFunc, pollInterval);
    glutSpecialUpFunc(specialUpFunc);
    glutVisibilityFunc(visibilityFunc);

    glutMainLoop();
}

void GlutSystem::joystickFunc(unsigned int buttonMask, int x, int y, int z)
{
}

void GlutSystem::visibilityFunc(int state)
{
    switch (state)
    {
        case GLUT_NOT_VISIBLE:
            break;
        case GLUT_VISIBLE:
            break;
        default:
            break;
    }
}

void GlutSystem::passiveMotionFunc(int x, int y)
{
}

void GlutSystem::motionFunc(int x, int y)
{
}
    
void GlutSystem::entryFunc(int state)
{
    switch (state)
    {
        case GLUT_LEFT:
            break;
        case GLUT_ENTERED:
            break;
        default:
            break;
    }
}

void GlutSystem::mouseFunc(int button, int state, int x, int y)
{
    int modifier = glutGetModifiers();
    // GLUT_ACTIVE_SHIFT    /* Set if the Shift modifier or Caps Lock is active. */
    // GLUT_ACTIVE_CTRL     /* Set if the Ctrl modifier is active. */
    // GLUT_ACTIVE_ALT      /* Set if the Alt modifier is active. */

    switch (button)
    {
        case GLUT_LEFT_BUTTON:
            break;
        case GLUT_MIDDLE_BUTTON:
            break;
        case GLUT_RIGHT_BUTTON:
            break;
        default:
            break;
    }
}

void GlutSystem::specialFunc(int key, int x, int y)
{
    int modifier = glutGetModifiers();
    // GLUT_ACTIVE_SHIFT    /* Set if the Shift modifier or Caps Lock is active. */
    // GLUT_ACTIVE_CTRL     /* Set if the Ctrl modifier is active. */
    // GLUT_ACTIVE_ALT      /* Set if the Alt modifier is active. */

    // glutIgnoreKeyRepeat();

    switch (key)
    {
        case GLUT_KEY_F1:
            break;
        case GLUT_KEY_F2:
            break;
        case GLUT_KEY_F3:
            break;
        case GLUT_KEY_F4:
            break;
        case GLUT_KEY_F5:
            break;
        case GLUT_KEY_F6:
            break;
        case GLUT_KEY_F7:
            break;
        case GLUT_KEY_F8:
            break;
        case GLUT_KEY_F9:
            break;
        case GLUT_KEY_F10:
            break;
        case GLUT_KEY_F11:
            break;
        case GLUT_KEY_F12:
            break;
        case GLUT_KEY_LEFT:
            cam.translate(-translateUnit, 0);
            break;
        case GLUT_KEY_UP:
            cam.translate(0, translateUnit);
            break;
        case GLUT_KEY_RIGHT:
            cam.translate(translateUnit, 0);
            break;
        case GLUT_KEY_DOWN:
            cam.translate(0, -translateUnit);
            break;
        case GLUT_KEY_PAGE_UP:
            break;
        case GLUT_KEY_PAGE_DOWN:
            break;
        case GLUT_KEY_HOME:
            break;
        case GLUT_KEY_END:
            break;
        case GLUT_KEY_INSERT:
            break;
        default:
            break;
    }
}

void GlutSystem::specialUpFunc(int key, int x, int y)
{
    int modifier = glutGetModifiers();
    // GLUT_ACTIVE_SHIFT    /* Set if the Shift modifier or Caps Lock is active. */
    // GLUT_ACTIVE_CTRL     /* Set if the Ctrl modifier is active. */
    // GLUT_ACTIVE_ALT      /* Set if the Alt modifier is active. */

    // glutIgnoreKeyRepeat();

    switch (key)
    {
        case GLUT_KEY_F1:
            break;
        case GLUT_KEY_F2:
            break;
        case GLUT_KEY_F3:
            break;
        case GLUT_KEY_F4:
            break;
        case GLUT_KEY_F5:
            break;
        case GLUT_KEY_F6:
            break;
        case GLUT_KEY_F7:
            break;
        case GLUT_KEY_F8:
            break;
        case GLUT_KEY_F9:
            break;
        case GLUT_KEY_F10:
            break;
        case GLUT_KEY_F11:
            break;
        case GLUT_KEY_F12:
            break;
        case GLUT_KEY_LEFT:
            break;
        case GLUT_KEY_UP:
            break;
        case GLUT_KEY_RIGHT:
            break;
        case GLUT_KEY_DOWN:
            break;
        case GLUT_KEY_PAGE_UP:
            break;
        case GLUT_KEY_PAGE_DOWN:
            break;
        case GLUT_KEY_HOME:
            break;
        case GLUT_KEY_END:
            break;
        case GLUT_KEY_INSERT:
            break;
        default:
            break;
    }
}

void GlutSystem::keyboardFunc(unsigned char key, int x, int y)
{
    int modifier = glutGetModifiers();
    // GLUT_ACTIVE_SHIFT    /* Set if the Shift modifier or Caps Lock is active. */
    // GLUT_ACTIVE_CTRL     /* Set if the Ctrl modifier is active. */
    // GLUT_ACTIVE_ALT      /* Set if the Alt modifier is active. */

    // glutIgnoreKeyRepeat();

    switch(key)
    {
        case 'q':
        case 'Q':
            exit(0);
        default:
            break;
    }
}

//void GlutSystem::keyboardUpFunc(unsigned char key, int x, int y)
//{
//    int modifier = glutGetModifiers();
//    // GLUT_ACTIVE_SHIFT    /* Set if the Shift modifier or Caps Lock is active. */
//    // GLUT_ACTIVE_CTRL     /* Set if the Ctrl modifier is active. */
//    // GLUT_ACTIVE_ALT      /* Set if the Alt modifier is active. */
//
//    // glutIgnoreKeyRepeat();
//
//    switch(key)
//    {
//        default:
//            break;
//    }
//}

void GlutSystem::timerFunc(int value)
{
    glutPostRedisplay();
    glutTimerFunc(timerDelay, timerFunc, 1);
}

void GlutSystem::displayFunc()
{
    // glutPostRedisplay();

    glClear(GL_COLOR_BUFFER_BIT);
    
    glutSwapBuffers();
	glFlush();
}

void GlutSystem::reshapeFunc(int width, int height)
{
    cam.reshape(width, height);
}
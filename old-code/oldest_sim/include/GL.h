#ifndef GL_H
#define GL_H

#include <cmath>
#include <vector>
#include "stdlib.h"
#include "glut.h"
#include "Random.h"
#include "GL_Constants.h"
#include "SoftBody.h"

namespace GL {
    // represents the window for the windowed application. most of the routines
    // were implemented elsewhere, this just gives a convenient interface, like
    // defining window pixel width and internal width and providing methods to
    // translate actual x, y to screen x, y.
    class GLWindow {
    public:
        // constructor
        GLWindow(GLint pixelWidth = INITIAL_WINDOW_PIXEL_WIDTH,
                 GLint pixelHeight = INITIAL_WINDOW_PIXEL_HEIGHT,
                 GLfloat WINDOW_WIDTH = 3.0,
                 GLfloat WINDOW_HEIGHT = 2.0,
                 GLint pixelLeftX = INITIAL_WINDOW_X,
                 GLint pixelTopY = INITIAL_WINDOW_Y);

        // translate the pixelX and pixelY coordinates into actual screen x, y coordinates
        void getXY(GLfloat &x, GLfloat &y, GLint pixelX, GLint pixelY);

        // prepare application window for Operating System
        void init(const char title[]);

        void scale(double value);

        // initalizet various application window parameters.
        // pixelWidth, pixelHeight: number of pixels wide and high for applicatin window (before resizing)
        // windowWidth, windowHeight: interal representation of coordinates
        // pixelLeftX, pixelTopY: left-most and top-most pixels for window
        void set(GLint pixelWidth, GLint pixelHeight,
                 GLfloat windowWidth, GLfloat windowHeight,
                 GLint pixelLeftX, GLint pixelTopY);

        GLint pixelWidth;
        GLint pixelHeight;

        GLint pixelLeftX;
        GLint pixelTopY;

        GLfloat windowWidth;
        GLfloat windowHeight;
    };
};

#endif
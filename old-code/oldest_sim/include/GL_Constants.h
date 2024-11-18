#ifndef GL_CONSTANTS
#define GL_CONSTANTS
#include "stdlib.h"
#include "glut.h"

// this is where the GL (graphics library) constants go. to use any GL functions or classes, you must include this file.
namespace GL {
    enum COLOR {WHITE, BLACK, RED, GREEN, BLUE, YELLOW, GOLD, GOLDEN_ROD}; // enumerated colors for Color class

    const GLfloat HALF = 0.5f;

    const GLfloat DEFAULT_RADIUS = 0.02f; // default radius for circle primitives
    const GLint DEFAULT_NUMBER_OF_SLICES = 12; // default number of triangle fans to use to approximate a circle

    // globals for window setup
    const GLfloat PI = 3.1415932385f;
    const GLfloat TWO_PI = 2.0f * PI;
    const GLfloat PI_OVER_180 = 0.0174532925f;
    const GLfloat DEFAULT_POLAR_X = 0.0f;
    const GLfloat DEFAULT_POLAR_Y = 0.0f;
    const GLfloat WINDOW_WIDTH = 3.0f;
    const GLfloat WINDOW_HEIGHT = 2.0f;

    const GLint INITIAL_WINDOW_PIXEL_WIDTH = 900;
    const GLint INITIAL_WINDOW_PIXEL_HEIGHT = 600;
    const GLint INITIAL_WINDOW_X = 100;
    const GLint INITIAL_WINDOW_Y = 100;

    // globals for the random moving circle animation
    const GLfloat ROTATION_RADIUS_LOWER_LIMIT = 0.06f;
    const GLfloat ROTATION_RADIUS_UPPER_LIMIT = 0.225f;
    const GLfloat ANGULAR_VELOCITY_LOWER_LIMIT = 0.05f;
    const GLfloat ANGULAR_VELOCITY_UPPER_LIMIT = 0.2f;
    const GLfloat ANGULAR_ACCELERATION_LOWER_LIMIT = 0.001f;
    const GLfloat ANGULAR_ACCELERATION_UPPER_LIMIT = -0.001f;
    const GLfloat DEFAULT_ANGULAR_ACCELERATION = 0.0f;
};

#endif
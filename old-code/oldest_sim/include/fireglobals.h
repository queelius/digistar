/////////////////////////////////////////////
// Globals.h - Global Constant Definitions //
/////////////////////////////////////////////

#ifndef GLOBALS_H
#define GLOBALS_H

#include "glut.h"

// ID numbers for all texture maps.
namespace TexID {
    const GLuint Wall       = 1;
    const GLuint TorchWood  = 2;
    const GLuint TorchFlame = 3;
    const GLuint TorchSmoke = 4;
};

/********************/
/* Global Constants */
/********************/

const GLfloat SMALL_EPSILON         = 0.000001;
const GLfloat MEDIUM_EPSILON        = 0.0001;
const GLfloat LARGE_EPSILON         = 0.01;

const GLfloat PI                    = 3.1415926535;
const GLfloat TWO_PI                = 2*PI;

// Window Position/Resizing/Title Constants
const GLint INIT_WINDOW_POSITION[2] = { 150, 150 };
const GLfloat ASPECT_RATIO          = 1.0;
const char *WINDOW_TITLE            = "Carrying a \"Torch\" for Particles";

// Lighting Constants
const GLfloat LIGHT_0_POSITION[]    = { 0.0, 5.0,   0.0,  0.0 };
const GLfloat LIGHT_1_POSITION[]    = { 0.0, -50.0, 50.0, 0.0 };
const GLfloat LIGHT_AMBIENT[]       = { 0.3, 0.3,   0.3,  1.0 };
const GLfloat LIGHT_DIFFUSE[]       = { 0.9, 0.9,   0.9,  1.0 };
const GLfloat LIGHT_SPECULAR[]      = { 1.0, 1.0,   1.0,  1.0 };
const GLfloat LIGHT_MODEL_AMBIENT[] = { 0.2, 0.2,   0.2,  1.0 };

// Torch texture bmp filenames
const char TORCH_WOOD_FILENAME[]    = "Bitmaps/wood.bmp";
const char TORCH_FLAME_FILENAME[]   = "Bitmaps/particleFlame.bmp";
const char TORCH_SMOKE_FILENAME[]   = "Bitmaps/particleSmoke.bmp";
// Wall texture bmp filename
const char WALL_FILENAME[]          = "Bitmaps/wall.bmp";

// Viewer Positioning Constants
const GLfloat VIEWER_DISTANCE                   = 7.5;
const GLfloat VIEWER_INITIAL_ALPHA_ANGLE        = 0.0;
const GLfloat VIEWER_INITIAL_BETA_ANGLE         = PI / 2.0;
const GLfloat VIEWER_ANGULAR_VELOCITY_INCREMENT = PI / 270.0;
const GLfloat VIEWER_FRICTION_COEFF             = 0.9;

// Torch constants
const GLfloat TORCH_RADIUS          = 0.125;
const GLfloat TORCH_HEIGHT          = 0.65;
const GLfloat TORCH_BASE_POS[]      = {0.0, 0.0, 0.0};

// Particle constants
const GLfloat MIN_VARIANCE          = 0.7;
const GLfloat MAX_VARIANCE          = 1.3;
const GLfloat DRAG_COEFF            = 0.05;
const GLfloat GRAVITY               = 0.0192;

// Flame particle constants
const GLfloat FLAME_VELOCITY        = 0.1;
const GLfloat FLAME_SIZE            = 0.5;
const GLuint  FLAME_MAX_LIFE_SPAN   = 20;
const GLuint  FLAME_MIN_LIFE_SPAN   = 1;
const GLuint  FLAME_INCREMENT       = 5;

// Smoke particle constants
const GLuint  SMOKE_INCREMENT       = 1;
const GLfloat SMOKE_VELOCITY        = 1.0/50;
const GLfloat SMOKE_SIZE            = 0.25;
const GLuint  SMOKE_MAX_LIFE_SPAN   = 35;
const GLuint  SMOKE_MIN_LIFE_SPAN   = 5;

#endif
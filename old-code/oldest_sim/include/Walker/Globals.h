/////////////////////////////////////////////
// Globals.h - Global Constant Definitions //
/////////////////////////////////////////////
#ifndef GLOBALS_H
#define GLOBALS_H

#include "glut.h"

/*******************************/
/* Global Constants - LIGHTING */
/*******************************/
const GLfloat LIGHT_0_POSITION[]    = { 0.0, 5.0,   0.0,  0.0 };
const GLfloat LIGHT_1_POSITION[]    = { 0.0, -50.0, 50.0, 0.0 };
const GLfloat LIGHT_AMBIENT[]       = { 0.3, 0.3,   0.3,  1.0 };
const GLfloat LIGHT_DIFFUSE[]       = { 0.9, 0.9,   0.9,  1.0 };
const GLfloat LIGHT_SPECULAR[]      = { 1.0, 1.0,   1.0,  1.0 };
const GLfloat LIGHT_MODEL_AMBIENT[] = { 0.2, 0.2,   0.2,  1.0 };

/************************************/
/* Global Constants - MISCELLANEOUS */
/************************************/
const GLint INIT_WINDOW_POSITION[2] = { 100, 100 };         // Screen location of upper-left window corner.
const GLfloat PI_OVER_180           = 0.0174532925f;        // 1 degree (in radians)
const GLfloat PI                    = 3.1416f;              // 180 degrees (in radians)
const GLfloat TWO_PI                = 6.2832f;              // 360 degrees (in radians)
const GLfloat EPSILON               = 0.0001f;

/*******************************/
/* Global Constants - TEXTURES */
/*******************************/
namespace TEXTURE_ID {
    const GLuint WALL  = 1;
    const GLuint FLAME = 2;
    const GLuint SMOKE = 3;
};

const char PARTICLE_FLAME_FILENAME[] = "Bitmaps/particleFlame.bmp";
const char PARTICLE_SMOKE_FILENAME[] = "Bitmaps/particleSmoke.bmp";

/********************************/
/* Global Constants - PARTICLES */
/********************************/
const GLuint  DEFAULT_SMOKE_MIN_LIFE_SPAN = 10;
const GLuint  DEFAULT_SMOKE_MAX_LIFE_SPAN = 50;
const GLfloat DEFAULT_SMOKE_SIZE          = 10;
const GLfloat DEFAULT_GRAVITY             = 0.01;
const GLfloat DEFAULT_MIN_VARIANCE        = 0.75;
const GLfloat DEFAULT_MAX_VARIANCE        = 1.25;
const GLfloat DEFAULT_DRAG_COEFF          = 0.05;


/*********************************/
/* Global Constants - SPHERE CAM */
/*********************************/
const GLfloat DEFAULT_SPHERE_CAM_DISTANCE      = -10.0;
const GLfloat DEFAULT_SPHERE_CAM_FOCAL[3]      = { 0.0, 0.0, 0.0 };
const GLfloat DEFAULT_SPHERE_CAM_THETA         = 0.0;
const GLfloat DEFAULT_SPHERE_CAM_PHI           = PI / 2.0;
const GLfloat DEFAULT_SPHERE_CAM_ANGULAR_ACCEL = PI / 270.0;
const GLfloat DEFAULT_SPHERE_CAM_ACCEL         = 0.1;
const GLfloat DEFAULT_SPHERE_CAM_FRICTION      = 1.125;

/******************************************/
/* Global Constants - MATERIAL PROPERTIES */
/******************************************/
const GLfloat AMBIENT_COEFF  = -1.0; // Minimal ambient reflectance.
const GLfloat DIFFUSE_COEFF  =  1.0; // Maximal diffuse reflectance.
const GLfloat SPECULAR_COEFF =  1.0; // Maximal specular reflectance.
const GLfloat SPECULAR_EXPON = 20.0; // Low level of shininess (scale: 0-128).

/******************************************/
/* Global Constants - LIGHTING PROPERTIES */
/******************************************/
const GLfloat LT_AMBIENT_COEFF  =  0.5;                // Medium ambient light intensity.
const GLfloat LT_DIFFUSE_COEFF  =  0.9;                // High diffuse light intensity.
const GLfloat LT_SPECULAR_COEFF =  0.8;                // High specular light intensity.
const GLfloat LT_CONST_ATTEN    =  0.0;                // No constant light attenuation.
const GLfloat LT_LIN_ATTEN      =  0.04;               // Very low linear light attenuation.
const GLfloat LT_QUAD_ATTEN     =  0.0;                // No quadratic light attenuation.
const GLfloat LT_POS[]          = { 14.0, 14.0, 14.0}; // 3D location of light source.

/********************************************/
/* Global Constants - Walker PROPERTIES */
/********************************************/
const GLfloat TORSO_AND_HEAD_SPHERE_RADIUS       = 2.6;
const GLfloat TORSO_AND_HEAD_ABOVE_SPHERE_RADIUS = 1.3;
const GLfloat TORSO_AND_HEAD_SPHERE_CENTER[]     = { 0.0, 0.0, 0.0 }; // elbow
const GLfloat TORSO_AND_HEAD_SPHERE_COLOR[]      = { 0.5, 1.0, 0.5 };
const GLfloat TORSO_AND_HEAD_CYLINDER_RADIUS     = 0.25;
const GLfloat TORSO_AND_HEAD_CYLINDER_LENGTH     = 11.0;
const GLfloat TORSO_AND_HEAD_CYLINDER_COLOR[]    = { 0.5, 0.5, 0.5 };
const GLfloat TORSO_AND_HEAD_PITCH               = 0.0;
const GLfloat TORSO_AND_HEAD_MIN_PITCH           = 0.0;
const GLfloat TORSO_AND_HEAD_MAX_PITCH           = 0.0;
const GLfloat TORSO_AND_HEAD_PITCH_INCREMENT     = 0.0;

const GLfloat SHOULDERS_SPHERE_RADIUS    = 0.7;
const GLfloat SHOULDERS_SPHERE_CENTER[]  = { 0.0, 6.5, 0.0 };
const GLfloat SHOULDERS_SPHERE_COLOR[]   = { 1.0, 0.0, 1.0 };
const GLfloat SHOULDERS_CYLINDER_RADIUS  = 0.2;
const GLfloat SHOULDERS_CYLINDER_LENGTH  = 6.5;
const GLfloat SHOULDERS_CYLINDER_COLOR[] = { 0.75, 1.0, 1.0 };
const GLfloat SHOULDERS_ROLL             = 0.0;
const GLfloat SHOULDERS_MIN_ROLL         = -7.5;
const GLfloat SHOULDERS_MAX_ROLL         = 7.5;
const GLfloat SHOULDERS_ROLL_INCREMENT   = -0.5;
const GLfloat SHOULDERS_YAW              = 0.0;
const GLfloat SHOULDERS_MIN_YAW          = -7.5;
const GLfloat SHOULDERS_MAX_YAW          = 7.5;
const GLfloat SHOULDERS_YAW_INCREMENT    = -0.5;

const GLfloat HUMERUS_SPHERE_RADIUS    = 0.6;
const GLfloat HUMERUS_SPHERE_CENTER[]  = { 0.0, 0.0, 0.0 };
const GLfloat HUMERUS_SPHERE_COLOR[]   = { 0.0, 0.75, 0.75 };
const GLfloat HUMERUS_CYLINDER_RADIUS  = 0.3;
const GLfloat HUMERUS_CYLINDER_LENGTH  = 2.5;
const GLfloat HUMERUS_CYLINDER_COLOR[] = { 1.0, 0.875, 0.75 };
const GLfloat HUMERUS_PITCH            = 0.0;
const GLfloat HUMERUS_MIN_PITCH        = -45.0;
const GLfloat HUMERUS_MAX_PITCH        = 45.0;
const GLfloat HUMERUS_PITCH_INCREMENT  = 3.0;

const GLfloat FOREARM_SPHERE_RADIUS    = 0.4;
const GLfloat FOREARM_SPHERE_CENTER[]  = { 0.0, -HUMERUS_CYLINDER_LENGTH, 0.0 };
const GLfloat FOREARM_SPHERE_COLOR[]   = { 0.75, 1.0, 0.0 };
const GLfloat FOREARM_CYLINDER_RADIUS  = 0.3;
const GLfloat FOREARM_CYLINDER_LENGTH  = 2.5;
const GLfloat FOREARM_CYLINDER_COLOR[] = { 0.875, 1.0, 0.75 };
const GLfloat FOREARM_PITCH            = 0.0;
const GLfloat FOREARM_MIN_PITCH        = 0.0;
const GLfloat FOREARM_MAX_PITCH        = 90.0;
const GLfloat FOREARM_PITCH_INCREMENT  = 3.0;

const GLfloat HAND_SPHERE_RADIUS    = 0.5;
const GLfloat HAND_SPHERE_CENTER[]  = { 0.0, 0.4, 0.0 };
const GLfloat HAND_SPHERE_COLOR[]   = { 1.0, 1.0, 0.0 };
const GLfloat HAND_CYLINDER_RADIUS  = 0.0;
const GLfloat HAND_CYLINDER_LENGTH  = 0.0;
const GLfloat HAND_CYLINDER_COLOR[] = { 0.0, 1.0, 1.0 };
const GLfloat HAND_PITCH            = 0.0;
const GLfloat HAND_MIN_PITCH        = 0.0;
const GLfloat HAND_MAX_PITCH        = 0.0;
const GLfloat HAND_PITCH_INCREMENT  = 0.0;

const GLfloat HIPS_SPHERE_RADIUS    = 0.8;
const GLfloat HIPS_SPHERE_CENTER[]  = { 0.0, 0.0, 0.0 };
const GLfloat HIPS_SPHERE_COLOR[]   = { 0.0, 1.0, 1.0 };
const GLfloat HIPS_CYLINDER_RADIUS  = 0.25;
const GLfloat HIPS_CYLINDER_LENGTH  = 3.5;
const GLfloat HIPS_CYLINDER_COLOR[] = { 0.75, 1.0, 1.0 };
const GLfloat HIPS_ROLL             = 0.0;
const GLfloat HIPS_MIN_ROLL         = -7.5;
const GLfloat HIPS_MAX_ROLL         = 7.5;
const GLfloat HIPS_ROLL_INCREMENT   = 0.5;
const GLfloat HIPS_YAW              = 0.0;
const GLfloat HIPS_MIN_YAW          = -7.5;
const GLfloat HIPS_MAX_YAW          = 7.5;
const GLfloat HIPS_YAW_INCREMENT    = 0.5;

const GLfloat THIGH_SPHERE_RADIUS    = 0.6;
const GLfloat THIGH_SPHERE_CENTER[]  = { 0.0, 0.0, 0.0 };
const GLfloat THIGH_SPHERE_COLOR[]   = { 1.0, 0.75, 0.0 };
const GLfloat THIGH_CYLINDER_RADIUS  = 0.25;
const GLfloat THIGH_CYLINDER_LENGTH  = 3.5;
const GLfloat THIGH_CYLINDER_COLOR[] = { 1.0, 0.875, 0.75 };
const GLfloat THIGH_PITCH            = 0.0;
const GLfloat THIGH_MIN_PITCH        = -45.0;
const GLfloat THIGH_MAX_PITCH        = 45.0;
const GLfloat THIGH_PITCH_INCREMENT  = 3.0;

const GLfloat SHIN_SPHERE_RADIUS    = 0.5;
const GLfloat SHIN_SPHERE_CENTER[]  = { 0.0, -THIGH_CYLINDER_LENGTH, 0.0 };
const GLfloat SHIN_SPHERE_COLOR[]   = { 0.0, 0.0, 1.0 };
const GLfloat SHIN_CYLINDER_RADIUS  = 0.25;
const GLfloat SHIN_CYLINDER_LENGTH  = 3.5;
const GLfloat SHIN_CYLINDER_COLOR[] = { 0.875, 1.0, 0.75 };
const GLfloat SHIN_PITCH            = 0.0;
const GLfloat SHIN_MIN_PITCH        = 0.0;
const GLfloat SHIN_MAX_PITCH        = 90.0;
const GLfloat SHIN_PITCH_INCREMENT  = 3.0;

const GLfloat HEEL_SPHERE_RADIUS    = 0.85;
const GLfloat HEEL_SPHERE_CENTER[]  = { 0.0, -SHIN_CYLINDER_LENGTH, 0.0 };
const GLfloat HEEL_SPHERE_COLOR[]   = { 1.0, 1.0, 1.0 };
const GLfloat HEEL_CYLINDER_RADIUS  = 0.25;
const GLfloat HEEL_CYLINDER_LENGTH  = 1.5;
const GLfloat HEEL_CYLINDER_COLOR[] = { 0.75, 0.75, 0.75 };
const GLfloat HEEL_PITCH            = -90.0;
const GLfloat HEEL_MIN_PITCH        = -105.0;
const GLfloat HEEL_MAX_PITCH        = -75.0;
const GLfloat HEEL_PITCH_INCREMENT  = 1.0;

const GLfloat TOE_SPHERE_RADIUS    = 0.7;
const GLfloat TOE_SPHERE_CENTER[]  = { 0.0, 0.0, HEEL_CYLINDER_LENGTH };
const GLfloat TOE_SPHERE_COLOR[]   = { 1.0, 1.0, 1.0 };
const GLfloat TOE_CYLINDER_RADIUS  = 0.0;
const GLfloat TOE_CYLINDER_LENGTH  = 0.0;
const GLfloat TOE_CYLINDER_COLOR[] = { 0.0, 0.0, 0.0 };
const GLfloat TOE_PITCH            = 0.0;
const GLfloat TOE_MIN_PITCH        = 0.0;
const GLfloat TOE_MAX_PITCH        = 0.0;
const GLfloat TOE_PITCH_INCREMENT  = 0.0;

#endif
//  Assignment: Program 2 - 3D Animation
//    Filename: WalkerDriver.cpp
//        Name: Alex Towell
//    Due Date: 9-29-2009
//      Course: CS 482 (Computer Graphics)
// Description: This program displays a running character
//              composed of spheres and linking cylinders.
//              The character's legs and hips are are
//              designed to move in a "natural" fashion".

#include <windows.h>
#include <iostream>
#include "glut.h"
#include "Common.h"
#include "Globals.h"
#include "MobileCam.h"
#include "SimpleRoom.h"
#include "ParticleSource.h"
#include "Walker.h"

//////////////////////
// Global Variables //
//////////////////////
GLint     currWindowSize[2] = { 800, 800 }; // Window size in pixels.
Walker    guy;                              // Animated character.
bool      doAnimation       = false;        // Animation mode on/off

MobileCam cam;
ParticleSource src;

/////////////////////////
// Function Prototypes //
/////////////////////////
void MakeImages();
void MakeImage(const char bmpFilename[], const GLuint &textureID, bool hasAlpha);

void drawPoints();
void SetMaterialProperties();
void SetLightingProperties();
void TimerFunction(int value);
void Display();
void ResizeWindow(GLsizei w, GLsizei h);
void SetUpCamera();
void SetLights();
void UpdateLight();

void KeyboardEvent(unsigned char pressedKey, int mouseXPosition, int mouseYPosition);
void KeyboardSpecialEvent(int pressedKey, int mouseXPosition, int mouseYPosition);

// The main function: uses the OpenGL Utility Toolkit to set
// the window up to display the window and its contents.
void main(int argc, char **argv) {
	// Set up the display window.
	glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA | GLUT_DEPTH);
    glutInitWindowPosition( INIT_WINDOW_POSITION[0], INIT_WINDOW_POSITION[1] );
	glutInitWindowSize( currWindowSize[0], currWindowSize[1] );
    glutCreateWindow("Programming Assignment #5");

	// Specify the resizing, refreshing, and interactive routines.
	glutReshapeFunc( ResizeWindow );
	glutDisplayFunc( Display );
	glutTimerFunc(25, TimerFunction, 1);

    glutKeyboardFunc(KeyboardEvent);
    glutSpecialFunc(KeyboardSpecialEvent);

    seed();
    MakeImages();
    guy.reset();

    GLfloat pos[3] = {0.0, 2.5, 0.0};
    GLfloat pos2[3] = {0.0, 2.5, 5.0};
    src.init(pos2, PI/2, PI/2, TEXTURE_ID::SMOKE, 0.5, 1, 10);
    cam.init(pos, 1.0, PI/2, 0.01, 1.125);

    // Set up standard lighting, shading, and depth testing.
	//glDepthFunc(GL_LEQUAL); // The Type Of Depth Testing To Do
	//glHint(GL_PERSPECTIVE_CORRECTION_HINT, GL_NICEST); // Really Nice Perspective Calculations
	//glCullFace(GL_BACK);	
	//glEnable(GL_CULL_FACE);
	//glShadeModel(GL_SMOOTH);
	//glEnable(GL_DEPTH_TEST);
	//glEnable(GL_NORMALIZE);
    glClearColor(0, 0, 0, 0);
	glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    
    // Set up standard lighting, shading, and depth testing.
   	SetLights();

	glViewport(0, 0, currWindowSize[0], currWindowSize[1]);
	glutMainLoop();
}

// Function to react to non-ASCII keyboard keys pressed by the user.
// Used to alter coordinates of the viewer's position by altering the
// angular velocities
void KeyboardSpecialEvent(int pressedKey, int mouseXPosition, int mouseYPosition) {
	glutIgnoreKeyRepeat(false);
    
    switch (pressedKey) {
        case GLUT_KEY_LEFT:  cam.accelerateLeft();     break;
        case GLUT_KEY_RIGHT: cam.accelerateRight();    break;
        case GLUT_KEY_UP:    cam.accelerateForward();  break;
        case GLUT_KEY_DOWN:  cam.accelerateBackward(); break;
    }
}

// Handles key press events for: p|P, r|R
// r=reset, p=toggles pause
void KeyboardEvent(unsigned char pressedKey, int mouseXPosition, int mouseYPosition) {
    switch (pressedKey) {
        case 'p': case 'P': // toggle pause
            doAnimation = !doAnimation; break;
        case 'r': case 'R': // reset world to default state (walker, camera, etc)
            guy.reset();
            cam.reset();
            doAnimation = false;
            glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
            glutPostRedisplay();
            break;
    }
}

// Set up the material properties of all displayed objects in the scene. Any objects with
// different properties (e.g., light emissiveness) are set up individually within the code.
void SetMaterialProperties() {
	GLfloat mat_ambient[]   = { AMBIENT_COEFF,  AMBIENT_COEFF,  AMBIENT_COEFF,  1.0 };
	GLfloat mat_diffuse[]   = { DIFFUSE_COEFF,  DIFFUSE_COEFF,  DIFFUSE_COEFF,  1.0 };
	GLfloat mat_specular[]  = { SPECULAR_COEFF, SPECULAR_COEFF, SPECULAR_COEFF, 1.0 };
	GLfloat mat_shininess[] = { SPECULAR_EXPON };

	glMaterialfv(GL_FRONT, GL_AMBIENT,   mat_ambient);
	glMaterialfv(GL_FRONT, GL_DIFFUSE,   mat_diffuse);
	glMaterialfv(GL_FRONT, GL_SPECULAR,  mat_specular);
	glMaterialfv(GL_FRONT, GL_SHININESS, mat_shininess);

	glColorMaterial(GL_FRONT, GL_AMBIENT_AND_DIFFUSE);
	glEnable(GL_COLOR_MATERIAL);
}

// Set up the lighting properties being used in the displayed scene.
void SetLightingProperties() {
	GLfloat lt_ambient[]    = { LT_AMBIENT_COEFF,  LT_AMBIENT_COEFF,  LT_AMBIENT_COEFF,  1.0 };
	GLfloat lt_diffuse[]    = { LT_DIFFUSE_COEFF,  LT_DIFFUSE_COEFF,  LT_DIFFUSE_COEFF,  1.0 };
	GLfloat lt_specular[]   = { LT_SPECULAR_COEFF, LT_SPECULAR_COEFF, LT_SPECULAR_COEFF, 1.0 };
	GLfloat lt_constant_attenuation[]  = { LT_CONST_ATTEN };
	GLfloat lt_linear_attenuation[]    = { LT_LIN_ATTEN };
	GLfloat lt_quadratic_attenuation[] = { LT_QUAD_ATTEN };
	GLfloat light_position0[] = { LT_POS[0], LT_POS[1], LT_POS[2], 1.0 };

	glLightfv(GL_LIGHT0, GL_AMBIENT,               lt_ambient);
	glLightfv(GL_LIGHT0, GL_DIFFUSE,               lt_diffuse);
	glLightfv(GL_LIGHT0, GL_SPECULAR,              lt_specular);
	glLightfv(GL_LIGHT0, GL_CONSTANT_ATTENUATION,  lt_constant_attenuation);
	glLightfv(GL_LIGHT0, GL_LINEAR_ATTENUATION,    lt_linear_attenuation);
	glLightfv(GL_LIGHT0, GL_QUADRATIC_ATTENUATION, lt_quadratic_attenuation);
	glLightfv(GL_LIGHT0, GL_POSITION,              light_position0);

	glEnable(GL_LIGHTING);
	glEnable(GL_LIGHT0);
}

// Function to update all of the program's animation.
void TimerFunction(int value) {
    if (doAnimation) {
	    guy.update();
    }
    cam.update();
    src.update();

    glutPostRedisplay();
    glutTimerFunc(25, TimerFunction, 1);
}

// Function to draw all displayed objects. Since no Ground class has been
// included, its display is handled within the driver, unlike the Walker.
void Display() {
    SetUpCamera();

    // Set up the viewer's perspective.
	// Position and orient viewer.
    cam.orient();
    UpdateLight();

	guy.draw();
    src.emit();

	glutSwapBuffers();
	glFlush();
}

// Window-reshaping routine, to scale the
// scene according to the window dimensions.
void ResizeWindow(GLsizei w, GLsizei h) {
	glViewport(0, 0, w, h);
	currWindowSize[0] = w;
	currWindowSize[1] = h;
    SetUpCamera();
}

// Set up the display camera to provide perspective projection.
void SetUpCamera() {
	// Initialize lighting.
	glLightModelfv(GL_LIGHT_MODEL_AMBIENT, LIGHT_MODEL_AMBIENT);	
	glEnable(GL_LIGHTING);
	
	// Set up the properties of the viewing camera.
	gluPerspective( 45.0, (GLfloat)currWindowSize[0] / (GLfloat)currWindowSize[1], 1.0f, 200.0f );
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT| GL_STENCIL_BUFFER_BIT);
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();

	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
}

void drawPoints() {
    glPushMatrix();
        glColor3f(1.0, 1.0, 1.0); // floor and ceiling color
        glPointSize(2.0);
        glBegin(GL_POINTS); // draw walls
            glVertex3f(-2.0, 2.5, 0.0);
            glVertex3f(2.0, 2.5, 0.0);
        glEnd();
    glPopMatrix();
}

void MakeImages() {
    MakeImage(PARTICLE_SMOKE_FILENAME, TEXTURE_ID::SMOKE, false);
}

// Convert the bitmap with the parameterized name into an OpenGL texture. //
void MakeImage(const char bitmapFilename[], const GLuint &textureID, bool hasAlpha) {
	RGBpixmap pix;
	pix.readBMPFile(bitmapFilename, hasAlpha);
	pix.setTexture(textureID);
}

void SetLights() {
	glLightfv(GL_LIGHT0, GL_AMBIENT,  LIGHT_AMBIENT);
	glLightfv(GL_LIGHT0, GL_DIFFUSE,  LIGHT_DIFFUSE);
	glLightfv(GL_LIGHT0, GL_SPECULAR, LIGHT_SPECULAR);
	glLightfv(GL_LIGHT0, GL_POSITION, LIGHT_0_POSITION);

	glLightfv(GL_LIGHT1, GL_AMBIENT,  LIGHT_AMBIENT);
	glLightfv(GL_LIGHT1, GL_DIFFUSE,  LIGHT_DIFFUSE);
	glLightfv(GL_LIGHT1, GL_SPECULAR, LIGHT_SPECULAR);
	glLightfv(GL_LIGHT1, GL_POSITION, LIGHT_1_POSITION);
}


// Enable the scene's lighting. //
void UpdateLight() {
	glPushMatrix();
		glLightfv(GL_LIGHT0, GL_POSITION, LIGHT_0_POSITION);
		glLightfv(GL_LIGHT1, GL_POSITION, LIGHT_1_POSITION);
	glPopMatrix();
	
	glEnable(GL_LIGHT0);
	glEnable(GL_LIGHT1);
}

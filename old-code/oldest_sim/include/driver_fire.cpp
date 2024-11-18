///////////////////////////////////////////////////////
//  Assignment: Program 4                            //
//    Filename: driver.cpp                           //
//        Name: Alex Towell                          //
//    Due Date: 11-3-2009                            //
//      Course: CS 482 (Computer Graphics)           //
// Description: Torch program. Implementation of two //
// particle systems: flame and smoke.                //
///////////////////////////////////////////////////////

#include <windows.h>
#include <vector>
#include <iostream>
#include <math.h>
#include "glut.h"
#include "RGBpixmap.h" 
#include "globals.h"
#include "objects.h"
using namespace std;

/********************/
/* Global Variables */
/********************/

Torch TheTorch;
//Torch Torches[10];

GLfloat angularVelocityBeta  = 0;
GLfloat angularVelocityAlpha = 0;
GLfloat viewerAngleAlpha     = VIEWER_INITIAL_ALPHA_ANGLE;   // left / right viewer angle
GLfloat viewerAngleBeta      = VIEWER_INITIAL_BETA_ANGLE;    // up / down viewer angle

// The initial window and viewport sizes (in pixels), set to ensure that
// the aspect ration for the viewport, will be a constant. If the window
// is resized, the viewport will be adjusted to preserve the aspect ratio.
GLint currWindowSize[2]   = { 700, 700 / ASPECT_RATIO };
GLint currViewportSize[2] = { 700, 700 / ASPECT_RATIO };

/***********************/
/* Function prototypes */
/***********************/

// Event handlers
void KeyboardPress(unsigned char pressedKey, int mouseXPosition, int mouseYPosition);
void NonASCIIKeyboardPress(int pressedKey, int mouseXPosition, int mouseYPosition);
void TimerFunction(int value);
void Display();
void ResizeWindow(GLsizei w, GLsizei h);

// texturizers
void MakeImages();
void MakeImage(const char bmpFilename[], const GLuint &texID, bool hasAlpha);

void DrawRoom(); // draw the main room
void SetLights();
void UpdateLight();
void KeyList(); // display main keyboard controls

/****************************/
/* Function implementations */
/****************************/

// The main function sets up the data and the   //
// environment to display the textured objects. //
void main() {
	KeyList(); // Display the keyboard options.

	// Set up the display window.
	glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA | GLUT_STENCIL| GLUT_DEPTH);
    glutInitWindowPosition(INIT_WINDOW_POSITION[0], INIT_WINDOW_POSITION[1]);
	glutInitWindowSize(currWindowSize[0], currWindowSize[1]);
    glutCreateWindow(WINDOW_TITLE);

	// Specify the resizing and refreshing routines.
	glutReshapeFunc(ResizeWindow);
    glutKeyboardFunc(KeyboardPress);
	glutSpecialFunc(NonASCIIKeyboardPress);
    glutDisplayFunc(Display);
	glutTimerFunc(20, TimerFunction, 1);
	glViewport(0, 0, currWindowSize[0], currWindowSize[1]);

	// Set up standard lighting, shading, and depth testing.
	glShadeModel(GL_SMOOTH);
	glEnable(GL_DEPTH_TEST);
	glDepthFunc(GL_LEQUAL); // The Type Of Depth Testing To Do
	glHint(GL_PERSPECTIVE_CORRECTION_HINT, GL_NICEST); // Really Nice Perspective Calculations
	glEnable(GL_NORMALIZE);
	glCullFace(GL_BACK);	
	glEnable(GL_CULL_FACE);
	glClearColor(0.0f, 0.0f, 0.0f, 0.9f);
	glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
	SetLights();

	MakeImages(); // Set up all texture maps and texture-mapped objects.
    seed(); // Seed random number generator, rand()

    // Initialize Torch
    TheTorch.init(TORCH_BASE_POS, TORCH_RADIUS, TORCH_HEIGHT, PI/4, PI/4);

    //GLfloat pos[3];
    //for (unsigned int i = 0; i < 10; ++i) {
    //    pos[0] = -1.5f + getRand(0.0f, 3.0f);
    //    pos[1] = -1.5f + getRand(0.0f, 3.0f);
    //    pos[2] = -1.5f + getRand(0.0f, 3.0f);
    //    Torches[i].init(pos, TORCH_RADIUS, TORCH_HEIGHT);
    //}

	glutMainLoop();
}

void KeyboardPress(unsigned char pressedKey, int mouseXPosition, int mouseYPosition) {
    switch (pressedKey) {
        case 'f':
        case 'F': TheTorch.changeMode(FLAME); break;
        case 'x':
        case 'X': TheTorch.changeMode(SMOKE); break;
    }
}

void NonASCIIKeyboardPress(int pressedKey, int mouseXPosition, int mouseYPosition) {
	glutIgnoreKeyRepeat(false);
    switch(pressedKey) {
        case GLUT_KEY_UP:    {  angularVelocityBeta  += VIEWER_ANGULAR_VELOCITY_INCREMENT; break; }
        case GLUT_KEY_DOWN:  {  angularVelocityBeta  -= VIEWER_ANGULAR_VELOCITY_INCREMENT; break; }
		case GLUT_KEY_RIGHT: {  angularVelocityAlpha += VIEWER_ANGULAR_VELOCITY_INCREMENT; break; }
        case GLUT_KEY_LEFT:  {  angularVelocityAlpha -= VIEWER_ANGULAR_VELOCITY_INCREMENT; break; }
    }
}

// Function to update the viewer's perspective. //
void TimerFunction(int value) {
    TheTorch.update();
    //for (unsigned int i = 0; i < 10; ++i)
    //    Torches[i].update();

    viewerAngleAlpha += angularVelocityAlpha; // rate of change of the angles determined by angular velocities
    viewerAngleBeta  += angularVelocityBeta;

    if (viewerAngleBeta >= PI) {
        angularVelocityBeta = 0.0;
        viewerAngleBeta = PI - SMALL_EPSILON;
    }
    if (viewerAngleBeta <= 0) {
        angularVelocityBeta = 0.0;
        viewerAngleBeta = SMALL_EPSILON;
    }
    if (viewerAngleAlpha > 2*PI) 
        viewerAngleAlpha -= 2*PI; 
    if (viewerAngleAlpha < 0.0)  
        viewerAngleAlpha += 2*PI; 

    angularVelocityAlpha *= VIEWER_FRICTION_COEFF;
    angularVelocityBeta  *= VIEWER_FRICTION_COEFF;

	glutPostRedisplay();
	glutTimerFunc(20, TimerFunction, 1);
}

// Principal display routine: sets up material, lighting, //
// and camera properties, clears the frame buffer, and    //
// draws all texture-mapped objects within the window.    //
void Display() {
	// Initialize lighting.
	glLightModelfv(GL_LIGHT_MODEL_AMBIENT, LIGHT_MODEL_AMBIENT);	
	glEnable(GL_LIGHTING);
	
	// Set up the properties of the viewing camera.
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
    gluPerspective(60.0, ASPECT_RATIO, 0.2, 100.0);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT| GL_STENCIL_BUFFER_BIT);
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();

    // Position and orient viewer.
	gluLookAt(TheTorch.base[0] + VIEWER_DISTANCE * sin(viewerAngleAlpha) * sin(viewerAngleBeta),
              TheTorch.base[1] + VIEWER_DISTANCE * cos(viewerAngleBeta),
              TheTorch.base[2] + VIEWER_DISTANCE * cos(viewerAngleAlpha) * sin(viewerAngleBeta),
              TheTorch.base[0],
              TheTorch.base[1],
              TheTorch.base[2],
              0.0, 1.0, 0.0);

	// Render scene.
	UpdateLight();
    TheTorch.draw();
    //for (unsigned int i = 0; i < 10; ++i)
    //    Torches[i].draw();

	// Reset for next time interval.
	glutSwapBuffers();
	glFlush();
}

// Create the textures associated with all texture-mapped objects being displayed. //
void MakeImages() {
    MakeImage(WALL_FILENAME,        TexID::Wall,       false);
    MakeImage(TORCH_WOOD_FILENAME,  TexID::TorchWood,  false);
    MakeImage(TORCH_FLAME_FILENAME, TexID::TorchFlame, false);
    MakeImage(TORCH_SMOKE_FILENAME, TexID::TorchSmoke, false);
}

// Convert the bitmap with the parameterized name into an OpenGL texture. //
void MakeImage(const char bitmapFilename[], const GLuint &textureName, bool hasAlpha) {
	RGBpixmap pix;
	pix.readBMPFile(bitmapFilename, hasAlpha);
	pix.setTexture(textureName);
}

// Set the two lights that will give the effect of a bright ceiling light. //
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

// Output keyboard interaction choices to the console window. //
void KeyList() {
    cout << "\n\tF Keys\tActivate Torch Flames";
    cout << "\n\tX Keys\tExtinguish Torch Flames";
	cout << endl;
}

// Window-reshaping callback, adjusting the viewport to be as large  //
// as possible within the window, without changing its aspect ratio. //
void ResizeWindow(GLsizei w, GLsizei h) {
	currWindowSize[0] = w;
	currWindowSize[1] = h;
	if (ASPECT_RATIO > w/h) {
		currViewportSize[0] = w;
		currViewportSize[1] = w / ASPECT_RATIO;
	}
    else {
        currViewportSize[1] = h;
		currViewportSize[0] = h * ASPECT_RATIO;
	}

	glViewport(0.5*(w-currViewportSize[0]), 0.5*(h-currViewportSize[1]), currViewportSize[0], currViewportSize[1]);

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluPerspective(60.0, (GLfloat)w / (GLfloat)h, 0.1, 100.0);
    glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
}
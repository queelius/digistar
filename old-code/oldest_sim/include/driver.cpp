// Assignment: Program 1 - Convex Hull
// File: driver.cpp
// Name: Alex Towell
// Date: 9-9-2009
// Course: CS 482 (Computer Graphics)

#include <windows.h>
#include <algorithm>
#include <vector>
#include <cstdlib>
#include "glut.h"
#include "Random.h"
#include "GL_Constants.h"
#include "GL.h"
#include "SoftBody.h"
//#include "GlobalConstants.h"
using namespace std;
using namespace GL;

//////////////////////
// GLOBAL VARIABLES //
//////////////////////

vector<Spring> springs;
vector<Node> nodes;

GLWindow scr; // stores setup data for screen / coordinate space
Random rndSrc; // random number generator source

int selection;
double dForce = 100000;
double appliedForce[2];
double scale = 1.0;

SpringBody softBox;
SpringBody softRope[4];

//////////////////////////////////////////
// prototypes event handles (delegates) //
//////////////////////////////////////////
void handleMouseEvent(int mouseButton, int mouseState, int mouseXPosition, int mouseYPosition);
void handleKeyEvent(unsigned char pressedKey, int mouseXPosition, int mouseYPosition);
void handleUpdateEvent(int value);
void handleRefreshEvent();
void handleResizeEvent(GLsizei w, GLsizei h);

SpringBody makeRope(double mass, double start[2], double end[2], double stiffness, unsigned int parts, bool fixedStart = true);
void applyUniformAcceleration(double a[2], SpringBody &body);

int main(int argc, char **argv) {
    nodes.reserve(1000);
    springs.reserve(1000);

    selection = 0;
    appliedForce[0] = 0;
    appliedForce[1] = 0;

    double ropeStiffness = 40000.0;
    double boxStiffness = 40000.0;
    double ropeMass = 4000.0;
    double boxMass = 500.0;

    // box definition

    // 0 1 2
    // 3 4 5
    // 6 7 8

    softBox.addNode(boxMass, -0.25, -7.5); // 0
    softBox.addNode(boxMass, 0.0, -7.5); // 1
    softBox.addNode(boxMass, 0.25, -7.5); // 2

    softBox.addNode(boxMass, -0.25, -7.75); // 3
    softBox.addNode(boxMass, 0.0, -7.725); // 4
    softBox.addNode(boxMass, 0.25, -7.75); // 5

    softBox.addNode(boxMass, -0.25, -8.0); // 6
    softBox.addNode(boxMass, 0.0, -8.0); // 7
    softBox.addNode(boxMass, 0.25, -8.0); // 8

    // row 1

    softBox.joinNodes(boxStiffness, 0, 1);
    softBox.joinNodes(boxStiffness, 0, 3);
    softBox.joinNodes(boxStiffness, 0, 4);

    softBox.joinNodes(boxStiffness, 1, 2);
    softBox.joinNodes(boxStiffness, 1, 3);
    softBox.joinNodes(boxStiffness, 1, 4);
    softBox.joinNodes(boxStiffness, 1, 5);

    softBox.joinNodes(boxStiffness, 2, 4);
    softBox.joinNodes(boxStiffness, 2, 5);

    // row 2

    softBox.joinNodes(boxStiffness, 3, 4);
    softBox.joinNodes(boxStiffness, 3, 7);
    softBox.joinNodes(boxStiffness, 3, 6);

    softBox.joinNodes(boxStiffness, 4, 5);
    softBox.joinNodes(boxStiffness, 4, 6);
    softBox.joinNodes(boxStiffness, 4, 7);
    softBox.joinNodes(boxStiffness, 4, 8);

    softBox.joinNodes(boxStiffness, 5, 7);
    softBox.joinNodes(boxStiffness, 5, 8);

    // row 3

    softBox.joinNodes(boxStiffness, 6, 7);
    softBox.joinNodes(boxStiffness, 7, 8);

    double start1[] = { 0.0, 0.0 };
    double end1[] = { 0.0, -5.0 };
    
    double start2[] = { 5.0, 0.0 };
    double end2[] = { 5.0, -5.0 };

    double start3[] = { 0.0, -2.5 };
    double end3[] = { 0.0, -7.5 };

    double start4[] = { 21, 21 };
    double end4[] = { 110.1, -41 };

    softRope[0] = makeRope(ropeMass, start1, end1, ropeStiffness, 24, true);
    softRope[1] = makeRope(ropeMass, start2, end2, ropeStiffness, 24, true);
//    joinBodies(ropeStiffness, softRope[0], 24*2-1, softRope[1], 24*2-1);

    softRope[2] = makeRope(ropeMass, start3, end3, ropeStiffness, 24, false);
  //  joinBodies(ropeStiffness, softRope[0], 12*2-1, softRope[2], 0);

    softRope[3] = makeRope(ropeMass, start4, end4, ropeStiffness, 24, false);

    joinBodies(ropeStiffness, softRope[2], 24*2-1, softBox, 4);

    // initalize graphics window
    scr.init("test");

    // Specify the resizing, displaying, and interactive routines.
    glutReshapeFunc(handleResizeEvent);
    glutDisplayFunc(handleRefreshEvent);
    glutMouseFunc(handleMouseEvent);
    glutKeyboardFunc(handleKeyEvent);
    glutTimerFunc(20, handleUpdateEvent, 1);

    glutMainLoop();
	return 0;
}

// Delegate function: creates or destroys primitives in the primtiive container
void handleMouseEvent(int mouseButton, int mouseState, int mouseXPosition, int mouseYPosition) {
    //GLfloat x, y;
    //scr.getXY(x, y, mouseXPosition, mouseYPosition);
}

// Delegate function: pressedKey is a key entered by the user; toggles various global state
// variables.
void handleKeyEvent(unsigned char pressedKey, int mouseXPosition, int mouseYPosition) {
    switch(pressedKey) {
        case '3': // down-right
            {
                appliedForce[0] += dForce;
                appliedForce[1] -= dForce;
                break;
            }
        case '1': // down-left
            {
                appliedForce[0] -= dForce;
                appliedForce[1] -= dForce;
                break;
            }
        case '9': // up-right
            {
                appliedForce[0] += dForce;
                appliedForce[1] += dForce;
                break;
            }
        case '7': // up-left
            {
                appliedForce[0] -= dForce;
                appliedForce[1] += dForce;
                break;
            }
        case '4': // left
            {
                appliedForce[0] -= dForce;
                break;
            }
        case '6': // right
            {
                appliedForce[0] += dForce;
                break;
            }
        case '2': // down
            {
                appliedForce[1] -= dForce;
                break;
            }
        case '8':
            {
                appliedForce[1] += dForce;
                break;
            }
        case '5':
            {
                selection = (selection + 1) % nodes.size();
                break;
            }
        case 'R': // reset
        case 'r':
            {
                // softBody.resetVelocity();
                break;
            }
        case '+':
            {
                if (scale < 1.0)
                    scale = 1.0;

                scale = scale + 0.01;
                glOrtho(-scale, scale, -scale, scale, -10.0, 10.0);
                break;
            }
        case '-':
            {
                if (scale > 1.0)
                    scale = 1.0;

                scale = scale - 0.01;
                glOrtho(-scale, scale, -scale, scale, -10.0, 10.0);
                break;
            }
    }
}


// Delegate function: updates the states of each primitive if animation is enabled.
void handleUpdateEvent(int value) {
    nodes[selection].addForce(appliedForce);
    appliedForce[0] = appliedForce[1] = 0.0;

    for (unsigned int i = 0; i < nodes.size(); ++i)
    {
        nodes[i].tick();
    }

    for (unsigned int i = 0; i < springs.size(); ++i)
    {
        springs[i].tick();
    }

    double g[] = {0, -10};

    // add a "add uniform force" or "impulse" method to a "world" kind of class
    // that is a container for all bodies in virtual environment
    applyUniformAcceleration(g, softBox);

    for (int i = 0; i < 4; ++i)
        applyUniformAcceleration(g, softRope[i]);

    //// find lowest primitive and polar sort primitive container
    //// with respect to it (polar angle ccw sort)
    //if (nodes.size() > 2) {
        //int lowest = findLowestBodyInComposite(softBox);
        //if (lowest != -1 && lowest != 0)
        //    swap(softBox.inodes[0], softBox.inodes[lowest]);

        //const double *p = softBox.inodes[0]->getPosition();
        //polarSort.setPole(p[0], p[1]);
        //sort(softBox.inodes.begin() + 1, softBox.inodes.end(), polarSort);
    //}
    
    glutPostRedisplay();
    glutTimerFunc(20, handleUpdateEvent, 1);
}

// Delegate function: handle the rendering of the graphics window.
void handleRefreshEvent() {
	glClear(GL_COLOR_BUFFER_BIT);

    //glOrtho(-1.0 * scale, 1.0 * scale, -1.0 * scale * (GLfloat)h / (GLfloat)w, 1.0 * scale * (GLfloat)h / (GLfloat)w, -10.0, 10.0);
    glPushMatrix();
        glPointSize(5);
        glBegin(GL_POINTS);
            for (unsigned int i = 0; i < nodes.size(); ++i) {
                if (selection == i)
                    glColor3f(0, 1, 0);
                else
                    glColor3f(1, 0, 0);
                glVertex2d(nodes[i].s[0], nodes[i].s[1]);
            }
        glEnd();
    glPopMatrix();

	glutSwapBuffers();
	glFlush();
}

// Delegate function: adjusts the virtual camera when the graphics window
// is resized; it will scale objects in the scene, and maintain proper
// aspect ratio for graphic primitives in the scene.
void handleResizeEvent(GLsizei w, GLsizei h) {
	glViewport(0, 0, w, h);

    scr.pixelWidth = w;
	scr.pixelHeight = h;

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    if (w <= h) {
        scr.windowWidth = 2.0;
        scr.windowHeight = 2.0 * (GLfloat)h / (GLfloat)w;
        //glOrtho(-1.0, 1.0, -1.0 * (GLfloat)h / (GLfloat)w, (GLfloat)h / (GLfloat)w, -10.0, 10.0);
        glOrtho(-1.0 * scale, 1.0 * scale, -1.0 * scale * (GLfloat)h / (GLfloat)w, 1.0 * scale * (GLfloat)h / (GLfloat)w, -10.0, 10.0);
	}
	else {
        scr.windowWidth = 2.0 * (GLfloat)w / (GLfloat)h;
        scr.windowHeight = 2.0;
        //glOrtho(-1.0 * (GLfloat)w / (GLfloat)h, (GLfloat)w / (GLfloat)h, -1.0, 1.0, -10.0, 10.0);
        glOrtho(-1.0 * scale * (GLfloat)w / (GLfloat)h, 1.0 * scale * (GLfloat)w / (GLfloat)h, -1.0 * scale, 1.0 * scale, -10.0, 10.0);
	}
    glMatrixMode(GL_MODELVIEW);
}


SpringBody makeRope2(double mass, double start[2], double end[2], double stiffness, unsigned int parts, bool fixedStart, bool fixedEnd)
{
    SpringBody rope;

    double unitMass = mass / (double)parts;
    double dx = (end[0] - start[0]) / (double)parts;
    double dy = (end[1] - start[1]) / (double)parts;

    double x = start[0];
    double y = start[1];
    
    rope.addNode(unitMass, x, y, fixedStart);
    for (unsigned int i = 1; i < parts - 1; ++i)
    {
        x += dx; y += dy;
        rope.addNode(unitMass, x, y, false);
        rope.joinNodes(stiffness, i - 1, i);
    }
    
    if (parts > 1)
    {
        x += dx; y += dy;
        rope.addNode(unitMass, x, y, fixedEnd);
        rope.joinNodes(stiffness, parts - 1, parts - 2);
    }

    return rope;
}

SpringBody makeRope(double mass, double start[2], double end[2], double stiffness, unsigned int parts, bool fixedStart)
{
    SpringBody rope;

    double unitMass = mass / (2*(double)parts);
    double dx = (end[0] - start[0]) / (double)parts;
    double dy = (end[1] - start[1]) / (double)parts;

    double x = start[0];
    double y = start[1];
    
    double width = 0.0333;

    rope.addNode(unitMass, x - width/2, y, fixedStart); // 0
    rope.addNode(unitMass, x + width/2, y, fixedStart); // 1

    rope.joinNodes(stiffness, 0, 1);

    // 0 1 i = 0
    // 2 3 i = 1
    // 4 5 i = 2
    // 6 7 i = 3

    for (unsigned int i = 1; i < parts; ++i)
    { // i = 2
        x += dx; y += dy;
        rope.addNode(unitMass, x - width/2, y, false); // 4
        rope.addNode(unitMass, x + width/2, y, false); // 5
        rope.joinNodes(stiffness, 2*i, 2*i+1); // 2*i = 2*2 = 4, 2*i+1 = 2*2+1 = 5
        rope.joinNodes(stiffness, 2*i, 2*i-2); // 4, 2
        rope.joinNodes(stiffness, 2*i+1, 2*i-1); // 5, 3

        rope.joinNodes(stiffness, 2*i, 2*i-1); // 5, 3
        rope.joinNodes(stiffness, 2*i+1, 2*i-2); // 5, 2
    }
    
    return rope;
}

void applyUniformAcceleration(double a[2], SpringBody &body)
{
    double force[2];
    for (unsigned int i = 0; i < body.inodes.size(); ++i)
    {
        force[0] = a[0] * body.inodes[i]->getMass();
        force[1] = a[1] * body.inodes[i]->getMass();
        body.inodes[i]->addForce(force);
    }    
}
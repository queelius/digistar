#include "Events.h"

namespace PhysicsTest {

void Events::mouseButtonEvent(int button, int state, int x, int y) {
    static GLfloat i = 0.5;
    Point pt = cam.getPosition(x, y);

    if (state == GLUT_DOWN && button == GLUT_LEFT_BUTTON) {
        GLfloat color[] = {1, 1, 1};
        world.push_back(new Planetoid(pt, Vector(i, i)/i, 10, 10, 0, 0, color));
        i += 0.1;
    }
}

void Events::specialKeyboardEvent(int key, int x, int y) {
    switch (key) {
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
        case GLUT_KEY_RIGHT:
            break;
        case GLUT_KEY_UP:
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

void Events::keyboardEvent(unsigned char key, int x, int y) {
    switch (key) {
        default:
            break;
    }
}

void Events::timerEvent(int value) {
    static GLfloat i = 0.1;
    // perform updates
    for (auto i = world.begin(); i != world.end(); ++i) {
        //(*i)->applyForce(Vector(1, PI/2), (*i)->getPosition() + Vector((*i)->getRadius(), PI/2));
        (*i)->update();
    }
    checkCollisions();

    //GLfloat color[] = {1, 0, 1};
    //Point pt(0, 0);
    //world.push_back(new Planetoid(pt, Vector(i, i)/i, 2, 1, 0, 0, color));
    //i += 0.1;

    glutPostRedisplay();
    glutTimerFunc(10, timerEvent, 1);
}

void Events::displayEvent() {
	glClear(GL_COLOR_BUFFER_BIT);
    
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();

    //cam.print(Point(0, 0), "Hello, world!");

    glOrtho(cam.getLeft(), cam.getRight(),
            cam.getBottom(), cam.getTop(),
            -10.0, 10.0);

    glMatrixMode(GL_MODELVIEW);

    Vector ZeroMag(0, 0);
    glPushMatrix();
        for (std::list<Object*>::iterator i = world.begin(); i != world.end(); ++i) {
            //(*i)->netForce.draw((*i)->getPosition(), (*i)->getColor());
            (*i)->draw();
            (*i)->netForce = ZeroMag;
            (*i)->netTorque = 0;
        }

        // do draw routines
        // use faster display lists
    glPopMatrix();

	glutSwapBuffers();
	glFlush();
}

void Events::reshapeEvent(GLsizei w, GLsizei h) {
    glViewport(0, 0, w, h);
    cam.setPixelDimensions(w, h);
}

}
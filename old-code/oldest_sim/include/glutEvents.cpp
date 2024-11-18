#include "glutEvents.h"
#include "spaceship.h"
#include "asteroid.h"
#include "gravityWell.h"

void mouseEvent(int mouseButton, int mouseState, int mouseXPosition, int mouseYPosition) {
    GLfloat pos[2];

    canvas.getPosition(pos[0], pos[1], mouseXPosition, mouseYPosition);
    GLfloat initialVel[] = {1, 1};

    if (mouseState == GLUT_DOWN && mouseButton == GLUT_LEFT_BUTTON) {
        o.push_back(new GravityWell(pos, 10000000, initialVel, false, 10000/5));
    }
}

/*
    GLUT_KEY_F1		F1 function key
    GLUT_KEY_F2		F2 function key
    GLUT_KEY_F3		F3 function key
    GLUT_KEY_F4		F4 function key
    GLUT_KEY_F5		F5 function key
    GLUT_KEY_F6		F6 function key
    GLUT_KEY_F7		F7 function key
    GLUT_KEY_F8		F8 function key
    GLUT_KEY_F9		F9 function key
    GLUT_KEY_F10		F10 function key
    GLUT_KEY_F11		F11 function key
    GLUT_KEY_F12		F12 function key
    GLUT_KEY_LEFT		Left function key
    GLUT_KEY_RIGHT		Right function key
    GLUT_KEY_UP		Up function key
    GLUT_KEY_DOWN		Down function key
    GLUT_KEY_PAGE_UP	Page Up function key
    GLUT_KEY_PAGE_DOWN	Page Down function key
    GLUT_KEY_HOME		Home function key
    GLUT_KEY_END		End function key
    GLUT_KEY_INSERT		Insert function key
*/
void specialKeyEvent(int pressedKey, int mouseXPosition, int mouseYPosition) {
    switch (pressedKey) {
        case GLUT_KEY_DOWN:
            player->transForce(player->getAngle() + PI, 100);
            break;
        case GLUT_KEY_UP:
            player->transForce(player->getAngle(), 100);
            break;
        case GLUT_KEY_LEFT:
            player->torqueForce(200);
            break;
        case GLUT_KEY_RIGHT:
            player->torqueForce(-200);
            break;
    }
}

void keyEvent(unsigned char pressedKey, int mouseXPosition, int mouseYPosition) {
    switch (pressedKey) {
        case 'g':
        case 'G':
            ++selObject;
            if (selObject == o.end())
                selObject = o.begin();
            ((Spaceship*)player)->setTarget(*selObject);
            break;
        case ' ':
            ((Spaceship*)player)->shoot(50.0, 1.0, 25.0);
            break;
        case 'f':
        case 'F':
            ((Spaceship*)player)->shootMissile();
            break;
        case '+':
        case '=':
            canvas.scale(1.1);
            break;
        case '-':
        case '_':
            canvas.scale(0.9);
            break;
        case 'q':
        case 'Q':
        case 27:
            exit(1);
            break;
    }
}

void updateEvent(int value) {
    for (auto i = o.begin(); i != o.end(); ++i)
        (*i)->update();

    applyCollisions();

    glutPostRedisplay();
    glutTimerFunc(MSEC_PER_UPDATE, updateEvent, 1);
}

void refreshEvent() {
	glClear(GL_COLOR_BUFFER_BIT);
    
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();

    canvas.setCenter(player->getPosition());
    canvas.showInfo(player, White);
    glOrtho(canvas.getLeft(), canvas.getRight(),
            canvas.getBottom(), canvas.getTop(),
            -10.0, 10.0);
    glMatrixMode(GL_MODELVIEW);

    glPushMatrix();
        for (auto i = o.begin(); i != o.end(); ++i)
            (*i)->draw();
    glPopMatrix();

	glutSwapBuffers();
	glFlush();
}

void resizeEvent(GLsizei w, GLsizei h) {
    glViewport(0, 0, w, h);
    canvas.pixelWidth = w;
    canvas.pixelHeight = h;
}

void applyCollisions() {
    return;


    std::list<DelayedUpdate> u;

    for (auto i = o.begin(); i != o.end(); ++i) {
        const GLfloat *pt = (*i)->getPosition();
        const GLfloat radius = (*i)->getRadius();

        for (auto j = o.begin(); j != o.end(); ++j) {
            if (getDistance(pt, (*j)->getPosition()) <= radius + (*j)->getRadius() && i != j) {
                //GLfloat ma = (*i)->mass;
                //GLfloat mb = (*j)->mass;

                //GLfloat va_x = (*i)->vel[0];
                //GLfloat va_y = (*i)->vel[1];
                    
                //GLfloat vb_x = (*j)->vel[0];
                //GLfloat vb_y = (*j)->vel[1];

                GLfloat pNew[] = {
                    ((*i)->mass * ((*i)->mass * (*i)->vel[0]  + (*j)->mass * (*j)->vel[0]) / ((*i)->mass + (*j)->mass)),
                    ((*i)->mass * ((*i)->mass * (*i)->vel[1]  + (*j)->mass * (*j)->vel[1]) / ((*i)->mass + (*j)->mass))
                };

                //if ((*i)->what() == GravityWellType && (*j)->what() != AsteroidType) {
                    //first = false;
                    /*std::cout << "Collision: on Pi = " << (*i)->getMomentum() << std::endl;
                    std::cout << "i mass = " << (*i)->getMass() << std::endl;
                    std::cout << "i speed = " << (*i)->getSpeed() << endl;

                    std::cout << " Pj = " << (*j)->getMomentum() << std::endl;
                    std::cout << "j mass = " << (*j)->getMass() << endl;
                    std::cout << "j speed = " << (*j)->getSpeed() << endl;

                    GLfloat newVel = sqrt(vNew[0]*vNew[0] + vNew[1]*vNew[1]);
                    std::cout << "After collision: Pi = " << newVel * (*i)->mass << std::endl;
                    std::cout << "i speed = " << newVel << endl;
                    std::cout << " Pj = " << newVel * (*j)->mass << std::endl;
                    std::cout << "j speed = " << newVel << endl;
                    */

                    DelayedUpdate tmp = {*i, pNew};
                    //DelayedUpdate tmp2 = {*j, vNew};
                    //tmp.objectRef = *i;
                    //tmp.newVelocity[0] = v[0];
                    //tmp.newVelocity[1] = v[1];
                    u.push_back(tmp);
                    break;
                    //u.push_back(tmp2);
                //}
            }
        }
    }
    for (auto i = u.begin(); i != u.end(); ++i) {
        //cout << "???" << endl;
        //memcpy(i->objectRef->vel, i->newVelocity, 8);
        i->objectRef->vel[0] += i->newVelocity[0];
        i->objectRef->vel[1] += i->newVelocity[1];
    }
};

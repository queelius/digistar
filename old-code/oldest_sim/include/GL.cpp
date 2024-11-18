#include "GL.h"

namespace GL {
    GLWindow::GLWindow(GLint pixelWidth, GLint pixelHeight, GLfloat windowWidth,
                       GLfloat windowHeight, GLint pixelLeftX, GLint pixelTopY) {
        set(pixelWidth, pixelHeight, windowWidth, windowHeight, pixelLeftX, pixelTopY);
    }

    void GLWindow::getXY(GLfloat &x, GLfloat &y, GLint pixelX, GLint pixelY) {
        x = windowWidth * pixelX / pixelWidth - 0.5 * windowWidth;
        y = 0.5 * windowHeight - (windowHeight * pixelY / pixelHeight);
    }

    void GLWindow::init(const char title[]) {
        glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA);
        glutInitWindowPosition(pixelLeftX, pixelTopY);
        glutInitWindowSize(pixelWidth, pixelHeight);
        glutCreateWindow(title);
    }

    void GLWindow::set(GLint pixelWidth, GLint pixelHeight, GLfloat windowWidth,
                       GLfloat windowHeight, GLint pixelLeftX, GLint pixelTopY)
    {
        this->pixelWidth = pixelWidth;
        this->pixelHeight = pixelHeight;
        this->windowWidth = windowWidth;
        this->windowHeight = windowHeight;
        this->pixelLeftX = pixelLeftX;
        this->pixelTopY = pixelTopY;            
    }

    void GLWindow::scale(double value)
    {
        this->windowWidth *= value;
        this->windowHeight *= value;
    }
}
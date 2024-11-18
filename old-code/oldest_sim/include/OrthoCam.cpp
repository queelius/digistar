#include "OrthoCam.h"
#include "glut.h"

OrthoCam::OrthoCam()
{
    // do nothing
}

OrthoCam::OrthoCam(double width, double height,
        unsigned int pixelWidth, unsigned int pixelHeight,
        const Vector2D &centerPosition)
{
    setViewDimensions(width, height);
    setPixelDimensions(pixelWidth, pixelHeight);
    setCenterPosition(centerPosition);
}

void OrthoCam::setPixelDimensions(unsigned int pxWidth, unsigned int pxHeight)
{
    this->pxWidth = pxWidth;
    this->pxHeight = pxHeight;
}

void OrthoCam::setViewDimensions(double width, double height)
{
    this->width = width;
    this->height = height;
}

void OrthoCam::zoom(double factor)
{
    width *= factor;
    height *= factor;   
}

void OrthoCam::setCenterPosition(const Vector2D &centerPosition)
{
    this->centerPosition = centerPosition;
}

void OrthoCam::translate(double x, double y)
{
    centerPosition.getX() += x;
    centerPosition.getY() += y;
}

Vector2D OrthoCam::getPosition(unsigned int pixelX, unsigned int pixelY) const
{
    return Vector2D(
        centerPosition.getX() + width * pixelX / pxWidth - 0.5 * width,
        centerPosition.getY() + 0.5 * height - (height * pixelY / pxHeight)
        );
}

Vector2D OrthoCam::getCenterPosition() const
{
    return centerPosition;
}

double OrthoCam::getLeft() const
{
    return centerPosition.getX() - width / 2.0;
}

double OrthoCam::getRight() const
{
    return centerPosition.getX() + width / 2.0;
}

double OrthoCam::getTop() const
{
    return centerPosition.getY() + height / 2.0;
}

double OrthoCam::getBottom() const
{
    return centerPosition.getY() - height / 2.0;
}

double OrthoCam::getWidth() const
{
    return width;
}

double OrthoCam::getHeight() const
{
    return height;
}

unsigned int OrthoCam::getPixelWidth() const
{
    return pxWidth;
}

unsigned int OrthoCam::getPixelHeight() const
{
    return pxHeight;
}

void OrthoCam::reshape(int width, int height)
{
    setPixelDimensions(width, height);

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();

    if (width <= height)
		setViewDimensions(getWidth(), getHeight() * (double)height / (double)width);
    else
        setViewDimensions(getWidth() * (double)width / (double)height, getHeight());

    glOrtho(getLeft(), getRight(), getBottom(), getTop(), -10.0, 10.0);
    glMatrixMode(GL_MODELVIEW);
}
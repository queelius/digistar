#ifndef ORTHO_CAM_H
#define ORTHO_CAM_H

#include "GlobalConstants.h"
#include "Vector2D.h"

class OrthoCam {
public:
                    OrthoCam();

                    OrthoCam(
                        double          width,
                        double          height,
                        unsigned int    pixelWidth,
                        unsigned int    pixelHeight,
                        const Vector2D  &centerPosition
                    );

    void            setPixelDimensions(
                        unsigned int    pixelWidth,
                        unsigned int    pixelHeight
                    );

    void            setViewDimensions(
                        double          width,
                        double          height
                    );

    void            setCenterPosition(
                        const Vector2D  &centerPosition
                    );

    void            translate(
                        double x,
                        double y
                    );

    void            zoom(
                        double factor
                    );

    Vector2D        getPosition(
                        unsigned int pixelX,
                        unsigned int pixelY
                    )                                           const;

    Vector2D        getCenterPosition()                         const;

    double          getLeft()                                   const;
    double          getRight()                                  const;
    double          getTop()                                    const;
    double          getBottom()                                 const;

    double          getWidth()                                  const;
    double          getHeight()                                 const;
    unsigned int    getPixelWidth()                             const;
    unsigned int    getPixelHeight()                            const;

    void            reshape(
                        int width,
                        int height
                    );

protected:
    unsigned int    pxWidth;
    unsigned int    pxHeight;
    double          width;
    double          height;
    Vector2D        centerPosition;
};

#endif
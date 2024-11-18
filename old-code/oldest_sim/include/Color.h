#ifndef COLOR_H
#define COLOR_H

#include "glut.h"
#include "GL_Constants.h"
#include "Exceptions.h"
#include "vec3.h"

class Color
{
public:
    static struct Names
    {
        const static Color White;
        const static Color Black;

        const static Color Red;
        const static Color Green;
        const static Color Blue;

        const static Color Yellow;
        const static Color Magenta;
        const static Color Cyan;

        const static Color Orange;
        const static Color Gold;
        const static Color Pink;
        const static Color Gray;
        const static Color Violet;
        const static Color Brown;
        const static Color Silver;
        const static Color Maroon;
        const static Color Olive;
        const static Color Lime;
        const static Color Aqua;
        const static Color Teal;
        const static Color Navy;
        const static Color Fuchsia;
        const static Color Purple;
    };

    Color(float r, float g, float b);
    Color(const float rgb[]);

    void toHSV(float& hue, float& saturation, float& value)
    {
	    float minimum = min(rgb[0], min(rgb[1], rgb[2]));
	    float maximum = max(rgb[0], max(rgb[1], rgb[2]));
	    float delta = maximum - minimum;

	    // r = g = b = 0, saturation = 0, v is undefined
        if (maximum == 0.0f)
        {		    
		    saturation = 0;
		    hue = -1;
	    }
        else
        {
            saturation = delta / maximum;

	        if (rgb[0] == maximum)  // between yellow & magenta
		        hue = (rgb[1] - rgb[2]) / delta;
	        else if(rgb[1] == maximum)     // between cyan & yellow
		        hue = 2.0f + (rgb[2] - rgb[0]) / delta;
	        else                    // between magenta & cyan
		        hue = 4.0f + (rgb[0] - rgb[1]) / delta;

	        hue *= 60.0f; // degrees
	        if (hue < 0.0f)
		        hue += 360.0f;
        }
    };

    void adjustSaturation(float amount)
    {
        float hue, saturation, value;

        toHSV(hue, saturation, value);
        float newSaturation = saturation + amount;
        if (newSaturation < 0.0f)
            newSaturation = 0.0f;
        else if (newSaturation > 1.0f)
            newSaturation = 1.0f;

        *this = getFromHSV(hue, newSaturation, value);
    };

    static Color getFromRGB(float red, float green, float blue)
    {
        return Color(red, green, blue);
    };

    static Color getFromHSV(float hue, float saturation, float value)
    {
	    if (saturation == 0.0f) // achromatic (grey)
		    return Color(value, value, value);

	    hue /= 60.0f; // sector 0 to 5
	    int i = (int)hue;
	    float f = hue - i; // factorial part of hue
	    float p = value * (1 - saturation);
	    float q = value * (1 - saturation * f);
	    float t = value * (1 - saturation * (1 - f));
	    switch (i)
        {
		    case 0: return Color(value, t, p);
		    case 1: return Color(q, value, p);
		    case 2: return Color(p, value, t);
		    case 3: return Color(p, q, value);
		    case 4: return Color(t, p, value);
            case 5: return Color(value, p, q);
            default: throw InvalidArgument();
	    }
    };

    void set(float r, float g, float b);
    void set(const float rgb[]);

    void mixWith(const Color& color);

    Color complement() const
    {
        throw UnsupportedOperation();
    };

    const float* toArray() const;

protected:
    vec3 rgb;
};

const Color Color::Names::Green = Color(0.0f, 1.0f, 0.0f);
const Color Color::Names::Black = Color(0.0f, 0.0f, 0.0f);
const Color Color::Names::White = Color(1.0f, 1.0f, 1.0f);
const Color Color::Names::Blue = Color(0.0f, 0.0f, 1.0f);
const Color Color::Names::Red = Color(1.0f, 0.0f, 0.0f);

#endif
#include "Utilities.h"

double getAngle(double dx, double dy)
{
    double theta;
    if (dx == 0.0)
    {
        if (dy > 0.0)
            theta = PI_OVER_2;
        else if (dy < 0.0)
            theta = 1.5 * PI;
        else
            throw Exception(INPUT_ERROR, "Undefined angle");
    }
    else
    {
        theta = atan(dy / dx);
        if (dx > 0.0 && dy < 0.0)
            theta += TWO_PI;
        else if (dx < 0.0)
            theta += PI;
    }

    return theta;
}

double radians_to_degrees(double radians)
{
    return radians * ONE_HUNDRED_EIGHTY_OVER_PI;
}

double normalize_radians(double radians)
{
    radians -= TWO_PI * (int)(radians / TWO_PI);
    if (radians < 0.0)
        radians += TWO_PI;

    return radians;
}

double normalize_degrees(double degrees)
{
    degrees -= 360 * (int)(degrees / 360);
    if (degrees < 0.0)
        degrees += 360;

    return degrees;
}

double degrees_to_radians(double degrees)
{
    return degrees * PI_OVER_180; 
}

double km_to_au(double km) {
    return km / KILOMETERS_PER_AU;
};

double m_to_km(double m) {
    return m / METERS_PER_KILOMETER;
};

double km_to_m(double km) {
    return km * METERS_PER_KILOMETER;
};

double rads_to_degrees(double rads) {
    return rads * 180 / PI;
};

double solar_mass_to_kg(double solar_masses)
{
    return solar_masses * KILOGRAMS_PER_SOLAR_MASS;
};

double earth_mass_to_kg(double earth_masses)
{
    return earth_masses * KILOGRAMS_PER_EARTH_MASS;
}

double kg_to_earth_mass(double kg)
{
    return kg / KILOGRAMS_PER_EARTH_MASS;
}

double kg_to_solar_mass(double kg)
{
    return kg / KILOGRAMS_PER_SOLAR_MASS;
}

double kilometers_to_au(double km)
{
    return km / KILOMETERS_PER_AU;
}

double au_to_km(double au)
{
    return au * KILOMETERS_PER_AU;
}

double meters_to_km(double m)
{
    return m / METERS_PER_KILOMETER;
}

double km_to_meters(double km) {
    return km * METERS_PER_KILOMETER;
}

double solar_masses_to_km(double solar_masses)
{
    return solar_masses * KILOGRAMS_PER_SOLAR_MASS;
}

double standard_weight_to_mass(double weight)
{
    return weight / STANDARD_GRAVITY;
}

double light_years_to_km(double light_years)
{
    return light_years * 9460730472580.8e+12;
}

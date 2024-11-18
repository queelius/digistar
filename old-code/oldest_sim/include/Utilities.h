#ifndef UTILITIES_H
#define UTILITIES_H

#include <cmath>
#include "GlobalConstants.h"
#include "Exception.h"

// get angle difference in radians [0 and 2*pi) of a vector
// defined by the difference of two points, v = point 2 - point 1
// = (dx, dy), where x is x-component difference and y is
// y-component difference.
//
// note: replaced by atan2 functionality
double getAngle(double dx, double dy);

double radians_to_degrees(double radians);      // returns (radians * 180 / pi)
double normalize_radians(double radian);        // convert radian to equivalent radian in the range of [0, 2*pi)
double normalize_degrees(double degree);        // convert degree to equivalent degree in the range of [0, 360)
double degrees_to_radians(double degrees);      // returns (degrees * pi / 180)
double kelvins_to_celsius(double kelvins);
double celsius_to_kelvins(double celsius);
double km_to_au(double km);                     // convert kilometers to astronomical units (distance from the Earth to the Sun)
double au_to_km(double au);                     // convert astronomical units (distance from the Earth to the Sun) to kilometers
double meters_to_km(double m);                  // convert meters to kilometers
double km_to_meters(double km);                 // convert kilometers to meters
double solar_masses_to_kg(double solar_masses); // convert solar masses (mass of the Sun) to kilograms
double earth_masses_to_kg(double earth_masses); // convert earth masses (mass of the Earth) to kilograms
double kg_to_earth_masses(double kg);           // convert kilograms to earth masses (mass of the Earth)
double kg_to_solar_masses(double kg);           // convert kilograms to solar masses (mass of the Sun)
double standard_weight_to_kg(double weight);    // convert standard weight (how much force is applied to a mass when on the surface of Earth) to kilograms
double light_years_to_km(double light_years);   // convert light years (how far light travels in a year) to kilometers

#endif
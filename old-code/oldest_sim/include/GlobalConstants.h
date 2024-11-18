#ifndef GLOBAL_CONSTANTS_H
#define GLOBAL_CONSTANTS_H

const char WINDOW_BAR_TITLE[]               = "2D Physics";

// color constants
const double COLOR_WHITE[]                  = { 1.0, 1.0, 1.0 };
const double COLOR_GRAY[]                   = { 0.5, 0.5, 0.5 };
const double COLOR_BLACK[]                  = { 0.0, 0.0, 0.0 };
const double COLOR_RED[]                    = { 1.0, 0.0, 0.0 };
const double COLOR_GREEN[]                  = { 0.0, 1.0, 0.0 };
const double COLOR_BLUE[]                   = { 0.0, 0.0, 1.0 };
const double COLOR_YELLOW[]                 = { 1.0, 1.0, 0.0 };
const double COLOR_CYAN[]                   = { 0.0, 1.0, 1.0 };
const double COLOR_MAGENTA[]                = { 1.0, 0.0, 1.0 };

// mathematical / physical constants
const double PI                             = 3.14159265;
const double TWO_PI                         = 2.0 * PI;
const double PI_OVER_2                      = PI / 2.0;
const double PI_OVER_180                    = PI / 180.0;
const double ONE_HUNDRED_EIGHTY_OVER_PI     = 180 / PI;
const double E                              = 2.71828183;

const double GRAMS_PER_KILOGRAM             = 1000;
const double ELECTRON_MASS                  = 9.10938215E-31;                       // kg
const double PROTON_MASS                    = 1.67262158E-27;                       // kg
const double KILOGRAMS_PER_EARTH_MASS       = 5.9742E+24;                           // kg
const double KILOGRAMS_PER_SOLAR_MASS       = 333000 * KILOGRAMS_PER_EARTH_MASS;    // kg
const double KILOGRAMS_PER_MERCURY_MASS     = 0.055 * KILOGRAMS_PER_EARTH_MASS;     // kg

const double MILLIMETERS_PER_CENTIMETER     = 10;
const double CENTIMETERS_PER_METER          = 100;
const double METERS_PER_KILOMETER           = 1000;
const double KILOMETERS_PER_AU              = 149597871;

const double NANOSECONDS_PER_MICROSECOND    = 1000;
const double MICROSECONDS_PER_MILLISECOND   = 1000;
const double MILLISECONDS_PER_SECOND        = 1000;
const double SECONDS_PER_MINUTE             = 60;
const double MINUTES_PER_HOUR               = 60;
const double HOURS_PER_DAY                  = 24;
const double DAYS_PER_YEAR                  = 365.242199;

const double METERS_PER_AU                  = KILOMETERS_PER_AU * METERS_PER_KILOMETER;
const double MICROSECONDS_PER_SECOND        = MICROSECONDS_PER_MILLISECOND * MILLISECONDS_PER_SECOND;
const double NANOSECONDS_PER_SECOND         = NANOSECONDS_PER_MICROSECOND * MICROSECONDS_PER_SECOND;
const double SECONDS_PER_HOUR               = SECONDS_PER_MINUTE * MINUTES_PER_HOUR;
const double MINUTES_PER_DAY                = MINUTES_PER_HOUR * HOURS_PER_DAY;
const double SECONDS_PER_DAY                = MINUTES_PER_DAY * SECONDS_PER_MINUTE;
const double SECONDS_PER_YEAR               = SECONDS_PER_DAY * DAYS_PER_YEAR;
const double MINUTES_PER_YEAR               = MINUTES_PER_DAY * DAYS_PER_YEAR;
const double HOURS_PER_YEAR                 = HOURS_PER_DAY * DAYS_PER_YEAR;

const double LIGHT_SECOND                   = 0.00200398707 * METERS_PER_AU;        // m
const double LIGHT_MINUTE                   = SECONDS_PER_MINUTE * LIGHT_SECOND;    // m
const double LIGHT_HOUR                     = MINUTES_PER_HOUR * LIGHT_MINUTE;      // m
const double LIGHT_DAY                      = HOURS_PER_DAY * LIGHT_HOUR;           // m
const double LIGHT_MONTH                    = 30 * LIGHT_DAY;                       // m
const double LIGHT_YEAR                     = DAYS_PER_YEAR * LIGHT_DAY;            // m
const double PARSEC                         = 206.26E+3 * METERS_PER_AU;            // m

const double SPEED_OF_LIGHT                 = LIGHT_SECOND;                         // m/s
const double ABSOLUTE_MAX_SPEED             = LIGHT_SECOND;                         // m/s
const double STANDARD_GRAVITY               = 9.8006;                               // m/s^2
const double GRAVITATIONAL_CONSTANT         = 6.6732E-11;                           // m^3/kg/s^2
const double COULOMB_CONSTANT               = 8.987551787E+9;                       // N/m^2*C^-2
const double ELEMENTARY_CHARGE              = 1.602176487E-19;                      // C

const double AVOGADRO_CONSTANT              = 6.02214179E+23;                       // mol^-1
const double FARADAY_CONSTANT               = ELEMENTARY_CHARGE * AVOGADRO_CONSTANT;

const double KILOMETERS_PER_EARTH_RADIUS    = 6371;                                 // km
const double METERS_PER_EARTH_RADIUS        = KILOMETERS_PER_EARTH_RADIUS * METERS_PER_KILOMETER;   // m
const double METERS_PER_SOLAR_RADIUS        = 109 * METERS_PER_EARTH_RADIUS;        // m
const double METERS_PER_MERCURY_RADIUS      = 0.3829 * METERS_PER_EARTH_RADIUS;     // m

#endif
#ifndef RANDOM_H
#define RANDOM_H

#include <ctime>

// Random number class.
class Random {
public:
    // constructor
    Random(int c = 1, int a = 16389, int m = 1073741824, int X = 0);

    // get a random number between low and high, inclusive
    double get(double low, double high);

    // get a random number between 0 and 1, inclusive
    double get0_1();

protected:
    int X, a, c, m;

    // get a seed (for random number generator); based on system clock
    unsigned int seed() const;
};

#endif
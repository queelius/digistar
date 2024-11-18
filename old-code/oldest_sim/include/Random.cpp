#include "Random.h"

Random::Random(int c, int a, int m, int X): c(c), a(a), m(m) {
    this->X = (!X ? seed() : X);
}

double Random::get(double low, double high) {
    return low + (high - low) * get0_1();
}

double Random::get0_1() {
    X = (a * X + c) % m;
    if (X < 0)
        X = -X;
    return (double)X / m;
}

unsigned int Random::seed() const {
    time_t now;
    time(&now);
    return (unsigned int)now;
}
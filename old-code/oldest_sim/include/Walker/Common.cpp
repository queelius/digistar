#include "Common.h"

// Copy source (src) into destination (dest)
// size specifies how many elements to copy from src into dest
template <class T> inline void cp(T *dest, const T *src, size_t size) {
    for (size_t i = 0; i < size; ++i)
        dest[i] = src[i];
}

void seed() {
    srand((unsigned)time(0));
}

GLint getRandInt(GLint low, GLint high) {
    return low + rand() % (high - low + 1);
}

GLfloat getRand(GLfloat low, GLfloat high) {
    return low + (high - low) * rand() / RAND_MAX;
}

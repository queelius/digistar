////////////////////////////////////////////////////
//    Filename: utils.h                           //
//        Name: Alex Towell                       //
//    Due Date: 11-3-2009                         //
//      Course: CS 482 (Computer Graphics)        //
// Description: Two fairly weak random number     //
// generators. But they are sufficient for this   //
// program's needs, with the important benefit of //
// being fast.                                    //
////////////////////////////////////////////////////

#include <ctime>
#include <cstdlib>

inline void seed() {
    srand((unsigned)time(0));
}

inline GLint getRandInt(GLint low, GLint high) {
    return low + rand() % (high - low + 1);
};

inline GLfloat getRand(GLfloat low, GLfloat high) {
    return low + (high - low) * rand() / RAND_MAX;
};

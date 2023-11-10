//
// Created by Jiang, Peng on 11/10/23.
//

#ifndef UTIL_H
#define UTIL_H

#include <math.h>
#include <stdlib.h>

template <class T=int>
int nbits(T n) {
    for (int i = 30; i>=0; i--) {
        if (n & (1 << i)) {
            return i + 1;
        }
    }
    return 0;
}


#endif //UTIL_H

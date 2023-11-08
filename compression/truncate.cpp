#include <iostream>

int main() {
    int DATA[32];
    int NBITS = 16;

    unsigned m = 0x0000FFFF;
    for (int j = 16; j != 0; j = j >> 1, m = m ^ (m << j)) {
        for (int k = 0; k < 32; k = (k + j + 1) & ~j) {
            unsigned t = (DATA[k] ^ (DATA[k+j] >> j)) & m;
            DATA[k] = DATA[k] ^ t;
            DATA[k+j] = DATA[k+j] ^ (t << j);
        }
    }

    for (int i=0; i < NBITS; i++) {
        std::cout << DATA[i] << std::endl;
    }

    return 0;
}
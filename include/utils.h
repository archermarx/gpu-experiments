#ifndef UTILS_H
#define UTILS_H

#include <cstdint>

#ifdef _WIN32
#include <Windows.h>
#else
#include <unistd.h>
#endif

// delay computation by a provided number of milliseconds
void delay (uint32_t ms);

// wrap a number a modulo b
template <typename T>
__host__ __device__
T wrap (T a, T b) {
    return (b + (a % b)) % b;
}

inline __host__ __device__
int wrapIndex2D(int i, int j, int nx, int ny) {
    return wrap(i, nx) + nx * wrap(j, ny);
}

#endif
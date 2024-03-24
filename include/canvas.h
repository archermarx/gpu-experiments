#ifndef CANVAS_H
#define CANVAS_H

#include <cstdint>
#include <vector>

class Canvas {
    public:
        uint32_t width;
        uint32_t height;
        std::vector<uint8_t> contents;

        Canvas(uint32_t w, uint32_t h);
        void fill(uint8_t r, uint8_t g, uint8_t b);
};

#endif


#ifndef COLOR_H
#define COLOR_H

#include <cstdint>

class Color{
    public:
        uint8_t r, g, b;
        Color(uint8_t _r, uint8_t _g, uint8_t _b): r(_r), g(_g), b(_b) {}
};

const Color BLACK(0, 0, 0);
const Color WHITE(255, 255, 255);
const Color RED(255, 0, 0);
const Color GREEN(0, 255, 0);
const Color BLUE(0, 0, 255);
const Color YELLOW(255, 255, 0);
const Color MAGENTA(255, 0, 255);
const Color CYAN(0, 255, 255);

#endif
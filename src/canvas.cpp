#include "canvas.h"

Canvas::Canvas(uint32_t w, uint32_t h): width(w), height(h), contents(w * h * 4) {}
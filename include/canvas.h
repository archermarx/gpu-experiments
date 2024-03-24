#ifndef CANVAS_H
#define CANVAS_H

#include <cstdint>
#include <vector>

#include "shader.h"

const float
SCREEN_QUAD_VERTS[] = {
    // positions          // texture coords
     1.0f,  1.0f, 1.0f,   1.0f, 1.0f,   // top right
     1.0f, -1.0f, 1.0f,   1.0f, 0.0f,   // bottom right
    -1.0f, -1.0f, 1.0f,   0.0f, 0.0f,   // bottom left
    -1.0f,  1.0f, 1.0f,   0.0f, 1.0f    // top left
};

const unsigned int
SCREEN_QUAD_ELEMS[] = {
    0, 1, 3,
    1, 2, 3
};

class Canvas {
    public:
        uint32_t width;
        uint32_t height;
        std::vector<uint8_t> contents;

        Canvas(uint32_t w, uint32_t h);
        void render();

    private:
        Shader shader;
        unsigned int texture;
        unsigned int VAO, VBO, EBO;
};

#endif


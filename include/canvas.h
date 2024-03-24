#ifndef CANVAS_H
#define CANVAS_H

#include <cstdint>
#include <vector>

#include "shader.h"

// set up vertex data (and buffer(s)) and configure vertex attributes
const float SCREEN_QUAD_VERTS[] = {
    // positions          // colors           // texture coords
        1.0f,  1.0f, 1.0f,   1.0f, 0.0f, 0.0f,     1.0f, 1.0f,   // top right
        1.0f, -1.0f, 1.0f,   0.0f, 1.0f, 0.0f,     1.0f, 0.0f,   // bottom right
        -1.0f, -1.0f, 1.0f,   0.0f, 0.0f, 1.0f,     0.0f, 0.0f,   // bottom left
        -1.0f,  1.0f, 1.0f,   1.0f, 1.0f, 0.0f,     0.0f, 1.0f    // top left
};

const unsigned int SCREEN_QUAD_ELEMS[] = {
    // note that we start from 0!
    0, 1, 3,   // first triangle
    1, 2, 3
};

class Canvas {
    public:
        uint32_t width;
        uint32_t height;
        std::vector<uint8_t> contents;

        Shader shader;
        unsigned int texture;

        // vertex and element buffers
        unsigned int VAO, VBO, EBO;

        Canvas(uint32_t w, uint32_t h);
        void fill(uint8_t r, uint8_t g, uint8_t b);
        void render();
};

#endif


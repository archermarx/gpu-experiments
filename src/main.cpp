// test.c
#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <iostream>
#include <string>
#include <vector>
#include <math.h>
#include <utility>

#include "window.h"
#include "utils.h"
#include "langton_ant.h"
#include "canvas.h"
#include "shader.h"

// settings
const unsigned int NUM_PIXELS_X = 100;
const unsigned int NUM_PIXELS_Y = 100;

const unsigned int SCR_WIDTH = 1000;
const unsigned int SCR_HEIGHT = 1000;

int main(void){

    Window window("OpenGL", SCR_WIDTH, SCR_HEIGHT);

    if (!window.open){
        return -1;
    }

    Canvas canvas(NUM_PIXELS_X, NUM_PIXELS_Y);

    LangtonAnt ant(
        NUM_PIXELS_X, NUM_PIXELS_Y,
        std::make_pair(NUM_PIXELS_X / 2, NUM_PIXELS_Y / 2),
        0
    );

    bool window_open = true;

    // Render loop
    while(window.open) {
        // update the ant state
        ant.update();
        ant.draw(canvas);

        // render the canvas
        canvas.render();

        window.checkForUpdates();
    }

    return 0;
}


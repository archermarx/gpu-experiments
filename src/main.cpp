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
#include "canvas.h"
#include "shader.h"
#include "langton_ant.h"
#include "game_of_life.h"

// process all input: query GLFW whether relevant keys are pressed/released this frame and react accordingly
// ---------------------------------------------------------------------------------------------------------
void processInput(Window window){
    if (glfwGetKey(window.window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
        glfwSetWindowShouldClose(window.window, true);
}

int main(void){

    unsigned int width = 1000, height = 1000;
    unsigned int pixelWidth = 128, pixelHeight = 128;

    // Create a window and check that it worked successfully
    Window window("OpenGL", width, height);

    if (!window.open){
        return -1;
    }

    // Create the canvas of pixels
    Canvas canvas(pixelWidth, pixelHeight);

    // Create our ant and associated state
    // LangtonAnt ant(
    //     pixelWidth, pixelHeight,
    //     std::make_pair(0.8f, 0.2f),
    //     0,
    //     "RRLLLRLLLRRR",
    //     std::vector<Color>({
    //         BLACK, CYAN, MAGENTA,
    //         YELLOW, RED, GREEN,
    //         BLUE, WHITE, Color(127, 127, 127),
    //         Color(127, 0, 0), Color(0, 127, 0), Color(0, 0, 127)
    //     })
    // );

    GameOfLife life(pixelWidth, pixelHeight);

    int i = pixelWidth / 2;
    int j = pixelHeight / 2;
    life.set(i,j, true);
    life.set(i-1,j,true);
    life.set(i,j-1,true);
    life.set(i,j+1, true);
    life.set(i+1,j+1, true);

    // Render loop
    while(window.open) {
        // Check for user input
        processInput(window);

        // draw state
        life.draw(canvas);
        canvas.render();

        // update state
        life.update();

        // Check for updates
        window.checkForUpdates();
    }

    return 0;
}


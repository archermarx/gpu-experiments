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

// process all input: query GLFW whether relevant keys are pressed/released this frame and react accordingly
// ---------------------------------------------------------------------------------------------------------
void processInput(Window window){
    if (glfwGetKey(window.window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
        glfwSetWindowShouldClose(window.window, true);
}

int main(void){

    unsigned int width = 1000, height = 1000;
    unsigned int pixelWidth = 100, pixelHeight = 100;

    // Create a window and check that it worked successfully
    Window window("OpenGL", width, height);

    if (!window.open){
        return -1;
    }

    // Create the canvas of pixels
    Canvas canvas(pixelWidth, pixelHeight);

    // Create our ant and associated state
    LangtonAnt ant(
        pixelWidth, pixelHeight,
        std::make_pair(0.5f, 0.5f),
        0
    );

    // Render loop
    while(window.open) {
        // Check for user input
        processInput(window);

        // update the ant state
        ant.update();
        ant.draw(canvas);

        // render the canvas
        canvas.render();

        // Check for updates
        window.checkForUpdates();
    }

    return 0;
}


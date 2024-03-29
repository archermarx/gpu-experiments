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
void processInput(Window& window){
    if (glfwGetKey(window.window, GLFW_KEY_ESCAPE) == GLFW_PRESS) {
        glfwSetWindowShouldClose(window.window, true);
    }
}

int program() {
    unsigned int width = 1024, height = 1024;
    unsigned int pixelWidth = 128, pixelHeight = 128;

    // Create a window and check that it worked successfully
    Window window("OpenGL", width, height);

    if (!window.open){
        return -1;
    }

    // Create the canvas of pixels
    Canvas canvas(pixelWidth, pixelHeight);

    //Create our ant and associated state
    // LangtonAnt automaton(
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

    GameOfLife automaton(pixelWidth, pixelHeight);

    int i = pixelWidth / 2;
    int j = pixelHeight / 2;
    automaton.set(i,j, true);
    automaton.set(i-1,j,true);
    automaton.set(i,j-1,true);
    automaton.set(i,j+1, true);
    automaton.set(i+1,j+1, true);

    int ticks = 0;
    int outputTimeInterval_ms = 100;
    float nextOutputTime;
    float elapsedTime;
    float totalTime = 0;

    // create timing instrumentation
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Render loop
    while(window.open) {
        // Check for user input
        processInput(window);

        // record iteration start time
        cudaEventRecord(start, 0);

        // draw state
        automaton.draw(canvas);
        canvas.render();

        // update state
        automaton.update();

        // record iteration stop time
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&elapsedTime, start, stop);
        totalTime += elapsedTime;
        ticks += 1;

        if (totalTime > nextOutputTime) {
            printf("Avg. frame time: %3.1f ms\n", totalTime / (float) ticks);
            nextOutputTime += outputTimeInterval_ms;
        }

        // Check for updates
        window.checkForUpdates();
    }

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}

int main(void){
    int retcode = program();
    std::cout << "Program successfully terminated." << std::endl;
    exit(retcode);
}


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
#include "reaction_diffusion.h"

// process all input: query GLFW whether relevant keys are pressed/released this frame and react accordingly
// ---------------------------------------------------------------------------------------------------------
void processInput(Window& window){
    if (glfwGetKey(window.window, GLFW_KEY_ESCAPE) == GLFW_PRESS) {
        glfwSetWindowShouldClose(window.window, true);
    }
}

int program() {
    unsigned int width = 1024, height = 1024;
    unsigned int pixelWidth = 256, pixelHeight = 256;

    // Create a window and check that it worked successfully
    Window window("OpenGL", width, height);

    if (!window.open){
        return -1;
    }

    // Create the canvas of pixels
    Canvas canvas(pixelWidth, pixelHeight);

    // Create our ant and associated state
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

    // GameOfLife automaton(pixelWidth, pixelHeight);

    // int i = pixelWidth / 2;
    // int j = pixelHeight / 2;

    // automaton.set(i,j-1, 1);
    // automaton.set(i,j, 1);
    // automaton.set(i,j+1, 1);
    // automaton.set(i+2,j, 1);
    // automaton.set(i+2,j+1, 1);
    // automaton.set(i+2,j+2, 1);
    // automaton.set(i+3,j+1, 1);
    // automaton.set(i-2,j-2, 1);
    // automaton.set(i-2,j-3, 1);
    // automaton.set(i-4,j-3, 1);

    float dt = 1;
    float du2 = 2e-5;
    float dv2 = 1e-5;
    float f = 0.034;
    float k = 0.058;

    ReactionDiffusion automaton(
        pixelWidth, pixelHeight,
        dt, du2, dv2, k, f
    );

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
            printf("Tick %d, avg. frame time: %3.1f ms\n", ticks, totalTime / (float) ticks);
            nextOutputTime += outputTimeInterval_ms;
        }

        // Check for updates
        window.checkForUpdates();

        //delay(1000);
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


// test.c
#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <iostream>
#include <string>
#include <vector>
#include <math.h>
#include <utility>

#include "utils.h"
#include "langton_ant.h"
#include "canvas.h"
#include "shader.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

// declarations
void framebuffer_size_callback(GLFWwindow* window, int width, int height);
void processInput(GLFWwindow *window);

// settings
const unsigned int NUM_PIXELS_X = 100;
const unsigned int NUM_PIXELS_Y = 100;

const unsigned int SCR_WIDTH = 1000;
const unsigned int SCR_HEIGHT = 1000;

const int SLEEP_INTERVAL = 0;

int main(void)
{
    glfwInit();
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    // Create a window and verify that it worked.
    GLFWwindow* window = glfwCreateWindow(SCR_WIDTH, SCR_HEIGHT, "LearnOpenGL", NULL, NULL);
    if (window == NULL) {
        std::cout << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        return -1;
    }

    // Make the context of our window the main context on the current thread
    glfwMakeContextCurrent(window);
    // Window resizing callback
    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);

    // Check that GLAD is loaded properly
    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
        std::cout << "Failed to initialize GLAD" << std::endl;
        glfwTerminate();
        return -1;
    }

    Canvas canvas(NUM_PIXELS_X, NUM_PIXELS_Y);

    LangtonAnt ant(
        NUM_PIXELS_X, NUM_PIXELS_Y,
        std::make_pair(NUM_PIXELS_X / 2, NUM_PIXELS_Y / 2),
        0
    );

    // Render loop!
    while(!glfwWindowShouldClose(window)) {
        // input
        processInput(window);

        // update the texture
        ant.update();
        ant.draw(canvas);

        // Activate the shader
        canvas.render();

        // check and call events and swap the buffers
        glfwSwapBuffers(window);
        glfwPollEvents();

        delay(SLEEP_INTERVAL);
    }

    glfwTerminate();

    return 0;
}

// process all input: query GLFW whether relevant keys are pressed/released this frame and react accordingly
// ---------------------------------------------------------------------------------------------------------
void processInput(GLFWwindow *window)
{
    if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
        glfwSetWindowShouldClose(window, true);
}

// glfw: whenever the window size changed (by OS or user resize) this callback function executes
// ---------------------------------------------------------------------------------------------
void framebuffer_size_callback(GLFWwindow* window, int width, int height)
{
    // make sure the viewport matches the new window dimensions; note that width and
    // height will be significantly larger than specified on retina displays.
    glViewport(0, 0, width, height);
}
#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include <cstdint>
#include <string>
#include <iostream>

#include "window.h"

Window::Window(std::string windowName, uint32_t w, uint32_t h)
    : name(windowName),
      width(w),
      height(h)
    {

    glfwInit();
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, OPENGL_MAJOR_VERSION);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, OPENGL_MINOR_VERSION);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    // Create a window and verify that it worked.
    window = glfwCreateWindow(width, height, windowName.c_str(), NULL, NULL);
    if (window == NULL) {
        std::cout << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        open = false;
        return;
    }

    // Make the context of our window the main context on the current thread
    glfwMakeContextCurrent(window);
    // Window resizing callback
    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);

    // Check that GLAD is loaded properly
    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
        std::cout << "Failed to initialize GLAD" << std::endl;
        glfwTerminate();
        open = false;
    }

    open = true;
};

void Window::checkForUpdates() {
    // check and call events and swap the buffers
    glfwSwapBuffers(window);
    glfwPollEvents();

    if (glfwWindowShouldClose(window)) {
        open = false;
        glfwTerminate();
    } else {
        open = true;
    }
}
// glfw: whenever the window size changed (by OS or user resize) this callback function executes
// ---------------------------------------------------------------------------------------------
void framebuffer_size_callback(GLFWwindow* window, int width, int height)
{
    // make sure the viewport matches the new window dimensions; note that width and
    // height will be significantly larger than specified on retina displays.
    glViewport(0, 0, width, height);
}
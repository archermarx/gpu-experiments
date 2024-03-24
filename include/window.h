#ifndef WINDOW_H
#define WINDOW_H

#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include <cstdint>
#include <string>
#include <bits/stdc++.h>

const int OPENGL_MAJOR_VERSION = 3;
const int OPENGL_MINOR_VERSION = 3;

void framebuffer_size_callback(GLFWwindow* window, int width, int height);

class Window {
    public:
        std::string name;
        uint32_t width;
        uint32_t height;
        bool open;
        GLFWwindow* window;

        Window(std::string windowName, uint32_t w, uint32_t h);
        void checkForUpdates();

    private:
};

#endif
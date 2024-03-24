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

void fillCanvas(std::vector<GLubyte>& canvas, int nx, int ny, int r, int g, int b) {
    int pixelIndex = 0;
    for (int y = 0; y < nx; y++) {
        for (int x = 0; x < ny; x++) {
            canvas[pixelIndex] = r;
            canvas[pixelIndex+1] = g;
            canvas[pixelIndex+2] = b;
            canvas[pixelIndex+3] = 255;
            pixelIndex += 4;
        }
    }
}

std::vector<GLubyte> generateCanvas(int nx, int ny, int r, int g, int b) {
    int num_pixels = nx * ny;
    std::vector<GLubyte> canvas(num_pixels * 4);
    fillCanvas(canvas, nx, ny, r, g, b);
    return canvas;
}



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

    // set up shaders
    Shader shader("shaders/vertex.glsl", "shaders/fragment.glsl");

    // set up vertex data (and buffer(s)) and configure vertex attributes
    // ------------------------------------------------------------------
    float vertices[] = {
        // positions          // colors           // texture coords
         1.0f,  1.0f, 1.0f,   1.0f, 0.0f, 0.0f,     1.0f, 1.0f,   // top right
         1.0f, -1.0f, 1.0f,   0.0f, 1.0f, 0.0f,     1.0f, 0.0f,   // bottom right
        -1.0f, -1.0f, 1.0f,   0.0f, 0.0f, 1.0f,     0.0f, 0.0f,   // bottom left
        -1.0f,  1.0f, 1.0f,   1.0f, 1.0f, 0.0f,     0.0f, 1.0f    // top left
    };

    unsigned int indices[] = {  // note that we start from 0!
        0, 1, 3,   // first triangle
        1, 2, 3
    };

    unsigned int VBO, VAO, EBO;
    glGenVertexArrays(1, &VAO);
    glGenBuffers(1, &VBO);
    glGenBuffers(1, &EBO);
    // bind the vertex array object first, then bind and set vertex buffers, and then configure vertex attribute(s).
    glBindVertexArray(VAO);

    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW);

    // Define attributes
    // Position attribute
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void*) 0);
    glEnableVertexAttribArray(0);

    // Color attribute
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void*)(3* sizeof(float)));
    glEnableVertexAttribArray(1);

    // Texture coord attribute
    glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void*)(6* sizeof(float)));
    glEnableVertexAttribArray(2);

    // note that this is allowed. the call to glVertexAttribPointer registered VBO as the vertex attribute's bound vertex
    // buffer object so afterwards we can safely unbind,
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    // You can unbind the VAO afterwards so other VAO calls won't accidentally modify this VAO, but this rarely happens.
    // Modifying other VAOs requires a call to glBindVertexArray anyways so we generally don't unbind VAOs (nor VAOs)
    // when it's not directly necessary.
    glBindVertexArray(0);

    std::vector<uint8_t> canvas = generateCanvas(NUM_PIXELS_X, NUM_PIXELS_Y, 255, 255, 255);

    auto ant = LangtonAnt(
        NUM_PIXELS_X, NUM_PIXELS_Y,
        std::make_pair(NUM_PIXELS_X / 2, NUM_PIXELS_Y / 2),
        0
    );

    unsigned int texture;
    glGenTextures(1, &texture);
    glBindTexture(GL_TEXTURE_2D, texture);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexImage2D(
        GL_TEXTURE_2D, 0, GL_RGBA, NUM_PIXELS_X, NUM_PIXELS_Y, 0,
        GL_RGBA, GL_UNSIGNED_BYTE, (GLvoid*)canvas.data()
    );

    // uncomment this call to draw in wireframe polygons.
    //glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);

    // Render loop!
    while(!glfwWindowShouldClose(window)) {
        // input
        processInput(window);

        // set background color
        glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);

        // Activate the shader
        shader.use();

        // update the texture
        ant.update();
        ant.draw(canvas);

        glTexImage2D(
            GL_TEXTURE_2D, 0, GL_RGBA, NUM_PIXELS_X, NUM_PIXELS_Y, 0,
            GL_RGBA, GL_UNSIGNED_BYTE, (GLvoid*)canvas.data()
        );

        // Render the triangles
        glBindTexture(GL_TEXTURE_2D, texture);
        glBindVertexArray(VAO);
        glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);
        glBindVertexArray(VAO);
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
        glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);

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
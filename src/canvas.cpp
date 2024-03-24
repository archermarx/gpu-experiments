#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include "canvas.h"
#include "shader.h"

Canvas::Canvas(uint32_t w, uint32_t h)
    : width(w),
      height(h),
      contents(w * h * 4),
      shader("shaders/canvas.vert", "shaders/canvas.frag") {

    glGenVertexArrays(1, &VAO);
    glGenBuffers(1, &VBO);
    glGenBuffers(1, &EBO);

    // bind the vertex array object first, then bind and set vertex buffers, and then configure vertex attribute(s).
    glBindVertexArray(VAO);

    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(SCREEN_QUAD_VERTS), SCREEN_QUAD_VERTS, GL_STATIC_DRAW);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(SCREEN_QUAD_ELEMS), SCREEN_QUAD_ELEMS, GL_STATIC_DRAW);

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

    glGenTextures(1, &texture);
    glBindTexture(GL_TEXTURE_2D, texture);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
}

void Canvas::render() {
    shader.use();

    glTexImage2D(
        GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0,
        GL_RGBA, GL_UNSIGNED_BYTE, (GLvoid*)contents.data()
    );
    glBindTexture(GL_TEXTURE_2D, texture);

    // Render the triangles
    glBindVertexArray(VAO);
    glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);
    glBindVertexArray(VAO);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
    glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);
}
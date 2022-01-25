// Standard headers
#include <iostream>

// Grapihcs headers
#include <glad/glad.h>
#include <GLFW/glfw3.h>

// GLM headers
#include <glm/gtc/matrix_transform.hpp>

// Headers
#include "object.hpp"

// Window size
const int WINDOW_WIDTH = 800;
const int WINDOW_HEIGHT = 600;

// Camera information
struct Camera {
        // Camera position
        vec3 position;

        // FOV
        float fov;

        // Orientation
        vec3 front;
        vec3 up;
        vec3 right;
};

Camera camera {
        vec3(0.0f, 0.0f, 0.0f),
        90.0f,
        
        vec3(0.0f, 0.0f, 1.0f),
        vec3(0.0f, 1.0f, 0.0f),
        vec3(1.0f, 0.0f, 0.0f)
};

// Pixel ubffer data
struct uvec4 {
        uint8_t r;
        uint8_t g;
        uint8_t b;
        uint8_t a;
};

// Keyboard callback
bool rerender = true;

void key_callback(GLFWwindow *window, int key, int scancode, int action, int mods)
{
        if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS)
                glfwSetWindowShouldClose(window, GL_TRUE);

        // Move camera
        float camera_speed = 0.5f;
        if (key == GLFW_KEY_W) {
                camera.position += camera.front * camera_speed;
                rerender = true;
        } else if (key == GLFW_KEY_S) {
                camera.position -= camera.front * camera_speed;
                rerender = true;
        }

        if (key == GLFW_KEY_A) {
                camera.position += camera.right * camera_speed;
                rerender = true;
        } else if (key == GLFW_KEY_D) {
                camera.position -= camera.right * camera_speed;
                rerender = true;
        }
}

// Initilize GLFW
GLFWwindow *init_glfw()
{
        // Initilize GLFW
        glfwInit();
       
        GLFWwindow *window;
        
        window = glfwCreateWindow(
                WINDOW_WIDTH,
                WINDOW_HEIGHT,
                "Cross hatching",
                NULL, NULL
        );
        
        glfwSetWindowTitle(window, "Cross hatching");
        glfwMakeContextCurrent(window);
        glfwSetKeyCallback(window, key_callback);
        glfwSwapInterval(1);

        // Initialize glad
        if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
                fprintf(stderr, "Failed to initialize GLAD\n");
                exit(EXIT_FAILURE);
        }
        
        return window;
}

// All objects
int nobjs = 4;

Object *objects[] = {
        new Sphere(vec3(0.0f, 0.0f, 4.0f), 1.0f),
        new Sphere(vec3(3.0f, 0.0f, 3.0f), 3.0f),
        new Sphere(vec3(6.0f, -2.0f, 5.0f), 6.0f),
        new Sphere(vec3(6.0f, 3.0f, 10.0f), 2.0f)
};

// Initialize the pixel buffer
uvec4 *init_pixels(const uvec4 &base)
{
        // TODO: color buffer for each unique object (by address)
        uvec4 *out = new uvec4[WINDOW_WIDTH * WINDOW_HEIGHT];
        for (int i = 0; i < WINDOW_WIDTH * WINDOW_HEIGHT; i++)
                out[i] = base;

        return out;
}

void clear(uvec4 *pixels, const uvec4 &base)
{
        for (int i = 0; i < WINDOW_WIDTH * WINDOW_HEIGHT; i++)
                pixels[i] = base;
}

void render(uvec4 *pixels)
{
        // Color wheel
        static uvec4 colors[] = {
                {0x00, 0x00, 0xFF, 0xFF},
                {0x00, 0xFF, 0x00, 0xFF},
                {0x00, 0xFF, 0xFF, 0xFF},
                {0xFF, 0x00, 0x00, 0xFF},
                {0xFF, 0x00, 0xFF, 0xFF},
                {0xFF, 0xFF, 0x00, 0xFF},
                {0xFF, 0xFF, 0xFF, 0xFF}
        };

        // Color wheel index
        int cid = 0;

        // Each object to a color
        int *obj_colors = new int[nobjs];
        for (int i = 0; i < nobjs; i++) {
                obj_colors[i] = cid++;
                if (cid == 8)
                        cid = 0;
        }

        // Aspect ratio
        float scale = glm::tan(glm::radians(0.5f * camera.fov));
        float aspect = (float) WINDOW_WIDTH / (float) WINDOW_HEIGHT;

        // Camera matrix
        mat4 camera_matrix = glm::identity <mat4> ();

        // Iterate over the pixels
        for (int y = 0; y < WINDOW_HEIGHT; y++) {
                for (int x = 0; x < WINDOW_WIDTH; x++) {
                        // NDC coordinates
                        float nx = (x + 0.5f) / (float) WINDOW_WIDTH;
                        float ny = (y + 0.5f) / (float) WINDOW_HEIGHT;

                        // Camera coordinates
                        float cx = (2.0f * nx - 1.0f) * aspect * scale;
                        float cy = (1.0f - 2.0f * ny) * scale;

                        // Final pixel coordinates
                        // vec3 p = camera.position + camera.front * cx + camera.up * cy;
                        vec3 p = vec3(cx, cy, -1.0f) + camera.position;

                        // Construct the ray
                        Ray ray {
                                camera.position,
                                glm::normalize(p - camera.position)
                        };

                        // Find the closest object
                        vec3 intersection;
                        
                        int iclose = -1;
                        float dclose = std::numeric_limits <float> ::max();

                        for (int i = 0; i < nobjs; i++) {
                                if (objects[i]->intersect(ray, intersection)) {
                                        float d = glm::length(intersection - camera.position);
                                        if (d < dclose) {
                                                dclose = d;
                                                iclose = i;
                                        }
                                }
                        }

                        // If there is an intersection
                        if (iclose != -1) {
                                // Color the pixel
                                pixels[y * WINDOW_WIDTH + x] = colors[obj_colors[iclose]];
                        }
                }
        }

        // Free memory
        delete[] obj_colors;

        // TODO: anti aliasing pass, plus any post processing
}

// TODO: imgui monitor

// Main function
int main()
{
        GLFWwindow *window = init_glfw();

        // Initialize the pixel buffer and texture
        uvec4 base = {50, 50, 50, 255};
        uvec4 *pixels = init_pixels(base);
        
        // Render to the pixel buffer
        render(pixels);

        while (!glfwWindowShouldClose(window)) {
                // Render the pixels if needed
                if (rerender) {
                        // Render the pixels
                        clear(pixels, base);
                        render(pixels);

                        // Update the texture
                        rerender = false;
                }

                // Render the texture
                glDrawPixels(
                        WINDOW_WIDTH,
                        WINDOW_HEIGHT,
                        GL_RGBA,
                        GL_UNSIGNED_BYTE,
                        pixels
                );

                // Swap the buffers
                glfwSwapBuffers(window);
                glfwPollEvents();
        }

        return 0;
}

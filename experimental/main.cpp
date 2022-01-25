// Standard headers
#include <iostream>

// Grapihcs headers
#include <glad/glad.h>
#include <GLFW/glfw3.h>

// Headers
#include "object.hpp"

// Window size
const int WINDOW_WIDTH = 800;
const int WINDOW_HEIGHT = 600;

// Camera information
struct Camera {
        // Camera position
        vec3 position;
        float distance; // Distance from the film to the center of the camera

        // Orientation
        vec3 front;
        vec3 up;

        // Frustum angles
        float hangle;
        float vangle;
};

Camera camera {
        vec3(0.0f, 0.0f, -10.0f),
        1.0f,
        
        vec3(0.0f, 0.0f, 1.0f),
        vec3(0.0f, 1.0f, 0.0f),
        
        45.0f,
        45.0f
};

// Pixel ubffer data
struct uvec4 {
        uint8_t r;
        uint8_t g;
        uint8_t b;
        uint8_t a;
};

// Keyboard callback
void key_callback(GLFWwindow *window, int key, int scancode, int action, int mods)
{
        if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS)
                glfwSetWindowShouldClose(window, GL_TRUE);
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
        new Sphere(vec3(3.0f, 0.0f, 5.0f), 3.0f),
        new Sphere(vec3(6.0f, -2.0f, 5.0f), 6.0f),
        new Sphere(vec3(6.0f, 3.0f, 10.0f), 2.0f)
};

// Initialize the pixel buffer
uvec4 *init_pixels(const uvec4 &base)
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

        // TODO: color buffer for each unique object (by address)
        uvec4 *out = new uvec4[WINDOW_WIDTH * WINDOW_HEIGHT];
        
        /* for (int i = 0; i < WINDOW_WIDTH * WINDOW_HEIGHT; i++)
                out[i] = base; */
        
        // Iterate over the pixels and generate rays
        for (int y = 0; y < WINDOW_HEIGHT; y++) {
                for (int x = 0; x < WINDOW_WIDTH; x++) {
                        // Calculate the ray direction
                        vec3 direction = vec3(
                                (2.0f * (x + 0.5f) / WINDOW_WIDTH - 1.0f) * tan(camera.vangle / 2.0f * M_PI / 180.0f),
                                (2.0f * (y + 0.5f) / WINDOW_HEIGHT - 1.0f) * tan(camera.hangle / 2.0f * M_PI / 180.0f),
                                -1.0f
                        );
                        
                        // Normalize the direction
                        direction = normalize(direction);

                        // Create the ray
                        Ray ray {
                                camera.position,
                                direction
                        };

                        // Set initial color
                        out[y * WINDOW_WIDTH + x] = base;
                        
                        // Iterate over all objects
                        vec3 pos;

                        // TODO: find the closest object
                        for (int i = 0; i < nobjs; i++) {
                                // Calculate the intersection
                                bool inter = objects[i]->intersect(ray, pos);
                                
                                // If the ray intersects the object
                                if (inter) {
                                        // Calculate the pixel color
                                        out[y * WINDOW_WIDTH + x] = colors[obj_colors[i]];

                                        // Break the loop
                                        break;
                                }
                        }
                }
        }

        delete[] obj_colors;

        return out;
}

// Main function
int main()
{
        GLFWwindow *window = init_glfw();

        // Initialize the pixel buffer and texture
        uvec4 base = {255, 255, 255, 255};
        uvec4 *pixels = init_pixels(base);

        while (!glfwWindowShouldClose(window)) {
                // Clear the screen
                glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
                glClear(GL_COLOR_BUFFER_BIT);

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

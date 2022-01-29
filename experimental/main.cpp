// Standard headers
#include <iostream>

// ImGui headers
#include <imgui.h>
#include <backends/imgui_impl_glfw.h>
#include <backends/imgui_impl_opengl3.h>

// Grapihcs headers
#include <glad/glad.h>
#include <GLFW/glfw3.h>

// GLM headers
#include <glm/gtc/matrix_transform.hpp>

// Headers
#include "object.hpp"
#include "light.hpp"
#include "camera.hpp"

// Window size
const int WINDOW_WIDTH = 800;
const int WINDOW_HEIGHT = 600;

// Printing vec3
std::ostream &operator<<(std::ostream &os, const vec3 &v)
{
        return os << "<" << v.x << ", " << v.y << ", " << v.z << ">";
}

// Pixel ubffer data
struct uvec4 {
        uint8_t r;
        uint8_t g;
        uint8_t b;
        uint8_t a;
};

// Basic operations
inline uvec4 operator*(float s, const uvec4 &v) {
        return {
                (uint8_t) (s * v.r),
                (uint8_t) (s * v.g),
                (uint8_t) (s * v.b),
                (uint8_t) (s * v.a)
        };
}

// Keyboard callback
bool rerender = true;

void key_callback(GLFWwindow *window, int key, int scancode, int action, int mods)
{
        if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS)
                glfwSetWindowShouldClose(window, GL_TRUE);

        // Camera movement
        float speed = 0.5f;
        if (key == GLFW_KEY_W) {
                camera.transform.position += camera.transform.forward * speed;
                rerender = true;
        } else if (key == GLFW_KEY_S) {
                camera.transform.position -= camera.transform.forward * speed;
                rerender = true;
        }

        if (key == GLFW_KEY_A) {
                camera.transform.position -= camera.transform.right * speed;
                rerender = true;
        } else if (key == GLFW_KEY_D) {
                camera.transform.position += camera.transform.right * speed;
                rerender = true;
        }
}

// Initilize GLFW
GLFWwindow *init_graphics()
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

        // Initialize ImGui
        IMGUI_CHECKVERSION();
        ImGui::CreateContext();
        ImGuiIO &io = ImGui::GetIO();

        ImGui::StyleColorsDark();

        ImGui_ImplGlfw_InitForOpenGL(window, true);
        ImGui_ImplOpenGL3_Init("#version 330");

        return window;
}

// All objects
int nobjs = 5;

Renderable *objects[] = {
        new Sphere(vec3(0.0f, 0.0f, 4.0f), 1.0f),
        new Sphere(vec3(3.0f, 0.0f, 3.0f), 3.0f),
        new Sphere(vec3(6.0f, -2.0f, 5.0f), 6.0f),
        new Sphere(vec3(6.0f, 3.0f, 10.0f), 2.0f),
        new Sphere(vec3(6.0f, 3.0f, -4.0f), 2.0f),
};

Object *lights[] = {
        new PointLight(vec3(0.0f, 0.0f, 0.0f))
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

// Convert color to discetized grey scale value
uvec4 dcgrey(const uvec4 &color)
{
        // Discrete steps
        static const int steps = 16;

        // Get grey scale vector
        uint8_t grey = (color.r + color.g + color.b) / 3;

        // Discretize (remove remainder)
        grey = grey - (grey % steps);

        // Return color
        return uvec4 {grey, grey, grey, color.a};
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

        // Iterate over the pixels
        glm::vec3 cpos = camera.transform.position;
        for (int y = 0; y < WINDOW_HEIGHT; y++) {
                for (int x = 0; x < WINDOW_WIDTH; x++) {
                        // NDC coordinates
                        float nx = (x + 0.5f) / (float) WINDOW_WIDTH;
                        float ny = (y + 0.5f) / (float) WINDOW_HEIGHT;

                        Ray ray = camera.ray(nx, ny);

                        // Find the closest object
                        int iclose = -1;

                        // TODO: change to tclose
                        float dclose = std::numeric_limits <float> ::max();

                        for (int i = 0; i < nobjs; i++) {
                                float d = objects[i]->intersect(ray);
                                if (d > 0 && dclose > d) {
                                        dclose = d;
                                        iclose = i;
                                }
                        }

                        // If there is an intersection
                        if (iclose != -1) {
                                // Get point of intersection
                                vec3 p = ray.at(dclose);

                                // Calculate diffuse
                                vec3 n = objects[iclose]->normal(p);
                                vec3 l = glm::normalize(lights[0]->position - p);
                                float diffuse = glm::max(glm::dot(n, l), 0.0f);

                                // Color the pixel
                                uvec4 fcolor = diffuse * colors[obj_colors[iclose]];
                                pixels[y * WINDOW_WIDTH + x] = dcgrey(fcolor);
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
        GLFWwindow *window = init_graphics();

        // Initialize the pixel buffer and texture
        uvec4 base = {200, 200, 200, 255};
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

                // Display ImGui fps counter
                // TODO: monitor function
                {
                        ImGui_ImplOpenGL3_NewFrame();
                        ImGui_ImplGlfw_NewFrame();
                        ImGui::NewFrame();

                        ImGui::Begin("Monitor");
                        ImGui::Text("%.1f fps", ImGui::GetIO().Framerate);
                        ImGui::End();

                        ImGui::Render();
                        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
                }

                // Swap the buffers
                glfwSwapBuffers(window);
                glfwPollEvents();
        }

        return 0;
}

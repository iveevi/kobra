#ifndef GLOBAL_H_
#define GLOBAL_H_

#define MERCURY_VALIDATION_LAYERS
#define MERCURY_VALIDATION_ERROR_ONLY
// #define MERCURY_THROW_ERROR

// Engine headers
#include "include/backend.hpp"
#include "include/camera.hpp"
#include "include/logger.hpp"
#include "include/types.h"

// Pixel buffers
extern Vulkan::Buffer pixel_buffer;
extern Vulkan::Buffer world_buffer;
extern Vulkan::Buffer object_buffer;
extern Vulkan::Buffer light_buffer;

// Compute shader
extern VkShaderModule compute_shader;

// Global camera
extern Camera camera;

// Vulkan/GLFW helpers
void cmd_buffer_maker(Vulkan *, size_t);
void descriptor_set_maker(Vulkan *, size_t);

void key_callback(GLFWwindow *, int, int, int, int);

void mouse_callback(GLFWwindow *, double, double);

#endif

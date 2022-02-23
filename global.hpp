#ifndef GLOBAL_H_
#define GLOBAL_H_

#define MERCURY_VALIDATION_LAYERS
#define MERCURY_VALIDATION_ERROR_ONLY
// #define MERCURY_THROW_ERROR

// Standard headers
#include <memory>

// Engine headers
#include "include/app.hpp"
#include "include/backend.hpp"
#include "include/buffer_manager.hpp"
#include "include/bvh.hpp"
#include "include/camera.hpp"
#include "include/capture.hpp"
#include "include/core.hpp"
#include "include/logger.hpp"
#include "include/mesh.hpp"
#include "include/model.hpp"
#include "include/primitive.hpp"
#include "include/profiler.hpp"
#include "include/scene.hpp"
#include "include/texture.hpp"
#include "include/timer.hpp"
#include "include/types.hpp"
#include "include/world.hpp"

// Global world data
extern mercury::World world;

// GLFW helpers
void key_callback(GLFWwindow *, int, int, int, int);
void mouse_callback(GLFWwindow *, double, double);

#endif

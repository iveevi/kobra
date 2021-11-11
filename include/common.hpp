#ifndef COMMON_H_
#define COMMON_H_

// Standard headers
#include <iostream>

// GLFW headers
#include "../glad/glad.h"
#include <GLFW/glfw3.h>

// GLM headers
#include <glm/glm.hpp>

// Engine headers
#include "include/logger.hpp"

// Error checking utilities
inline GLenum __glCheckError(const char *file, int line)
{
        GLenum err;
        while ((err = glGetError()) != GL_NO_ERROR) {
                std::string error;
                switch (err) {
                case GL_INVALID_ENUM:
                        error = "INVALID_ENUM";
                        break;
                case GL_INVALID_VALUE:
                        error = "INVALID_VALUE";
                        break;
                case GL_INVALID_OPERATION:
                        error = "INVALID_OPERATION";
                        break;
                case GL_STACK_OVERFLOW:
                        error = "STACK_OVERFLOW";
                        break;
                case GL_STACK_UNDERFLOW:
                        error = "STACK_UNDERFLOW";
                        break;
                case GL_OUT_OF_MEMORY:
                        error = "OUT_OF_MEMORY";
                        break;
                case GL_INVALID_FRAMEBUFFER_OPERATION:
                        error = "INVALID_FRAMEBUFFER_OPERATION";
                        break;
                }

		mercury::Logger::error("OpenGL error: " + error
			+ " at " + file + " (" + std::to_string(line) + ")");
        }

        return err;
}

#define glCheckError() __glCheckError(__FILE__, __LINE__)

// Printing utilities
std::ostream &operator<<(std::ostream &, const glm::vec3 &);

// Miscellaneous
std::string read_code(const char *);
std::string read_code(const std::string &);

#endif

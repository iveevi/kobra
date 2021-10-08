#ifndef INIT_H_
#define INIT_H_

// Standard headers
#include <iostream>
#include <unordered_map>

// GLFW headers
#include "../glad/glad.h"
#include <GLFW/glfw3.h>

// GLM headers
#include <glm/glm.hpp>

// FreeType headers
#include <ft2build.h>
#include FT_FREETYPE_H

// Engine headers
#include "shader.hpp"
#include "mouse_bus.hpp"

namespace mercury {

// Wrapped struct
struct Window {
	GLFWwindow *window;
	float width;
	float height;
	MouseBus mouse_handler;
} extern cwin;

glm::vec2 transform(const glm::vec2 &);

// Character struct and map
struct Char {
	unsigned int tid;
	glm::ivec2 size;
	glm::ivec2 bearing;
	unsigned int offset;

	static Shader shader;
};

// Character mapping
// TODO: need to consider loading multiple font packs
// 	make functions for these
extern std::unordered_map <char, Char> cmap;

// Parts of initialization
// TODO: hide
void load_fonts();

// First function that should run
// TODO: should take some kind of configuration file?
void init();

// Extra functions
void focus(float, float, float, float);

}

#endif

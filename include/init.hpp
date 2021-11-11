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
struct WindowManager {
	GLFWwindow *cwin;

	// TODO: remove these
	float width;
	float height;

	MouseBus mouse_handler;

	// Array of windows
	std::vector <GLFWwindow *> wins;

	// TODO: add hash table from title to indices

	// Methods
	void add_win(GLFWwindow *);

	void set_wcontext(size_t);
} extern winman;

// glm::vec2 transform(const glm::vec2 &);

// Character struct and map
struct Char {
	unsigned int tid;
	glm::ivec2 size;
	glm::ivec2 bearing;
	unsigned int offset;

	// TODO: put this shader into the Shader class
	static Shader shader;
};

// Character mapping
// TODO: need to consider loading multiple font packs
// 	make functions for these
extern std::unordered_map <char, Char> cmap;

// First function that should run
void init(bool = true);

}

#endif

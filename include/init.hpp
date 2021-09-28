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

namespace mercury {

// Dimensions of current window
// TODO: wrap in nameless struct later?
extern float win_width;
extern float win_height;

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
void load_fonts();

// First function that should run
// TODO: should take some kind of configuration file?
GLFWwindow *init();

}

#endif

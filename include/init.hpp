#ifndef INIT_H_
#define INIT_H_

// Standard headers
#include <iostream>
#include <vector>
#include <functional>
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
#include "include/shader.hpp"
#include "include/mouse_bus.hpp"

// Defined constants
#define DEFAULT_WINDOW_WIDTH	800.0
#define DEFAULT_WINDOW_HEIGHT	600.0

namespace mercury {

// Wrapped struct
class WindowManager {
	void _add_win(GLFWwindow *);
public:
	// Aliases
	using Renderer = void (*)();

	// Members
	GLFWwindow *cwin;

	// TODO: update everytime the context changes
	float width;
	float height;

	// TODO: change this...
	MouseBus mouse_handler;

	// Array of windows
	// TODO: make private and add a size getter
	std::vector <GLFWwindow *> wins;

	// Loop bindings for each window
	//	references to the current
	//	window can be made using
	//	this sentinel object
	std::vector <Renderer> bindings;

	// TODO: add hash table from title to indices
	// 	TODO: add title indexing as well

	// Adders
	void add_win(const std::string &, float = DEFAULT_WINDOW_WIDTH,
		float = DEFAULT_WINDOW_HEIGHT);

	// Setters
	void set_wcontext(size_t);
	void set_renderer(size_t, Renderer);

	// Render the specified context using
	//	its corresponding binding
	void render(size_t);

	// Indexing
	GLFWwindow *get(size_t index) const {
		return wins[index];
	}

	GLFWwindow *operator[](size_t index) const {
		return wins[index];
	}
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

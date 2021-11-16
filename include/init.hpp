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

// Character struct
struct Char {
	unsigned int	tid;
	unsigned int	offset;
	glm::ivec2	size;
	glm::ivec2	bearing;
};

// Wrapped struct
class WindowManager {
public:
	// Aliases
	using Initializer = void (*)();
	using Renderer = void (*)();
	using RCondition = bool (*)();

	using CMap = std::unordered_map <char, Char>;
private:
	void _add_win(GLFWwindow *);

	// Common resources
	struct {
		std::vector <Shader> text_shaders;
		std::vector <Shader> sb_shaders;
		std::vector <Shader> line_shaders;
		std::vector <CMap> character_maps;
	} _cmn;
public:
	// Members
	GLFWwindow *cwin;

	// TODO: update everytime the context changes
	float width;
	float height;

	// Current resources
	struct {
		Shader *text_shader;
		Shader *sb_shader;
		Shader *line_shader;
		CMap *character_map;
	} cres;

	// TODO: change this...
	MouseBus mouse_handler;

	// Array of windows
	// TODO: make private and add a size getter
	std::vector <GLFWwindow *> wins;

	// Initializer for each context
	//	because resources are generally
	//	not sharable across contexts
	std::vector <Initializer> initers;

	// Loop bindings for each window
	//	references to the current
	//	window can be made using
	//	this sentinel object
	std::vector <Renderer> bindings;

	// Hook condition: terminates the
	//	render loop if false
	RCondition condition = nullptr;

	// TODO: add hash table from title to indices
	// 	TODO: add title indexing as well

	// Adders
	void add_win(const std::string &, float = DEFAULT_WINDOW_WIDTH,
		float = DEFAULT_WINDOW_HEIGHT);

	// Setters
	void set_wcontext(size_t);
	void set_initializer(size_t, Initializer);
	void set_renderer(size_t, Renderer);
	void set_condition(RCondition);

	// Loaders
	void load_font(size_t);
	void load_skybox(size_t);
	void load_lines(size_t);

	// Initial the specified context
	//	using the corresponding initializer
	void initialize(size_t);

	// Render the specified context using
	//	its corresponding binding
	void render(size_t);

	// Initializes all windows sequentially
	void initialize_all();

	// Renders all windows sequentially
	void render_all();

	// Full program run
	//	first initializes all contexts, using the bindings
	//	then renders all in a loop
	void run();

	// Turn off all windows
	void close_all();

	// Indexing
	GLFWwindow *get(size_t index) const {
		return wins[index];
	}

	GLFWwindow *operator[](size_t index) const {
		return wins[index];
	}
} extern winman;

// First function that should run
void init(bool = true);

}

#endif

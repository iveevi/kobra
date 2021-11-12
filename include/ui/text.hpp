#ifndef TEXT_H_
#define TEXT_H_

// Standard headers
#include <string>

// GLFW headers
#include "../glad/glad.h"
#include <GLFW/glfw3.h>

// GLM headers
#include <glm/glm.hpp>

// Engine headers
#include "../init.hpp"
#include "ui_element.hpp"
#include "include/ui/pure_rect.hpp"

namespace mercury {

namespace ui {

class Text : public UIElement {
	// Each Text object has its own buffer indices
	// to potentially allow multithreaded drawing
	// TODO: should this be made static?
	unsigned int	_vao;
	unsigned int	_vbo;

	float		_xpos;
	float		_ypos;
	float		_scale;

	float		_maxy;

	glm::vec3	_color;

	std::string	_str;

	// TODO: store pointer to character map

	void _alloc();
	void _get_maxy();
public:
	Text();
	Text(const std::string &,
		float = 0.0f, float = 0.0f, float = 1.0f,
		const glm::vec3 & = {1.0, 1.0, 1.0});

	// Setters
	void set_scale(float = 1.0);
	void set_str(const std::string &);
	void set_color(const glm::vec3 &);
	void set_position(float = 0.0, float = 0.0);

	// Getters
	float get_width() const;

	// Alignment
	void center_within(const PureRect &, bool = false);

	// Virtual overrides
	void draw(Shader &) override;

	glm::vec2 get_position() const override;

	void set_position(const glm::vec2 &) override;
};

}

}

#endif

#ifndef KOBRA_UI_SHAPES_H_
#define KOBRA_UI_SHAPES_H_

// GLM headers
#include <glm/glm.hpp>

// Engine headers
#include "../shader_program.hpp"

namespace kobra {

namespace ui {

// Generic shape class
struct Shape {
	// Custom shader program
	// TODO: pointer?
	ShaderProgram shader_program;
};

// Rectangle
struct Rect : public Shape {
	glm::vec2 min;
	glm::vec2 max;
	glm::vec3 color;
	float radius;
	float border_width;

	// Constructor
	Rect(const glm::vec2 &min_ = glm::vec2 {0.0f},
			const glm::vec2 &max_ = glm::vec2 {1.0f},
			const glm::vec3 &color_ = glm::vec3 {1.0f},
			float radius_ = 0.0f, float border_width_ = 0.0f)
			: min(min_), max(max_), color(color_),
			radius(radius_), border_width(border_width_) {}
};

// TODO: circle and polygons

}

}

#endif

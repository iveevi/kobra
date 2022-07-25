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
	glm::vec2 min {0.0f};
	glm::vec2 max {0.0f};
	glm::vec3 color {1.0f};
	float radius {0.0f};
	float border_width {0.0f};
};

// TODO: circle and polygons

}

}

#endif

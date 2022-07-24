#ifndef KOBRA_UI_SHAPES_H_
#define KOBRA_UI_SHAPES_H_

// GLM headers
#include <glm/glm.hpp>

namespace kobra {

namespace ui {

// Rectangle
struct Rect {
	glm::vec2 min {0.0f};
	glm::vec2 max {0.0f};
	glm::vec3 color {1.0f};
	float radius {0.0f};
	float border_width {0.0f};
};

}

}

#endif

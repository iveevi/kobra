#ifndef KOBRA_UI_TEXT_H_
#define KOBRA_UI_TEXT_H_

// Standard headers
#include <string>
#include <memory>

// GLM headers
#include <glm/glm.hpp>

namespace kobra {

// Forward declarations
namespace layers {

class FontRenderer;

}

namespace ui {

// Text object
struct Text {
	std::string	text = "text";
	glm::vec2	anchor {0.0f};
	glm::vec3	color {1.0f};
	float		size {1.0f};

	// Friends
	friend class layers::FontRenderer;
};

}

using TextPtr = std::shared_ptr <ui::Text>;

}

#endif

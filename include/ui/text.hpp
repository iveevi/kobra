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
	std::string	text;
	glm::vec2	anchor;
	glm::vec3	color;
	float		size;

	// Constructor
	Text(const std::string &text_ = "text",
			const glm::vec2 &anchor_ = glm::vec2 {0.0f},
			const glm::vec3 &color_ = glm::vec3 {1.0f},
			float size_ = 1.0f)
			: text(text_), anchor(anchor_), color(color_), size(size_) {}

	// Friends
	friend class layers::FontRenderer;
};

}

using TextPtr = std::shared_ptr <ui::Text>;

}

#endif

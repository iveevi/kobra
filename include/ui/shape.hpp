#ifndef SHAPE_H_
#define SHAPE_H_

// GLM
#include <glm/glm.hpp>

// Engine headers
#include "ui_element.hpp"

namespace mercury {

namespace ui {

class Shape : public UIElement {
public:
	virtual bool contains(const glm::vec2 &) const = 0;
};

}

}

#endif

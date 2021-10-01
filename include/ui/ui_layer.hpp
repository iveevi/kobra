#ifndef UI_LAYER_H_
#define UI_LAYER_H_

// Standard headers
#include <vector>

// Engine headers
#include "ui_element.hpp"

namespace mercury {

namespace ui {

// UILayer is a subclass of UIElement
// to allow for nested of layers...
class UILayer : public UIElement {
	std::vector <UIElement *> 	_elements;
	glm::vec2			_pos = {0, 0};
public:
	void add_element(UIElement *);

	void draw() override;
	glm::vec2 get_position() const override;
	void set_position(const glm::vec2 &) override;
	void move(const glm::vec2 &) override;
};

}

}

#endif

#include "../../include/ui/ui_layer.hpp"

namespace mercury {

namespace ui {

void UILayer::add_element(UIElement *uie)
{
	_elements.push_back(uie);
}

void UILayer::draw()
{
	for (UIElement *uie : _elements)
		uie->draw();
}

glm::vec2 UILayer::get_position() const
{
	return _pos;
}

void UILayer::set_position(const glm::vec2 &pos)
{
	glm::vec2 dpos = pos - _pos;
	_pos = pos;

	for (UIElement *uie : _elements)
		uie->move(dpos);
}

}

}

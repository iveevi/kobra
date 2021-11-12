#include "include/ui/ui_layer.hpp"
#include "include/logger.hpp"

namespace mercury {

namespace ui {

void UILayer::add_element(UIElement *uie)
{
	if (!uie)
		Logger::warn("Adding null element to UILayer");
	_elements.push_back(uie);
}

void UILayer::draw(Shader &shader)
{
	for (UIElement *uie : _elements)
		uie->draw(shader);
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

void UILayer::move(const glm::vec2 &dpos)
{
	_pos += dpos;
	for (UIElement *uie : _elements)
		uie->move(dpos);
}

}

}

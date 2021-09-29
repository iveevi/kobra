#include "../../include/ui/ui_layer.hpp"

namespace mercury {

namespace ui {

void UILayer::add_element(UIElement *uie)
{
	_elements.push_back(uie);
}

void UILayer::draw() const
{
	for (UIElement *uie : _elements)
		uie->draw();
}

}

}

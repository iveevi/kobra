#include "../../include/ui/button.hpp"

namespace mercury {

namespace ui {

Button::Button(const Shape *shape, const EventQueue *queue)
		: _shape(shape), _queue(queue) {}

void Button::draw()
{
	_shape.draw();
}

}

}

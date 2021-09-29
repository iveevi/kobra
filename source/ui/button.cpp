#include "../../include/ui/button.hpp"

// Engine headers
#include "include/init.hpp"

namespace mercury {

namespace ui {

Button::Button(Shape *shape, Handler *handler)
		: _shape(shape), _handler(handler)
{
	// TODO: store returned index
	win_mhandler.subscribe(this, &Button::handler);
}

void Button::draw() const
{
	_shape->draw();
}

void Button::handler(size_t *data)
{
	glm::vec2 mpos = ((MouseBus::Data *) data)->pos;
	if (_shape->contains(mpos))
		on_pressed(mpos);
}

void Button::on_pressed(const glm::vec2 &mpos) const
{
	if (_handler)
		_handler->run((size_t *) &mpos);
}

}

}

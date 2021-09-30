#include "../../include/ui/button.hpp"

// Engine headers
#include "include/init.hpp"

namespace mercury {

namespace ui {

Button::Button(Shape *shape, Handler *handler1, Handler *handler2)
		: _shape(shape), _press_handler(handler1),
		_release_handler(handler2)
{
	// TODO: store returned index
	win_mhandler.subscribe(this, &Button::handler);
}

void Button::handler(size_t *data)
{
	MouseBus::Data *mdata = ((MouseBus::Data *) data);
	if (!_shape->contains(mdata->pos))
		return;

	if (mdata->type == MouseBus::MOUSE_PRESSED)
		on_pressed(mdata->pos);
	if (mdata->type == MouseBus::MOUSE_RELEASED)
		on_released(mdata->pos);
}

void Button::on_pressed(const glm::vec2 &mpos)
{
	if (_press_handler)
		_press_handler->run((size_t *) &mpos);
}

void Button::on_released(const glm::vec2 &mpos)
{
	if (_release_handler)
		_release_handler->run((size_t *) &mpos);
}

void Button::draw()
{
	_shape->draw();
}

glm::vec2 Button::get_position() const
{
	return _shape->get_position();
}


void Button::set_position(const glm::vec2 &pos)
{
	_shape->set_position(pos);
}

void Button::move(const glm::vec2 &dpos)
{
	_shape->move(dpos);
}

}

}

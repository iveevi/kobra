#include "../../include/ui/button.hpp"

// Engine headers
#include "include/init.hpp"
#include "include/ui/ui_layer.hpp"

namespace mercury {

namespace ui {

Button::Button() {}

// cbox is collision region
Button::Button(Shape *cbox, Handler *handler1,
		Handler *handler2, UILayer *layer)
		: _cbox(cbox), _press_handler(handler1),
		_release_handler(handler2), _layer(layer)
{
	// TODO: store returned index
	winman.mouse_handler.subscribe(this, &Button::handler);

	// Localize the ui_layer
	layer->set_position(cbox->get_position());
}

void Button::handler(size_t *data)
{
	MouseBus::Data *mdata = ((MouseBus::Data *) data);
	if (!_cbox->contains(mdata->pos))
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
	_cbox->draw();

	if (_layer)
		_layer->draw();
}

glm::vec2 Button::get_position() const
{
	return _cbox->get_position();
}


void Button::set_position(const glm::vec2 &pos)
{
	_cbox->set_position(pos);

	if (_layer)
		_layer->set_position(pos);
}

void Button::move(const glm::vec2 &dpos)
{
	_cbox->move(dpos);

	if (_layer)
		_layer->move(dpos);
}

}

}

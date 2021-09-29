#include "../../include/ui/button.hpp"

// Engine headers
#include "include/init.hpp"

namespace mercury {

namespace ui {

Button::Button(Shape *shape)
		: _shape(shape)
{
	// TODO: store returned index
	/* win_mhandler.subscribe(
		new MemFtnHandler <Button> (this, &Button::handler)
	); */
	win_mhandler.subscribe(this, &Button::handler);
}

void Button::draw()
{
	_shape->draw();
}

void Button::handler(size_t *data)
{
	std::cout << "BUTTON (" << this << ") HANDLER!" << std::endl;
	if (*data == MouseBus::MOUSE_PRESSED)
		std::cout << "\tPRESS!" << std::endl;
}

}

}

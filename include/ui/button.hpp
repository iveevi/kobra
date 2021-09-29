#ifndef BUTTON_H_
#define BUTTON_H_

// Engine headers
#include "../event_handler.hpp"
#include "../mouse_bus.hpp"
#include "shape.hpp"

namespace mercury {

namespace ui {

// TODO: derive from mousehandler?
class Button {
	Shape *		_shape;
public:
	Button(Shape *);

	void draw();

	void handler(size_t *);
};

}

}

#endif

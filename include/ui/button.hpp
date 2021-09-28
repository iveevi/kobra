#ifndef BUTTON_H_
#define BUTTON_H_

// Engine headers
#include "../event_handler.hpp"
#include "shape.hpp"

namespace mercury {

namespace ui {

class Button {
	Shape *		_shape;
	// EventQueue *	_queue;
public:
	Button(const Shape *, const EventQueue *);

	void draw();

	void process(const glm::vec2 &);
};

}

}

#endif

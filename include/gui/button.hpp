#ifndef BUTTON_H_
#define BUTTON_H_

// Standard headers
#include <memory>

// GLFW headers
#include <GLFW/glfw3.h>

// Engine headers
#include "../app.hpp"
#include "area.hpp"
#include "rect.hpp"

namespace mercury {

namespace gui {

// Button class
class Button {
	// TODO: allow a variety of shapes after abstraction
	std::shared_ptr <Area>	_area;

	Rect			_idle;		// TODO: shapes
	Rect			_hover;
	Rect			_pressed;

	int			_button = -1;
	int			_state = 0;
public:
	// Default
	Button() = default;

	// Constructor needs app window context
	//	to subscribe to mouse events
	Button(App::Window &wctx,
			std::shared_ptr <Area> area,
			Rect idle,
			Rect hover,
			Rect press,
			int button = GLFW_MOUSE_BUTTON_LEFT)
			: _area(area), _idle(idle),
			_hover(hover), _pressed(press),
			_button(button) {
		// Subscribe lambda to mouse events
		// TODO: static method
		auto mouse_callback = [](void *ptr, const io::MouseEvent &me) {
			// Check hover
			Button *button = static_cast <Button *> (ptr);
			if (button->_area->contains(me.xpos, me.ypos)) {
				// Check button
				if (me.button == button->_button  && me.action == GLFW_PRESS)
					button->_state = 2;
				else
					button->_state = 1;
			} else {
				button->_state = 0;
			}
		};

		// TODO: later subscribe to specific events
		wctx.mouse_events.subscribe(mouse_callback, this);
	}

	// Render
	void render(VertexBuffer &vb, IndexBuffer &ib) {
		switch (_state) {
		case 0:
			_idle.upload(vb, ib);
			break;
		case 1:
			_hover.upload(vb, ib);
			break;
		case 2:
			_pressed.upload(vb, ib);
			break;
		}
	}
};

}

}

#endif

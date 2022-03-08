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
class Button : public _element {
public:
	// Specialized button maker
	struct RectButton {
		float x;
		float y;
		float w;
		float h;

		int button;

		glm::vec3 idle;
		glm::vec3 hover;
		glm::vec3 active;
	};
private:
	// TODO: allow a variety of shapes after abstraction
	std::shared_ptr <Area>	_area;

	Rect			_idle;		// TODO: elements
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
		wctx.mouse_events->subscribe(mouse_callback, this);
	}

	// Specialized constructors
	Button(App::Window &wctx, RectButton rb)
			: Button(wctx,
				std::shared_ptr <Area> (new RectArea(rb.x, rb.y, rb.w, rb.h)),
				Rect(wctx, rb.x, rb.y, rb.w, rb.h, rb.idle),
				Rect(wctx, rb.x, rb.y, rb.w, rb.h, rb.hover),
				Rect(wctx, rb.x, rb.y, rb.w, rb.h, rb.active),
				rb.button) {}

	// Render
	void render(RenderPacket &rp) override {
		switch (_state) {
		case 0:
			_idle.render(rp);
			break;
		case 1:
			_hover.render(rp);
			break;
		case 2:
			_pressed.render(rp);
			break;
		}
	}
};

}

}

#endif

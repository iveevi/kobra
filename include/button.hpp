#ifndef KOBRA_BUTTON_H_
#define KOBRA_BUTTON_H_

// Standard headers
#include <memory>

// remov
#include <iostream>
#include <glm/gtx/string_cast.hpp>

// Engine headers
#include "io/event.hpp"
#include "shapes.hpp"

namespace kobra {

// Button class
class Button {
public:
	// Handlers
	// TODO: on hover as well
	using Handler = std::pair <void *, std::function <void (void *)>>;

	// Parameters
	struct Args {
		glm::vec2 min;
		glm::vec2 max;

		glm::vec3 idle;
		glm::vec3 hover;
		glm::vec3 pressed;

		int button;

		std::vector <Handler> handlers;
	};
private:
	// TODO: make a more general button class
	// (i.e. area and shape)
	Rect			_rect;
	glm::vec3		_idle;
	glm::vec3		_hover;
	glm::vec3		_pressed;
	int			_button = GLFW_MOUSE_BUTTON_LEFT;
	std::vector <Handler> _handlers = {};

	io::MouseEventQueue	*_queue = nullptr;

	// Event hanlder
	static void _event_handler(void *user, const io::MouseEvent &event) {
		Button *button = static_cast <Button *> (user);

		bool in_bounds = button->_rect.min.x <= event.xpos &&
				button->_rect.max.x >= event.xpos &&
				button->_rect.min.y <= event.ypos &&
				button->_rect.max.y >= event.ypos;

		if (in_bounds) {
			if (event.button == button->_button && event.action == GLFW_PRESS) {
				button->_rect.color = button->_pressed;

				for (auto &handler : button->_handlers)
					handler.second(handler.first);
			} else {
				button->_rect.color = button->_hover;
			}
		} else {
			button->_rect.color = button->_idle;
		}
	}
public:
	// Default
	Button() = default;

	// Constructor
	Button(io::MouseEventQueue &mouse_events, Args args)
			: _rect {args.min, args.max, args.idle},
			_idle {args.idle},
			_hover {args.hover},
			_pressed {args.pressed},
			_button(args.button),
			_handlers(args.handlers),
			_queue(&mouse_events) {
		// Subscribe lambda to mouse events
		_queue->subscribe(_event_handler, this);
	}

	// No copy
	Button(const Button &) = delete;
	Button &operator=(const Button &) = delete;

	// Move must be very careful
	Button(Button &&other)
			: _rect(std::move(other._rect)),
			_idle(std::move(other._idle)),
			_hover(std::move(other._hover)),
			_pressed(std::move(other._pressed)),
			_button(other._button),
			_handlers(std::move(other._handlers)),
			_queue(other._queue) {
		// Unsubscribe from other's mouse events
		_queue->unsubscribe(&other);

		// Subscribe to this's mouse events
		_queue->subscribe(_event_handler, this);
	}

	// Move assignment
	Button &operator=(Button &&other) {
		_rect = std::move(other._rect);
		_idle = std::move(other._idle);
		_hover = std::move(other._hover);
		_pressed = std::move(other._pressed);
		_button = other._button;
		_handlers = std::move(other._handlers);
		_queue = other._queue;

		// Unsubscribe from other's mouse events
		_queue->unsubscribe(&other);

		// Subscribe to this's mouse events
		_queue->subscribe(_event_handler, this);

		return *this;
	}

	// Get the button's shape
	const Rect &shape() const {
		return _rect;
	}
};

using ButtonPtr = std::shared_ptr <Button>;

}

#endif

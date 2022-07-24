#ifndef KOBRA_UI_BUTTON_H_
#define KOBRA_UI_BUTTON_H_

// Standard headers
#include <memory>

// remov
#include <iostream>
#include <glm/gtx/string_cast.hpp>

// Engine headers
#include "../io/event.hpp"
#include "shapes.hpp"

namespace kobra {

namespace ui {

// Button class
// TODO: template on shapes?
class Button {
public:
	// Handlers
	using OnClick = std::pair <void *, std::function <void (void *)>>;
	using OnDrag = std::pair <void *, std::function <void (void *, glm::vec2)>>;

	// Parameters
	struct Args {
		glm::vec2 min;
		glm::vec2 max;
		float radius;
		float border_width;

		glm::vec3 idle;
		glm::vec3 hover;
		glm::vec3 pressed;

		int button = GLFW_MOUSE_BUTTON_LEFT;

		std::vector <OnClick> on_click;
		std::vector <OnDrag> on_drag;
	};
private:
	// TODO: make a more general button class
	// (i.e. area and shape)
	Rect			_rect;
	bool			_is_pressed = false;
	glm::vec2		_drag_start;
	glm::vec3		_hover;
	glm::vec3		_idle;
	glm::vec3		_pressed;
	int			_button = GLFW_MOUSE_BUTTON_LEFT;
	std::vector <OnClick>	_on_click;
	std::vector <OnDrag>	_on_drag;

	io::MouseEventQueue	*_queue = nullptr;

	// Event hanlder
	static void _event_handler(void *user, const io::MouseEvent &event) {
		Button *button = static_cast <Button *> (user);

		bool is_button = (event.button == button->_button);
		bool is_pressed = (event.action == GLFW_PRESS);
		bool is_released = (event.action == GLFW_RELEASE);
		bool in_bounds = button->_rect.min.x <= event.xpos &&
				button->_rect.max.x >= event.xpos &&
				button->_rect.min.y <= event.ypos &&
				button->_rect.max.y >= event.ypos;

		if (in_bounds && is_button) {
			if (is_pressed) {
				button->_is_pressed = true;
				button->_rect.color = button->_pressed;

				for (auto &handler : button->_on_click)
					handler.second(handler.first);

				button->_drag_start = {event.xpos, event.ypos};
			} else if (is_released) {
				// std::cout << "On release" << std::endl;
				button->_is_pressed = false;
				button->_rect.color = button->_hover;
			}
		} else {
			if (is_button && is_released)
				button->_is_pressed = false;

			if (button->_is_pressed) {
				button->_rect.color = button->_pressed;

				glm::vec2 pos = {event.xpos, event.ypos};
				for (auto &handler : button->_on_drag)
					handler.second(handler.first, button->_drag_start - pos);

				button->_drag_start = pos;
			} else if (in_bounds) {
				button->_rect.color = button->_hover;
			} else {
				button->_rect.color = button->_idle;
				button->_is_pressed = false;
			}
		}
	}
public:
	// Default
	Button() = default;

	// Constructor
	Button(io::MouseEventQueue &mouse_events, Args args)
			: _rect {args.min, args.max, args.idle, args.radius, args.border_width},
			_idle {args.idle},
			_hover {args.hover},
			_pressed {args.pressed},
			_button(args.button),
			_on_click(args.on_click),
			_on_drag(args.on_drag),
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
			_on_click(std::move(other._on_click)),
			_on_drag(std::move(other._on_drag)),
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
		_on_click = std::move(other._on_click);
		_on_drag = std::move(other._on_drag);
		_queue = other._queue;

		// Unsubscribe from other's mouse events
		_queue->unsubscribe(&other);

		// Subscribe to this's mouse events
		_queue->subscribe(_event_handler, this);

		return *this;
	}

	// Get the button's shape
	Rect &shape() {
		return _rect;
	}

	const Rect &shape() const {
		return _rect;
	}
};

}

using ButtonPtr = std::shared_ptr <ui::Button>;

}

#endif

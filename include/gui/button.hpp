#ifndef BUTTON_H_
#define BUTTON_H_

// Standard headers
#include <memory>

// GLFW headers
#include <GLFW/glfw3.h>

// Engine headers
#include "../app.hpp"
#include "area.hpp"
#include "gui.hpp"
#include "rect.hpp"

namespace kobra {

namespace gui {

// Button class
class Button : public _element {
public:
	// Specialized button maker
	struct RectButton {
		coordinates::Screen pos;
		coordinates::Screen size;

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
	Button(App::IO &io, std::shared_ptr <Area> area,
			Rect idle,
			Rect hover,
			Rect press,
			int button = GLFW_MOUSE_BUTTON_LEFT)
			: _area(area),
			_idle {std::move(idle)},
			_hover {std::move(hover)},
			_pressed {std::move(press)},
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
		io.mouse_events.subscribe(mouse_callback, this);
	}

	// Specialized constructors
	Button(const vk::raii::PhysicalDevice &phdev,
			const vk::raii::Device &device,
			App::IO &io, RectButton rb)
			: Button(io,
				std::shared_ptr <Area> (new RectArea(
					rb.pos.x, rb.pos.y,
					rb.size.x, rb.size.y
				)),
				Rect(phdev, device, rb.pos, rb.size, rb.idle),
				Rect(phdev, device, rb.pos, rb.size, rb.hover),
				Rect(phdev, device, rb.pos, rb.size, rb.active),
				rb.button) {}

	// Virtual methods
	glm::vec2 position() const override {
		// Get min position of all elements
		glm::vec2 p = _idle.position();
		p = glm::min(p, _hover.position());
		p = glm::min(p, _pressed.position());
		return p;
	}

	glm::vec4 bounding_box() const override {
		std::vector <_element *> elements {
			(_element *) &_idle,
			(_element *) &_hover,
			(_element *) &_pressed
		};

		return get_bounding_box(elements);
	}

	// Latching onto a layer
	void latch(LatchingPacket &lp) override {}

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

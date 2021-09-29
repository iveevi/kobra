#ifndef BUTTON_H_
#define BUTTON_H_

// Engine headers
#include "../event_handler.hpp"
#include "../mouse_bus.hpp"
#include "shape.hpp"

namespace mercury {

namespace ui {

// TODO: derive from mousehandler?
// TODO: derive from UIElement
class Button : public UIElement {
	Shape *		_shape;
	Handler *	_handler;
public:
	Button(Shape *, Handler * = nullptr);

	void draw() const override;
	void handler(size_t *);

	virtual void on_pressed(const glm::vec2 &) const;
};

}

}

#endif

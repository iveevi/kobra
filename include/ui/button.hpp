#ifndef BUTTON_H_
#define BUTTON_H_

// Engine headers
#include "../event_handler.hpp"
#include "../mouse_bus.hpp"
#include "include/ui/ui_layer.hpp"
#include "include/ui/shape.hpp"

namespace mercury {

namespace ui {

// TODO: derive from mousehandler?
// TODO: derive from UIElement
class Button : public UIElement {
protected:
	Shape *		_cbox;
	Handler *	_press_handler;
	Handler *	_release_handler;
	UILayer *	_layer;
public:
	Button();
	Button(Shape *, Handler * = nullptr,
		Handler * = nullptr, UILayer * = nullptr);

	void handler(size_t *);

	virtual void on_pressed(const glm::vec2 &);
	virtual void on_released(const glm::vec2 &);

	void draw(Shader &) override;
	glm::vec2 get_position() const override;
	void set_position(const glm::vec2 &) override;
	void move(const glm::vec2 &) override;
};

}

}

#endif

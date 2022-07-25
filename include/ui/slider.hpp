#ifndef KOBRA_UI_SLIDER_H_
#define KOBRA_UI_SLIDER_H_

// Standard headers
#include <functional>

// GLM headers
#include <glm/glm.hpp>

// Engine headers
#include "../common.hpp"
#include "../layers/font_renderer.hpp"
#include "button.hpp"
#include "shapes.hpp"
#include "text.hpp"

namespace kobra {

namespace ui {

class Slider {
private:
	// TODO: lambda value function
	float		_value = 0.0f;
	float		_width = 0.0f;

	ui::Button	_button;
	ui::Rect	_rect;

	ui::Text	_t_label;
	ui::Text	_t_value;

	std::function <float (float)>
			_value_func;

	// Members
	layers::FontRenderer *_fr = nullptr;

	// Default value lambda
	struct DefaultValue {
		float min;
		float max;

		DefaultValue(float min_, float max_)
				: min(min_), max(max_) {}

		float operator()(float x) {
			return min + (max - min) * x;
		}
	};

	// Dragging handlers
	static void on_drag(void *user, glm::vec2 dpos) {
		auto *slider = static_cast <Slider *> (user);
		ui::Button &button = slider->_button;

		glm::vec2 max = slider->_rect.max;
		glm::vec2 min = slider->_rect.min;
		float width = slider->_width;

		slider->_value += dpos.x/(max.x - min.x);
		slider->_value = glm::clamp(slider->_value, 0.0f, 1.0f);

		button.shape().min.x = min.x + slider->_value * (max.x - min.x) - width/2.0f;
		button.shape().max.x = min.x + slider->_value * (max.x - min.x) + width/2.0f;

		slider->_t_value.text = common::sprintf("%.2f", slider->_value_func(slider->_value));
		slider->_t_value.anchor.x = max.x - slider->_fr->size(slider->_t_value).x/2.0f + width/2.0f;
	};
public:
	// Arguments for construction
	struct Args {
		float			percent = 0.0f;
		float			step = 0.01f;
		float			dial_width = 10.0f;
		float			font_size = 0.4f;

		glm::vec2		max;
		glm::vec2		min;

		std::string		name;

		layers::FontRenderer	*font_renderer;

		std::function <float (float)>
					value_func = DefaultValue(0.0f, 1.0f);
	};

	// TODO: option to show bounds
	Slider() = default;

	Slider(io::MouseEventQueue &mouse_events, const Args &args)
			: _value(args.percent), _width(args.dial_width),
			_value_func(args.value_func),
			_fr(args.font_renderer) {
		// TODO: configurable:
		static const float thickness = 5.0f;

		// Assert valid font renderer
		KOBRA_ASSERT(_fr != nullptr, "Slider: font renderer is null");

		// Get min/max
		glm::vec2 max = args.max;
		glm::vec2 min = args.min;

		// Correct min/max
		if (min.x > max.x)
			std::swap(min.x, max.x);

		if (min.y > max.y)
			std::swap(min.y, max.y);

		// Value bar
		float miny = (min.y + max.y - thickness)/2.0f;
		float maxy = (min.y + max.y + thickness)/2.0f;

		_rect = ui::Rect {
			.min = {min.x, miny},
			.max = {max.x, maxy},
			.color = glm::vec3 {0.5f}
		};

		// Slider button
		float minx = min.x + _value * (max.x - min.x) - _width/2.0f;
		float maxx = min.x + _value * (max.x - min.x) + _width/2.0f;

		ui::Button::Args button_args {
			.min = {minx, min.y},
			.max = {maxx, max.y},
			.radius = 0.01f,
			.border_width = 0.01f,

			.idle = glm::vec3 {0.5f},
			.hover = glm::vec3 {0.6f},
			.pressed = {0.7f, 0.7f, 0.8f},

			.on_drag = {{this, on_drag}}
		};

		_button = ui::Button(mouse_events, button_args);

		// Label
		_t_label = ui::Text {
			.text = args.name,
			.anchor = {min.x - _width/2.0f, min.y},
			.color = glm::vec3 {1.0f},
			.size = args.font_size,
		};

		float height = min.y - _fr->size(_t_label).y/2.0f - 2.0f * thickness;
		_t_label.anchor.y = height;

		// Value
		_t_value = ui::Text {
			.text = common::sprintf("%.2f", _value),
			.anchor = {max.x + _width/2.0f, height},
			.color = glm::vec3 {0.8f},
			.size = args.font_size,
		};

		_t_value.anchor.x = max.x - _fr->size(_t_value).x/2.0f;
	}

	// No copy, move only
	Slider(const Slider &) = delete;
	Slider &operator=(const Slider &) = delete;

	// Move constructor
	Slider(Slider &&other)
			: _value(other._value), _width(other._width),
			_button(std::move(other._button)), _rect(std::move(other._rect)),
			_t_label(std::move(other._t_label)), _t_value(std::move(other._t_value)),
			_fr(other._fr), _value_func(other._value_func) {
		_button.clear_handlers();
		_button.add_on_drag({this, on_drag});
	}

	// Move assignment
	Slider &operator=(Slider &&other) {
		_value = other._value;
		_width = other._width;
		_button = std::move(other._button);
		_rect = std::move(other._rect);
		_t_label = std::move(other._t_label);
		_t_value = std::move(other._t_value);
		_fr = other._fr;
		_value_func = other._value_func;

		_button.clear_handlers();
		_button.add_on_drag({this, on_drag});

		return *this;
	}

	// Current value
	float value() const {
		return _value_func(_value);
	}

	// Shapes
	std::vector <ui::Rect> shapes() {
		std::vector <ui::Rect> shapes;
		shapes.push_back(_rect);
		shapes.push_back(_button.shape());
		return shapes;
	}

	// Text
	std::vector <ui::Text> texts() {
		std::vector <ui::Text> text;
		text.push_back(_t_label);
		text.push_back(_t_value);
		return text;
	}
};

}

}

#endif

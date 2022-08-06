#ifndef KOBRA_UI_COLOR_PICKER_H_
#define KOBRA_UI_COLOR_PICKER_H_

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

// Color picker
// TODO: ui elements and move methods for layout managers
struct ColorPicker {
	ui::Rect r_saturation;
	ui::Rect r_hue;
	ui::Rect r_result;

	// TODO: remove all text from color picker
	ui::Text t_label;
	ui::Text t_color;

	ui::Button b_square;
	ui::Button b_saturation;

	ui::Button b_square_hue;
	ui::Button b_hue_slider;

	float hue;
	glm::vec2 saturation;

	glm::vec3 *ref = nullptr;

	static constexpr float button_size = 15.0f;

	layers::FontRenderer *fr;

	static glm::vec3 hue_to_rgb(float hue, float saturation, float lightness) {
		// Convert to RGB
		float chroma = (1 - abs(2 * lightness - 1)) * saturation;
		float x = chroma * (1 - abs(fmod(hue/60.0f, 2) - 1));
		float m = lightness - chroma / 2;

		float r = 0.0f;
		float g = 0.0f;
		float b = 0.0f;

		if (hue < 60.0f) {
			r = chroma;
			g = x;
		} else if (hue < 120.0f) {
			r = x;
			g = chroma;
		} else if (hue < 180.0f) {
			g = chroma;
			b = x;
		} else if (hue < 240.0f) {
			g = x;
			b = chroma;
		} else if (hue < 300.0f) {
			r = x;
			b = chroma;
		} else if (hue < 360.0f) {
			r = chroma;
			b = x;
		}

		return glm::vec3 {r + m, g + m, b + m};
	}

	static float rgb_to_hue(const glm::vec3 &color) {
		float min = glm::min(color.r, glm::min(color.g, color.b));
		float max = -FLT_MAX;
		int arg_max = 0;

		if (color.r > max) {
			max = color.r;
			arg_max = 0;
		}

		if (color.g > max) {
			max = color.g;
			arg_max = 1;
		}

		if (color.b > max) {
			max = color.b;
			arg_max = 2;
		}

		float hue = 0.0;
		if (arg_max == 0) {
			// Max is red
			hue = 60.0 * (color.g - color.b) / (max - min);
		} else if (arg_max == 1) {
			// Max is green
			hue = 60.0 * (color.b - color.r) / (max - min) + 120.0;
		} else if (arg_max == 2) {
			// Max is blue
			hue = 60.0 * (color.r - color.g) / (max - min) + 240.0;
		}

		if (hue < 0.0)
			hue += 360.0;

		return hue;
	}

	// Set hue from mouse position
	void set_hue(glm::vec2 pos) {
		float y = glm::clamp(pos.y, r_hue.min.y, r_hue.max.y);
		hue = fmod((y - r_hue.min.y)/(r_hue.max.y - r_hue.min.y) * 360.0f, 360.0f);

		glm::vec3 color = hue_to_rgb(hue, saturation.x, 1 - saturation.y);

		// Set position of hue slider
		b_hue_slider.shape()->min = {r_hue.min.x - 2.0f, y - button_size/2.0f};
		b_hue_slider.shape()->max = {r_hue.max.x + 2.0f, y + button_size/2.0f};

		r_saturation.color.x = hue/360.0f;
		r_result.color = color;

		glm::vec3 hue_color = hue_to_rgb(hue, 1.0f, 0.5f);
		b_hue_slider.set_idle(hue_color);
		b_hue_slider.set_hover(hue_color * 0.5f);
		b_hue_slider.set_pressed(hue_color * 0.75f);

		b_saturation.set_idle(color);
		b_saturation.set_hover(color * 0.5f);
		b_saturation.set_pressed(color * 0.75f);

		// Set text
		t_color.text = common::sprintf("%.2f %.2f %.2f", color.x, color.y, color.z),
		t_color.anchor.x = r_hue.max.x - fr->size(t_color).x - 5.0f;

		if (ref)
			*ref = color;
	}

	// Moving the hue slider
	static void on_drag_hue(void *user, glm::vec2 dpos) {
		ColorPicker *cp = (ColorPicker *) user;

		auto b_hue = cp->b_hue_slider.shape();
		glm::vec2 pos = (b_hue->min + b_hue->max)/2.0f + dpos;

		cp->set_hue(pos);
	}

	static void on_click_hue(void *user, glm::vec2 pos) {
		ColorPicker *cp = (ColorPicker *) user;
		cp->set_hue(pos);
	}

	// Get saturation point
	glm::vec2 get_saturation_position() {
		return (b_saturation.shape()->min + b_saturation.shape()->max)/2.0f;
	}

	// Set saturation position
	void set_saturation_position(glm::vec2 pos) {
		glm::vec2 min = r_saturation.min;
		glm::vec2 max = r_saturation.max;

		// Clamp
		pos.x = glm::clamp(pos.x, min.x, max.x);
		pos.y = glm::clamp(pos.y, min.y, max.y);

		// Get uv coordinates
		glm::vec2 uv = glm::vec2 {
			(pos.x - min.x) / (max.x - min.x),
			(pos.y - min.y) / (max.y - min.y)
		};

		saturation = uv;

		// Set color
		glm::vec3 color = hue_to_rgb(hue, saturation.x, 1 - saturation.y);
		r_result.color = color;

		// Set position of button
		glm::vec2 size_2 = glm::vec2 {button_size, button_size}/2.0f;
		b_saturation.shape()->min = pos - size_2;
		b_saturation.shape()->max = pos + size_2;

		b_saturation.set_idle(color);
		b_saturation.set_hover(color * 0.5f);
		b_saturation.set_pressed(color * 0.75f);

		// Set text
		t_color.text = common::sprintf("%.2f %.2f %.2f", color.x, color.y, color.z),
		t_color.anchor.x = r_hue.max.x - fr->size(t_color).x - 5.0f;

		if (ref)
			*ref = color;
	}

	//  Moving the saturation button
	static void on_drag_saturation(void *user, glm::vec2 dpos) {
		// TODO: set position functio since we should also be
		// able  to click at a specific position
		ColorPicker *cp = (ColorPicker *) user;

		// TODO: min/max for ui elements and buttons
		glm::vec2 pos = cp->get_saturation_position();
		pos += dpos;

		cp->set_saturation_position(pos);
	}

	static void on_click_saturation(void *user, glm::vec2 pos) {
		ColorPicker *cp = (ColorPicker *) user;
		cp->set_saturation_position(pos);
	}

	struct Args {
		glm::vec2	min;
		glm::vec2	max;
		std::string	label;

		glm::vec3	*ref = nullptr;

		layers::FontRenderer *font_renderer;
	};

	ColorPicker() = default;

	// TODO: reference target for color picker
	ColorPicker(io::MouseEventQueue &mouse_events, const Args &args)
			: fr(args.font_renderer), ref(args.ref) {
		// Compute hue and saturation positions
		glm::vec2 hue_pos {-1.0f/0.0f};
		glm::vec2 sat_pos {1.0/0.0, -1};

		const float gap = 20.0f;

		glm::vec2 min = args.min + glm::vec2 {gap, gap};
		glm::vec2 max = args.max - glm::vec2 {gap, gap};

		float saturation_width = 3.0f * (max.x - min.x - gap)/4.0f;
		float hue_width = (max.x - min.x - gap) - saturation_width;

		// Saturation panel
		hue = 0.0f;
		r_saturation = ui::Rect(
			min, glm::vec2 {min.x + saturation_width, max.y},
			hue_to_rgb(0.0f, 0.0f, 0.0f),
			0.005f
		);

		r_saturation.shader_program.set_file("./shaders/ui/color_picker_square.frag");

		// Hue panel
		r_hue = ui::Rect(
			glm::vec2 {min.x + saturation_width + gap, min.y},
			glm::vec2 {max.x, max.y}
		);

		r_hue.shader_program.set_file("./shaders/ui/color_picker_hue.frag");


		// Set position if valid ref
		if (args.ref) {
			if (*args.ref == glm::vec3 {1}) {
				// Completely white
				sat_pos.x = -1.0f/0.0f;
			} else if (*args.ref == glm::vec3 {0}) {
				// Completely black
				sat_pos.y = 1.0f/0.0f;
			} else {
				float h = rgb_to_hue(*args.ref);

				hue_pos.y = h * (r_hue.max.y - r_hue.min.y)/360.0f + r_hue.min.y;

				float max = glm::max(args.ref->x, glm::max(args.ref->y, args.ref->z));
				float min = glm::min(args.ref->x, glm::min(args.ref->y, args.ref->z));

				float l = (max + min) / 2.0f;
				float delta = max - min;

				float s = delta / (1.0f - glm::abs(2.0f * l - 1.0f));

				sat_pos.x = s * (r_saturation.max.x - r_saturation.min.x) + r_saturation.min.x;
				sat_pos.y = (1 - l) * (r_saturation.max.y - r_saturation.min.y) + r_saturation.min.y;
			}
		}

		// Result panel
		float offset = 5.0f + button_size/2.0f;
		float result_size = 15.0f;
		r_result = ui::Rect(
			{min.x + 5.0f, max.y + offset},
			{min.x + result_size + 5.0f, max.y + result_size + offset},
			r_saturation.color,
			0.005f
		);

		// Text
		t_label = ui::Text(
			args.label,
			{min.x + result_size + 10.0f, max.y + offset},
			glm::vec3 {1.0},
			0.4f
		);

		t_color = ui::Text {
			"text",
			{max.x, max.y + offset},
			glm::vec3 {1.0},
			0.4f
		};

		t_color.anchor.x -= fr->size(t_color).x + 5.0f;

		// Buttons
		ui::Button::Args button_args = ui::Button::Args {};
		button_args.min = min;
		button_args.max = {min.x + saturation_width, max.y};
		button_args.on_click = {{this, on_click_saturation}};

		// TODO: pass ui element to button constructor
		b_square = ui::Button {
			mouse_events,
			button_args
		};

		button_args = ui::Button::Args {};
		button_args.radius = 0.01f;
		button_args.border_width = 0.005f;
		button_args.idle = glm::vec3 {0.5, 0.5, 1.0};
		button_args.on_drag = {{this, on_drag_saturation}};

		b_saturation = ui::Button {
			mouse_events,
			button_args
		};

		// TODO: we dont even need the sliders, just the areas...
		button_args = ui::Button::Args {};
		button_args.min = {min.x + saturation_width + gap, min.y};
		button_args.max = {max.x, max.y};
		button_args.on_click = {{this, on_click_hue}};

		b_square_hue = ui::Button {
			mouse_events,
			button_args
		};
				
		button_args = ui::Button::Args {};
		button_args.min = {min.x + saturation_width + gap, min.y};
		button_args.max = {max.x, max.y};
		button_args.radius = 0.002f;
		button_args.border_width = 0.005f;
		button_args.on_drag = {{this, on_drag_hue}};

		b_hue_slider = ui::Button {
			mouse_events,
			button_args
		};

		set_hue(hue_pos);
		set_saturation_position(sat_pos);
	}

	// No copy, only move
	ColorPicker(const ColorPicker &) = delete;
	ColorPicker &operator=(const ColorPicker &) = delete;

	// Move constructor
	// TODO: technical note: in order to be able to
	// click AND drag in sequence the square button (background)
	// must ALWAYS be initialized before the saturation button
	//
	// TODO: improve this by perhaps using a priority system
	// also make it easier in general to move buttons properly...,
	// maybe with a readdress method?
	ColorPicker(ColorPicker &&other)
			: fr(other.fr),
			hue(other.hue),
			saturation(other.saturation),
			ref(other.ref),
			r_saturation(other.r_saturation),
			r_hue(other.r_hue),
			r_result(other.r_result),
			t_label(other.t_label),
			t_color(other.t_color),
			b_square(std::move(other.b_square)),
			b_saturation(std::move(other.b_saturation)),
			b_square_hue(std::move(other.b_square_hue)),
			b_hue_slider(std::move(other.b_hue_slider)) {
		b_square.clear_handlers();
		b_square.add_on_click({this, on_click_saturation});

		b_saturation.clear_handlers();
		b_saturation.add_on_drag({this, on_drag_saturation});

		b_square_hue.clear_handlers();
		b_square_hue.add_on_click({this, on_click_hue});

		b_hue_slider.clear_handlers();
		b_hue_slider.add_on_drag({this, on_drag_hue});
	}

	ColorPicker &operator=(ColorPicker &&other) {
		fr = other.fr;
		hue = other.hue;
		saturation = other.saturation;
		ref = other.ref;

		r_saturation = other.r_saturation;
		r_hue = other.r_hue;
		r_result = other.r_result;

		t_label = other.t_label;
		t_color = other.t_color;

		b_square = std::move(other.b_square);
		b_saturation = std::move(other.b_saturation);
		b_square_hue = std::move(other.b_square_hue);
		b_hue_slider = std::move(other.b_hue_slider);

		b_square.clear_handlers();
		b_square.add_on_click({this, on_click_saturation});

		b_saturation.clear_handlers();
		b_saturation.add_on_drag({this, on_drag_saturation});

		b_square_hue.clear_handlers();
		b_square_hue.add_on_click({this, on_click_hue});

		b_hue_slider.clear_handlers();
		b_hue_slider.add_on_drag({this, on_drag_hue});

		return *this;
	}

	// Get color
	glm::vec3 get_color() const {
		return r_saturation.color;
	}

	std::vector <ui::Rect *> shapes() {
		return {
			&r_saturation, &r_hue, &r_result,
			b_saturation.shape(), b_hue_slider.shape()
		};
	}

	std::vector <ui::Text> texts() {
		return {t_label, t_color};
	}
};

}

}

#endif

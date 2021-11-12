#ifndef RECT_H_
#define RECT_H_

// Engine headers
#include "../init.hpp"
#include "pure_rect.hpp"

namespace mercury {

namespace ui {

class Rect : public Shape {
	PureRect	_fill;
	PureRect	_border;
	bool		_border_on;
	float		_border_width;

	// TODO: add curvature later
public:
	Rect();
	Rect(const glm::vec2 &, const glm::vec2 &,
		const glm::vec4 & = glm::vec4(1.0),
		float = 0.0,
		const glm::vec4 & = glm::vec4 {0, 0, 0, 1});

	void set_fill_color(const glm::vec4 &);
	void set_border_color(const glm::vec4 &);

	const PureRect &get_bounds() const;

	void draw(Shader &) override;
	glm::vec2 get_position() const override;
	void set_position(const glm::vec2 &) override;
	bool contains(const glm::vec2 &) const override;
};

}

}

#endif

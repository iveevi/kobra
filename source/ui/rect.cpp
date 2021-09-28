#include "../../include/ui/rect.hpp"

namespace mercury {

namespace ui {

// TODO: add Default Constructor for purerect
Rect::Rect(const glm::vec2 &p1, const glm::vec2 &p2,
		const glm::vec4 &fcolor, float border,
		const glm::vec4 &bcolor)
		: _fill(p1, p2), _border(p1, p2), _border_on(border != 0)
{
	if (border != 0) {
		_border = PureRect(
			p1 + glm::vec2 {-border, -border},
			p2 + glm::vec2 {border, border}
		);

		_border.set_color(bcolor);
	}

	_fill.set_color(fcolor);
}

void Rect::set_fill_color(const glm::vec4 &color)
{
	_fill.set_color(color);
}

void Rect::set_border_color(const glm::vec4 &color)
{
	_border.set_color(color);
}

void Rect::draw()
{
	if (_border_on)
		_border.draw();
	_fill.draw();
}

}

}

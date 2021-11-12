#include "../../include/ui/rect.hpp"

namespace mercury {

namespace ui {

Rect::Rect() {}

// TODO: add Default Constructor for purerect
Rect::Rect(const glm::vec2 &p1, const glm::vec2 &p2,
		const glm::vec4 &fcolor, float border,
		const glm::vec4 &bcolor)
		: _fill(p1, p2), _border(p1, p2),
		_border_on(border > 0), _border_width(border)
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

const PureRect &Rect::get_bounds() const
{
	return _fill;
}

void Rect::draw(Shader &shader)
{
	if (_border_on)
		_border.draw(shader);
	_fill.draw(shader);
}

glm::vec2 Rect::get_position() const
{
	return _fill.get_position();
}

void Rect::set_position(const glm::vec2 &pos)
{
	_fill.set_position(pos);
	if (_border_on) {
		_border.set_position(pos +
			glm::vec2 {
			-_border_width,
			-_border_width
		});
	}
}

bool Rect::contains(const glm::vec2 &mpos) const
{
	return _fill.contains(mpos);
}

}

}

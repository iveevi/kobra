#include "include/ui/pure_rect.hpp"
#include "include/common.hpp"

namespace mercury {

namespace ui {

PureRect::PureRect() {}

PureRect::PureRect(const glm::vec2 &p1, const glm::vec2 &p2)
		: _tl(p1), _br(p2), _color(1.0f)
{
	_genbufs(p1, p2);
}

PureRect::PureRect(float x1, float y1, float x2, float y2)
		: _tl(x1, y1), _br(x2, y2)
{
	_genbufs({x1, y1}, {x2, y2});
}

void PureRect::_genbufs(const glm::vec2 &p1, const glm::vec2 &p2)
{
	float h = UIElement::sheight;
	float vertices[] = {
		p2.x, h - p1.y, 0.0f,
		p2.x, h - p2.y, 0.0f,
		p1.x, h - p2.y, 0.0f,
		p1.x, h - p1.y, 0.0f,
	};

	unsigned int indices[] = {
		0, 3, 1,
		1, 3, 2
	};

	glGenVertexArrays(1, &_vao);
	glGenBuffers(1, &_vbo);
	glGenBuffers(1, &_ebo);
	glBindVertexArray(_vao);

	glBindBuffer(GL_ARRAY_BUFFER, _vbo);
	glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, _ebo);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW);

	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
	glEnableVertexAttribArray(0);
	glBindBuffer(GL_ARRAY_BUFFER, 0);
}

void PureRect::set_color(const glm::vec3 &color)
{
	_color = color;
}

float PureRect::get_width() const
{
	return _br.x - _tl.x;
}

float PureRect::get_height() const
{
	return _br.y - _tl.y;
}

const glm::vec2 &PureRect::get_tl() const
{
	return _tl;
}

void PureRect::draw()
{
	// Use and set shader properties
	UIElement::shader.use();
	UIElement::shader.set_vec3("shape_color", _color);
	glCheckError();

	glBindVertexArray(_vao);
	glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);
}

glm::vec2 PureRect::get_position() const
{
	return _tl;
}

void PureRect::set_position(const glm::vec2 &pos)
{
	glm::vec2 diag = _br - _tl;
	_tl = pos;
	_br = _tl + diag;

	_genbufs(_tl, _br);
}

bool PureRect::contains(const glm::vec2 &mpos) const
{
	return (mpos.x >= _tl.x && mpos.x <= _br.x)
		&& (mpos.y >= _tl.y && mpos.y <= _br.y);
}

}

}

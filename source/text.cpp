#include "../include/text.hpp"

// Standard headers
#include <stdexcept>

namespace mercury {

Text::Text(const std::string &str, float x, float y,
		float scale, const glm::vec3 &color)
		: _xpos(x), _ypos(y),
		_scale(scale), _color(color), _str(str)
{
	_alloc();
}

// Private methods
void Text::_alloc()
{
	// Create the buffers
	glGenVertexArrays(1, &_vao);
	glGenBuffers(1, &_vbo);

	// Allocate data
	glBindVertexArray(_vao);
	glBindBuffer(GL_ARRAY_BUFFER, _vbo);
	glBufferData(GL_ARRAY_BUFFER, sizeof(float) * 6 * 4, NULL, GL_DYNAMIC_DRAW);
	glEnableVertexAttribArray(0);
	glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, 4 * sizeof(float), 0);
	glBindBuffer(GL_ARRAY_BUFFER, 0);
	glBindVertexArray(0);
}

// Public methods
void Text::set_scale(float scale)
{
	_scale = scale;
}

void Text::set_str(const std::string &str)
{
	_str = str;
}

void Text::set_color(const glm::vec3 &color)
{
	_color = color;
}

// TODO: fix position from top left down rightwards
void Text::set_position(float x, float y)
{
	_xpos = x;
	_ypos = y;
}

void Text::draw()
{
	// Set current things
	Char::shader.use();
	Char::shader.set_vec3("text_color", _color);
	glBindVertexArray(_vao);

	// Construct the text
	float cxpos = _xpos;
	for (const auto &c : _str) {
		if (cmap.find(c) == cmap.end())
			throw std::runtime_error("No character " + std::string(1, c) + " in map...");

		Char ch = cmap[c];

		float xpos = cxpos + ch.bearing.x * _scale;
		float ypos = _ypos - (ch.size.y - ch.bearing.y) * _scale;

		float w = ch.size.x * _scale;
		float h = ch.size.y * _scale;

		// Update VBOs
		float vertices[6][4] = {
			{xpos,     ypos + h,   0.0f, 0.0f},
			{xpos,     ypos,       0.0f, 1.0f},
			{xpos + w, ypos,       1.0f, 1.0f},

			{xpos,     ypos + h,   0.0f, 0.0f},
			{xpos + w, ypos,       1.0f, 1.0f},
			{xpos + w, ypos + h,   1.0f, 0.0f}
		};

		// render glyph texture over quad
		glBindTexture(GL_TEXTURE_2D, ch.tid);
		// update content of VBO memory
		glBindBuffer(GL_ARRAY_BUFFER, _vbo);
		glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(vertices), vertices);

		glBindBuffer(GL_ARRAY_BUFFER, 0);
		// render quad
		glDrawArrays(GL_TRIANGLES, 0, 6);

		// Get pixels to next glyph
		cxpos += (ch.offset >> 6) * _scale;
	}

	glBindVertexArray(0);
	glBindTexture(GL_TEXTURE_2D, 0);
}

}

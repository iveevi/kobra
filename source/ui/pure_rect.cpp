#include "../../include/ui/pure_rect.hpp"

namespace mercury {

namespace ui {

const char *PureRect::default_vs = R"(
#version 330 core

layout (location = 0) in vec3 apos;

void main()
{
	gl_Position = vec4(apos.x, apos.y, apos.z, 1.0);
}
)";

const char *PureRect::default_fs = R"(
#version 330 core

out vec4 fragment_color;

uniform vec4 rect_color;

void main()
{
	fragment_color = rect_color; // vec4(1.0);
}
)";

PureRect::PureRect(const glm::vec2 &p1, const glm::vec2 &p2)
{
	_genbufs(
		transform(p1),
		transform(p2)
	);

	_shader = Shader::from_source(
		default_vs,
		default_fs
	);

	set_color(glm::vec4(1.0));
}

PureRect::PureRect(float x1, float y1, float x2, float y2)
{
	_genbufs(
		transform({x1, y1}),
		transform({x2, y2})
	);

	_shader = Shader::from_source(
		default_vs,
		default_fs
	);
}

void PureRect::_genbufs(const glm::vec2 &p1, const glm::vec2 &p2)
{
	float vertices[] = {
		p2.x, p1.y, 0.0f,
		p2.x, p2.y, 0.0f,
		p1.x, p2.y, 0.0f,
		p1.x, p1.y, 0.0f,
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

void PureRect::set_color(const glm::vec4 &color)
{
	_shader.use();
	_shader.set_vec4("rect_color", color);
}

void PureRect::draw() {
	_shader.use();

	glBindVertexArray(_vao);
	glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);
}

}

}

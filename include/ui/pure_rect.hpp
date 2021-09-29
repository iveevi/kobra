#ifndef PURE_RECT_H_
#define PURE_RECT_H_

// Engine headers
#include "../init.hpp"
#include "shape.hpp"

namespace mercury {

namespace ui {

class PureRect : public Shape {
	unsigned int 	_vao;
	unsigned int 	_vbo;
	unsigned int 	_ebo;

	Shader		_shader;

	static const char *default_vs;
	static const char *default_fs;

	void _genbufs(const glm::vec2 &, const glm::vec2 &);
public:
	PureRect(const glm::vec2 &, const glm::vec2 &);
	PureRect(float, float, float, float);

	void set_color(const glm::vec4 &);

	void draw() const override;
	bool contains(const glm::vec2 &) const override;
};

}

}

#endif

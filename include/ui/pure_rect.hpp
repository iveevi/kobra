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

	glm::vec2	_tl;
	glm::vec2	_br;

	glm::vec3	_color;

	void _genbufs(const glm::vec2 &, const glm::vec2 &);
public:
	PureRect();
	PureRect(const glm::vec2 &, const glm::vec2 &);
	PureRect(float, float, float, float);

	void set_color(const glm::vec3 &);

	// Getters
	float get_width() const;
	float get_height() const;
	const glm::vec2 &get_tl() const;

	void draw(Shader &) override;
	glm::vec2 get_position() const override;
	void set_position(const glm::vec2 &) override;
	bool contains(const glm::vec2 &) const override;
};

}

}

#endif

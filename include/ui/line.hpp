#ifndef LINE_H_
#define LINE_H_

// Engine headers
#include "include/varray.hpp"

namespace mercury {

namespace ui {

// Line class (with arrow options)
// TODO: add corresponding source file
class Line {
	SVA3 line;
public:
	// TODO: should this be private?
	glm::vec3 color;

	Line() {}

	// Create line between two points	
	Line(const glm::vec3 &p1, const glm::vec3 &p2,
		const glm::vec3 &col = glm::vec3(1.0f, 1.0f, 1.0f))
		: line({
			p1.x, p1.y, p1.z,
			p2.x, p2.y, p2.z
		}), color(col) {}

	Line(const std::vector <float> vertices) : line(vertices) {}

	// TODO: make an mvp class?
	void set_mvp(const glm::mat4 &model, const glm::mat4 &view, const glm::mat4 &projection) {
		Shader *shader = winman.cres.line_shader;

		// TODO: add a shader method for this	
		shader->use();
		shader->set_mat4("model", model);
		shader->set_mat4("view", view);
		shader->set_mat4("projection", projection);
	}

	enum EndType : uint8_t {
		NONE = 0,
		ARROW = 1
	};

	void draw(EndType start = NONE, EndType end = NONE) const {
		Shader *shader = winman.cres.line_shader;

		shader->use();
		shader->set_vec3("color", color);
		shader->set_int("start_mode", start);
		shader->set_int("end_mode", end);

		line.draw(GL_LINES);
	}
};

}

}

#endif
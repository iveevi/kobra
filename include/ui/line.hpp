#ifndef LINE_H_
#define LINE_H_

// Engine headers
#include "include/init.hpp"
#include "include/varray.hpp"

namespace mercury {

namespace ui {

// Line class (with arrow options)
// TODO: add corresponding source file
class Line : public SVA3 {
public:
	// Public enums
	enum EndType : uint8_t {
		NONE = 0,
		ARROW = 1
	};
private:
	EndType	_start_type;
	EndType	_end_type;
public:
	Line() {}

	// Create line between two points	
	Line(const glm::vec3 &p1, const glm::vec3 &p2,
		const glm::vec3 &color = glm::vec3(1.0f, 1.0f, 1.0f),
		EndType start_type = NONE, EndType end_type = NONE)
		: SVA3({
			p1.x, p1.y, p1.z,
			p2.x, p2.y, p2.z
		}, color, GL_LINES), _start_type(start_type),
		_end_type(end_type) {}

	Line(const std::vector <float> vertices) : SVA3(vertices) {}

	// TODO: make an mvp class?
	void set_mvp(const glm::mat4 &model, const glm::mat4 &view, const glm::mat4 &projection) {
		Shader *shader = winman.cres.line_shader;

		// TODO: add a shader method for this	
		shader->use();
		shader->set_mat4("model", model);
		shader->set_mat4("view", view);
		shader->set_mat4("projection", projection);
	}

	virtual void draw(Shader *shader) override {
		shader->use();
		shader->set_int("start_mode", _start_type);
		shader->set_int("end_mode", _end_type);

		// _line.draw(shader);
		SVA3::draw(shader);
	}
};

}

}

#endif
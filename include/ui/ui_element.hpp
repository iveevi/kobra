#ifndef UI_ELEMENT_H_
#define UI_ELEMENT_H_

// GLM
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

// Engine headers
#include "include/init.hpp"
#include "include/shader.hpp"
#include "include/common.hpp"

namespace mercury {

namespace ui {

class UIElement {
public:
	virtual void draw() = 0;

	virtual glm::vec2 get_position() const = 0;
	virtual void set_position(const glm::vec2 &) = 0;

	virtual void move(const glm::vec2 &dpos) {
		glm::vec2 pos = get_position();
		set_position(pos + dpos);
	}

	static Shader shader;
	static glm::mat4 projection;	// TODO: is this even necessary

	// Screen width and height
	// 	distinct from cwin data to
	// 	induce the notion of UI locality
	static float swidth;
	static float sheight;

	static void set_projection(
			float left, float right,
			float bottom, float top,
			float width, float height) {
		// Recaculate dimensions
		swidth = width;
		sheight = height;

		projection = glm::ortho(left, right, bottom, top);

		// Set projection matrix of all uies
		shader.use();
		glCheckError();

		shader.set_mat4("projection", projection);
		glCheckError();
	}
};

}

}

#endif

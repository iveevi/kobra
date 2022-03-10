#ifndef INPUT_H_
#define INPUT_H_

// GLFW headers
#include <GLFW/glfw3.h>

// GLM headers
#include <glm/glm.hpp>

namespace kobra {

namespace io {

// The input structure
// serves as a more continous
// and immediate input
class Input {
public:
	static constexpr int MAX_KEYS = 256;
private:
	// Associated window
	GLFWwindow *	window;
	int		states[MAX_KEYS];
	
	// Every time a key state is requested,
	// its state is updated
	// TODO: should event queues be notifed?
	void _getc(int key) {
		states[key] = glfwGetKey(window, key);
	}
public:
	// Default constructor
	Input() {}

	// Constructor needs window
	Input(GLFWwindow *window) : window(window) {
		for (int i = 0; i < MAX_KEYS; i++) {
			states[i] = glfwGetKey(window, i);
		}
	}

	// Check if key state is down
	bool is_key_down(int key) {
		_getc(key);
		return states[key] == GLFW_PRESS;
	}

	// Get mouse position
	glm::vec2 mouse_position() const {
		double x, y;
		glfwGetCursorPos(window, &x, &y);
		return glm::vec2(x, y);
	}
};

}

}

#endif

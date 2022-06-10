#include "../../include/app.hpp"
#include "../../include/io/event.hpp"

namespace kobra {

namespace io {

////////////////////
// GLFW callbacks //
////////////////////

void mouse_button_callback(GLFWwindow *window, int button, int action, int mods)
{
	MouseEvent event {
		.button = button,
		.action = action,
		.mods = mods
	};

	// Get mouse position
	double x;
	double y;

	glfwGetCursorPos(window, &x, &y);

	event.xpos = x;
	event.ypos = y;

	// Get event queue
	App::IO *io = (App::IO *) glfwGetWindowUserPointer(window);

	// Push event
	io->mouse_events.push(event);
}

void mouse_position_callback(GLFWwindow *window, double xpos, double ypos)
{
	MouseEvent event {
		.button = -1,
		.action = -1,
		.mods = -1
	};

	event.xpos = xpos;
	event.ypos = ypos;

	// Get event queue
	App::IO *io = (App::IO *) glfwGetWindowUserPointer(window);

	// Push event
	io->mouse_events.push(event);
}

void keyboard_callback(GLFWwindow *window, int key, int scancode, int action, int mods)
{
	KeyboardEvent event {
		.key = key,
		.scancode = scancode,
		.action = action,
		.mods = mods
	};

	// Get event queue
	App::IO *io = (App::IO *) glfwGetWindowUserPointer(window);

	// Push event
	io->keyboard_events.push(event);
}

}

}

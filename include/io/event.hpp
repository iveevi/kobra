#ifndef EVENT_H_
#define EVENT_H_

// Standard headers
#include <vector>
#include <functional>

// GLFW headers
#include <GLFW/glfw3.h>

namespace kobra {

namespace io {

// Types of events
struct MouseEvent {
	int button;
	int action;
	int mods;
	double xpos;
	double ypos;
};

struct KeyboardEvent {
	int key;
	int scancode;
	int action;
	int mods;
};

// Generic event queue type
template <class T>
class EventQueue {
public:
	// Event handler type
	using Handler = std::function <void (void *, const T &)>;
private:
	// List of subscribed callbacks
	using HandlerInstance = std::pair <Handler, void *>;

	std::vector <HandlerInstance> handlers;
public:
	// Subscribe to event
	void subscribe(const Handler &handler, void *user) {
		handlers.push_back({handler, user});
	}

	// Push event to handlers
	void push(const T &event) {
		for (auto &handler : handlers)
			handler.first(handler.second, event);
	}
};

// Aliases
using MouseEventQueue = EventQueue <MouseEvent>;
using KeyboardEventQueue = EventQueue <KeyboardEvent>;

// GLFW callbacks
void mouse_button_callback(GLFWwindow *, int, int, int);
void mouse_position_callback(GLFWwindow *, double, double);
void keyboard_callback(GLFWwindow *, int, int, int, int);

}

}

#endif

#ifndef EVENT_H_
#define EVENT_H_

// Standard headers
#include <vector>
#include <functional>

namespace mercury {

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
	using Handler = std::function <void (const T &)>;
private:
	// List of subscribed callbacks
	std::vector <Handler> handlers;
public:
	// Subscribe to event
	void subscribe(const Handler &handler) {
		handlers.push_back(handler);
	}

	// Push event to handlers
	void push(const T &event) {
		for (auto &handler : handlers)
			handler(event);
	}
};

// Aliases
using MouseEventQueue = EventQueue <MouseEvent>;
using KeyboardEventQueue = EventQueue <KeyboardEvent>;

}

}

#endif
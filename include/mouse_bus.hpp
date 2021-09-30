#ifndef MOUSE_BUS_H_
#define MOUSE_BUS_H_

// Standard headers
#include <typeindex>
#include <unordered_map>
#include <vector>

// GLM
#include <glm/glm.hpp>

// Engine headers
#include "handler.hpp"

// TODO: handlers directory?
namespace mercury {

// MouseBus class
// TODO: make a generic EventBus class...
class MouseBus {
public:
	enum Type : size_t {
		MOUSE_RELEASED,
		MOUSE_PRESSED,
		MOUSE_REPEATED
	};

	// Input data expectation
	struct Data {
		Type type;
		glm::vec2 pos;
	};
private:
	// NOTE: using an event_type "filtering"
	// could reduce the number of objects to broadcast this to
	std::vector <Handler *> _handlers;
public:
	// TODO: should store index that can be stored
	void subscribe(Handler *);

	template <class T>
	void subscribe(T *, typename MemFtnHandler <T> ::MemFtn);

	void publish(Data);
};

// Template functions
template <class T>
void MouseBus::subscribe(T *instance, typename MemFtnHandler <T> ::MemFtn mftn)
{
	subscribe(new MemFtnHandler <T> (instance, mftn));
}

}

#endif

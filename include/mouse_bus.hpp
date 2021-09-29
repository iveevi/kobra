#ifndef MOUSE_BUS_H_
#define MOUSE_BUS_H_

// Standard headers
#include <typeindex>
#include <unordered_map>
#include <vector>

// GLM
#include <glm/glm.hpp>

// TODO: handlers directory?
namespace mercury {

// TODO: put in generic handler header
class Handler {
	// Pass the pointer to the data
	// (not on the heap) and check
	// size_t value at the pointer
	// as "inheritance"
	virtual void call(size_t *) const = 0;
public:
	inline void run(size_t *data) {
		return call(data);
	}

};

// TODO: should we cast to reduce casting overhead within functions?
template <class T>
class MemFtnHandler : public Handler {
public:
	using MemFtn = void (T::*)(size_t *);
private:
	T *	_obj;
	MemFtn	_mftn;
public:
	MemFtnHandler(T *instance, MemFtn mftn)
		: _obj(instance), _mftn(mftn) {}

	void call(size_t *data) const override {
		(_obj->*_mftn)(data);
	}
};

// MouseBus class
// TODO: make a generic EventBus class...
class MouseBus {
public:
	enum Type : size_t {
		MOUSE_PRESSED,
		MOUSE_RELEASED
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

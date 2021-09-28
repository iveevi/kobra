#ifndef EVENT_HANDLER_H_
#define EVENT_HANDLER_H_

namespace mercury {

enum EventType {
	BUTTON_PRESSED,
	BUTTON_RELEASED
};

// Queue class
class EventQueue {
public:
	// Public structs
	struct Event {
		void *source;
		EventType type;
	};
public:
	void publish();
};

}

#endif

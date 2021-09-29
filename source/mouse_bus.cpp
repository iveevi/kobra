#include "../include/mouse_bus.hpp"

// TODO: remove
#include <iostream>

namespace mercury {

void MouseBus::subscribe(Handler *handler)
{
	_handlers.push_back(handler);
}

void MouseBus::publish(Data data)
{
	for (Handler *handler : _handlers)
		handler->run((size_t *) &data);
}

}

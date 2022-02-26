#ifndef LAYER_H_
#define LAYER_H_

// Standard headers
#include <vector>

// Engine headers
#include "gui.hpp"

namespace mercury {

namespace gui {

// Contains a set of
// GUI objects to render
class Layer {
	std::vector <ObjectPtr>	_objects;
};

}

}

#endif
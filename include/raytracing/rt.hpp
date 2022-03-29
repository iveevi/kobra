#ifndef KOBRA_RT_H_
#define KOBRA_RT_H_

// Standard headers
#include <memory>

namespace kobra {

namespace rt {

// Element type
struct _element {
	// Destructor
	virtual ~_element() = default;

	// Render
	virtual void render() = 0;
};

// Memory safe
using Element = std::shared_ptr <_element>;

}

}

#endif

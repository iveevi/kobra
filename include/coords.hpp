#ifndef COORDS_H_
#define COORDS_H_

// Engine headers
#include "vec.hpp"

namespace kobra {

namespace coordinates {

// Screen structure
struct Screen {
	float x;
	float y;

	size_t width;
	size_t height;

	// Convert Screen to NDC
	glm::vec2 to_ndc() const {
		return {
			(2.0f * x) / width - 1.0f,
			(2.0f * y) / height - 1.0f
		};
	}

	// Scale by width
	glm::vec2 to_unit() const {
		return {
			x/width,
			y/height
		};
	}
};

}

}

#endif

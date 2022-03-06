#ifndef AREA_H_
#define AREA_H_

// GLM headers
#include <glm/glm.hpp>

// Engine headers
#include "../app.hpp"

namespace mercury {

namespace gui {

// Area class for detecting mouse events
// and other similar point based events
struct Area {
	// Based on window coordinates
	virtual bool contains(int, int) = 0;
	virtual bool contains(const glm::vec2 &) = 0;
};

// Rectangle area
struct RectArea : Area {
	int x, y, w, h;

	RectArea() {}
	RectArea(int x, int y, int w, int h)
		: x(x), y(y), w(w), h(h) {}

	bool contains(int x, int y) {
		return (x >= this->x) && (x <= this->x + this->w)
			&& (y >= this->y) && (y <= this->y + this->h);
	}

	bool contains(const glm::vec2 &p) {
		return contains(p.x, p.y);
	}
};

}

}

#endif

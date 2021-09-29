#ifndef SHAPE_H_
#define SHAPE_H_

// GLM
#include <glm/glm.hpp>

namespace mercury {

namespace ui {

class Shape {
public:
	virtual void draw() const = 0;
	virtual bool contains(const glm::vec2 &) const = 0;
};

}

}

#endif

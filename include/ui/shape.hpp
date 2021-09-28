#ifndef SHAPE_H_
#define SHAPE_H_

namespace mercury {

namespace ui {

class Shape {
public:
	virtual bool contains(const glm::vec2 &) const = 0;
};

}

}

#endif

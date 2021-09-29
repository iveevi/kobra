#ifndef UI_LAYER_H_
#define UI_LAYER_H_

// Standard headers
#include <vector>

// Engine headers
#include "ui_element.hpp"

namespace mercury {

namespace ui {

class UILayer {
	std::vector <UIElement *> _elements;
public:
	void add_element(UIElement *);

	void draw() const;
};

}

}

#endif

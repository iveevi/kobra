#ifndef UI_ELEMENT_H_
#define UI_ELEMENT_H_

namespace mercury {

namespace ui {

class UIElement {
public:
	virtual void draw() const = 0;
};

}

}

#endif

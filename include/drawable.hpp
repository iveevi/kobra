#ifndef DRAWABLE_H_
#define DRAWABLE_H_

// Engine headers
#include "include/shader.hpp"

namespace mercury {

// Drawable abstract class
class Drawable {
public:
        virtual void draw(Shader *) = 0;
};

}

#endif
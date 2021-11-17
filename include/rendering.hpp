#ifndef RENDERING_H_
#define RENDERING_H_

// Standard headers
#include <unordered_map>

// Engine headers.
#include "include/logger.hpp"
#include "include/drawable.hpp"
#include "include/shader.hpp"

namespace mercury {

namespace rendering {

// Rendering daemon structure
class Daemon {
        // Map of mesh to shader
        std::unordered_map <Drawable *, Shader *>       _map;

        // All drawables
        std::vector <Drawable *>                        _drawables;
        
        // Render single drawable
        void _render(Drawable *);
public:
        void add(Drawable *, Shader *);

        void render();
};

}

}

#endif
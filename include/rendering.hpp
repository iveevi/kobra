#ifndef RENDERING_H_
#define RENDERING_H_

// Standard headers
#include <unordered_map>

// GLM headers
#include <glm/glm.hpp>

// Engine headers.
#include "include/logger.hpp"
#include "include/drawable.hpp"
#include "include/shader.hpp"

namespace mercury {

namespace rendering {

// Rendering daemon structure
class Daemon {
        // Drawable map
        template <class Resource>
        using DMap = std::unordered_map <Drawable *, Resource>;

        // Default model: TODO: make a transform
        glm::mat4			_default_model = glm::mat4(1.0);

        // Drawable maps
        DMap <Shader *>			_shader_map;
        DMap <glm::mat4 *>		_model_map;             // TODO: change to transforms map

        // All drawables
        std::vector <Drawable *>        _drawables;
        
        // Render single drawable
        void _render(Drawable *);
public:
        void add(Drawable *, Shader *);
        void add(Drawable *, Shader *, glm::mat4 *);

        void render();
};

}

}

#endif
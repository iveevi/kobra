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
#include "include/transform.hpp"

namespace mercury {

namespace rendering {

// Rendering daemon structure
class Daemon {
        // Drawable map
        template <class Resource>
        using DMap = std::unordered_map <Drawable *, Resource>;

        // Default model: TODO: make a transform
        Transform			_default_transform;

        // Drawable maps
        DMap <Shader *>			_shader_map;
        DMap <Transform *>		_transform_map;

        // All drawables
        std::vector <Drawable *>        _drawables;
        
        // Render single drawable
        void _render(Drawable *);
public:
        void add(Drawable *, Shader *);
        void add(Drawable *, Shader *, Transform *);

        void render();
};

}

}

#endif
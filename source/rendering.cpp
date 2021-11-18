// Engine headers
#include "include/rendering.hpp"

namespace mercury {

namespace rendering {

// Add drawable and corresponding resources to the maps
void Daemon::add(Drawable* drawable, Shader* shader)
{
        // Add drawable to the drawable map
        _shader_map[drawable] = shader;
        _transform_map[drawable] = &_default_transform;
        _drawables.push_back(drawable);
}

void Daemon::add(Drawable* drawable, Shader* shader, Transform *transform)
{
        // Add drawable to the drawable map
        _shader_map[drawable] = shader;
        _transform_map[drawable] = transform;
        _drawables.push_back(drawable);
}

// Render a single drawable
void Daemon::_render(Drawable* drawable)
{
        // Get shader from the map
        Shader* shader = _shader_map[drawable];

        // Set transforms
        shader->use();
        shader->set_mat4("model", _transform_map[drawable]->model());

        // Draw the drawable
        drawable->draw(shader);
}

// Render all
void Daemon::render()
{
        // Render all drawables
        for (Drawable* drawable : _drawables)
                _render(drawable);
}

}

}
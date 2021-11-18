// Engine headers
#include "include/rendering.hpp"

namespace mercury {

namespace rendering {

// Add drawable and corresponding resources to the maps
void Daemon::add(Drawable* drawable, Shader* shader)
{
        // Add drawable to the drawable map
        _shader_map[drawable] = shader;
        _model_map[drawable] = &_default_model;
        _drawables.push_back(drawable);
}

void Daemon::add(Drawable* drawable, Shader* shader, glm::mat4 *model)
{
        // Add drawable to the drawable map
        _shader_map[drawable] = shader;
        _model_map[drawable] = model;
        _drawables.push_back(drawable);
}

// Render a single drawable
void Daemon::_render(Drawable* drawable)
{
        // Get shader from the map
        Shader* shader = _shader_map[drawable];

        // Set transforms
        shader->use();
        shader->set_mat4("model", *_model_map[drawable]);

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
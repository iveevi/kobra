// Engine headers
#include "include/rendering.hpp"

namespace mercury {

namespace rendering {

// Add drawable and corresponding shader to the shader map
void Daemon::add(Drawable* drawable, Shader* shader)
{
        // Add drawable to the drawable map
        _map[drawable] = shader;
        _drawables.push_back(drawable);
}

// Render a single drawable
void Daemon::_render(Drawable* drawable)
{
        // Get shader from the map
        Shader* shader = _map[drawable];

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
// GLM headers
#include <glm/gtc/matrix_transform.hpp>

// Engine headers
#include "include/transform.hpp"

namespace mercury {

// Sets translation and rotation to zero, scale to 1
Transform::Transform() : translation(glm::vec3(0.0f)),
                erot(glm::vec3(0.0f)), scale(glm::vec3(1.0f)) {}

Transform::Transform(const glm::vec3 &t, const glm::vec3 &r,
                const glm::vec3 &s) : translation(t),
                erot(r), scale(s) {}

// Methods
void Transform::move(const glm::vec3 &dpos)
{
        translation += dpos;
}

void Transform::rotate(const glm::vec3 &drot)
{
        erot += drot;
}

// Returns the model matrix
glm::mat4 Transform::model() const
{
        static const glm::vec3 x_axis {1.0f, 0.0f, 0.0f};
        static const glm::vec3 y_axis {0.0f, 1.0f, 0.0f};
        static const glm::vec3 z_axis {0.0f, 0.0f, 1.0f};
        
        // Create and return the model matrix
        glm::mat4 model(1.0f);
        model = glm::translate(model, translation);
        model = glm::rotate(model, glm::radians(erot.x), x_axis);
        model = glm::rotate(model, glm::radians(erot.y), y_axis);
        model = glm::rotate(model, glm::radians(erot.z), z_axis);
        model = glm::scale(model, scale);
        return model;
}

}
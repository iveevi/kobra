// GLM headers
#include <glm/gtc/matrix_transform.hpp>

// Engine headers
#include "include/logger.hpp"
#include "include/transform.hpp"

namespace mercury {

// Sets translation and rotation to zero, scale to 1
Transform::Transform() : translation(glm::vec3(0.0f)),
                scale(glm::vec3(1.0f)) {}

Transform::Transform(const glm::vec3 &t, const glm::vec3 &r,
                const glm::vec3 &s) : translation(t),
                scale(s), orient(r) {}

// Methods
void Transform::move(const glm::vec3 &dpos)
{
        translation += dpos;
}

void Transform::rotate(const glm::vec3 &drot)
{
        orient = glm::quat(drot) * orient;
}

void Transform::rotate(const glm::quat &drot)
{
        orient = drot * orient;
        glm::normalize(orient);
}

// Returns the model matrix
glm::mat4 Transform::model() const
{
        // Create and return the model matrix
        glm::mat4 t = glm::translate(glm::mat4(1.0f), translation);
        glm::mat4 r = glm::toMat4(orient);
        glm::mat4 s = glm::scale(glm::mat4(1.0f), scale);
        return t * r * s;
}

}
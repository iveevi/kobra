// GLM headers
#include <glm/gtc/matrix_transform.hpp>

// Engine headers
#include "include/logger.hpp"
#include "include/transform.hpp"

namespace mercury {

// Sets translation and rotation to zero, scale to 1
Transform::Transform() : translation {0, 0, 0},
                scale {1, 1, 1} {}

Transform::Transform(const vec3 &t, const vec3 &r,
                const vec3 &s) : translation(t),
                scale(s), orient(glm::quat(r)) {}

// Methods
void Transform::move(const vec3 &dpos)
{
        translation += dpos;
}

void Transform::rotate(const vec3 &drot)
{
        orient = glm::quat(drot) * glm::quat(orient);
}

void Transform::rotate(const quat &drot)
{
        orient = glm::normalize(glm::quat(drot) * glm::quat(orient));
}

// Returns the model matrix
glm::mat4 Transform::model() const
{
        // Create and return the model matrix
        glm::mat4 t = glm::translate(glm::mat4(1.0f), glm::vec3(translation));
        glm::mat4 r = glm::toMat4(glm::quat(orient));
        glm::mat4 s = glm::scale(glm::mat4(1.0f), glm::vec3(scale));
        return t * r * s;
}

}
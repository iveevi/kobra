#ifndef TRANSFORM_H_
#define TRANSFORM_H_

// Use glm for matrix math
#include <glm/glm.hpp>

class Transform {
        glm::mat4 mat;
        glm::mat4 inv;
public:
        // Constuctors
        Transform() :mat(1), inv(1) {}
        Transform(const glm::mat4 &m) : mat(m), inv(glm::inverse(m)) {}
        Transform(const glm::mat4 &m, const glm::mat4 &i) : mat(m), inv(i) {}

        // Operators
        Transform operator*(const Transform &t) const {
                return Transform(mat * t.mat, t.inv * inv);
        }

        // Multiply as a vector
        glm::vec3 mul_vec(const glm::vec3 &v) const {
                return glm::vec3(mat * glm::vec4(v, 1.0f));
        }

        // Multiply as a point
        glm::vec3 mul_pt(const glm::vec3 &v) const {
                return glm::vec3(mat * glm::vec4(v, 1.0f));
        }

        // Movement and rotation
        void translate(const glm::vec3 &v) {
                mat = glm::translate(mat, v);
                inv = glm::translate(inv, -v);
        }

        // Set orientation with lookAt
        void look_at(const glm::vec3 &front, const glm::vec3 &up) {
                glm::mat4 rot = glm::lookAt(glm::vec3(0), front, up);
                mat = rot * mat;
                inv = glm::inverse(mat);
        }

        // Invert the transform
        Transform invert() const {
                glm::mat4 i_mat = glm::inverse(mat);
                glm::mat4 i_inv = glm::inverse(inv);

                return Transform(i_mat, i_inv);
        }
};

#endif

#ifndef CORE_H_
#define CORE_H_

// Standard headers
#include <vector>

// GLM headers
#include <glm/glm.hpp>

// Aligned structures
// TODO: remove?
struct alignas(16) aligned_vec3 {
	glm::vec3 data;

	aligned_vec3() {}
	aligned_vec3(const glm::vec3 &d) : data(d) {}
};

// TODO: move to another place
struct alignas(16) aligned_vec4 {
	glm::vec4 data;

	aligned_vec4() {}
	aligned_vec4(const glm::vec4 &d) : data(d) {}

	aligned_vec4(const glm::vec3 &d) : data(d, 0.0f) {}
	aligned_vec4(const glm::vec3 &d, float w) : data(d, w) {}
};

// Buffer type aliases
using Buffer = std::vector <aligned_vec4>;
using Indices = std::vector <uint32_t>;

#endif

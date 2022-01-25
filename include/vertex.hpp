#ifndef VERTEX_H_
#define VERTEX_H_

// GLFW and Vulkan headers
#define GLFW_INCLUDE_VULKAN

#include <GLFW/glfw3.h>

// GLM headers
#include <glm/glm.hpp>

namespace mercury {

// Vertex types
//	separated using bitwise operators
enum VertexType : uint32_t {
	POSITION = 0,
	NORMAL = 1,
	COLOR = 2,
};

// Defalut vertex type
constexpr VertexType DEFAULT_VERTEX_TYPE = POSITION | NORMAL | COLOR;

// Default vertex structure
// 	default uses position, color, and normal
template <VertexType type = DEFAULT_VERTEX_TYPE>
struct Vertex {
	glm::vec3 position;
	glm::vec3 normal;
	glm::vec3 color;
};

}

#endif

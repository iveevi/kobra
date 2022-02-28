#ifndef RECT_H_
#define RECT_H_

// Standard headers
#include <array>

// GLM headers
#include <glm/glm.hpp>
#include <vulkan/vulkan_core.h>

// Engine headers
#include "gui.hpp"

namespace mercury {

namespace gui {

class Rect : public Object {
	// Store normalized coordinates
	//	(0, 0) is the top-left corner
	//	(1, 1) is the bottom-right corner
	// TODO: store as vec4 instead
	glm::vec2 min;
	glm::vec2 max;
	glm::vec3 color;
public:
	// Constructors
	Rect() : min(0.0f), max(0.0f) {}
	Rect(const glm::vec2 &min, const glm::vec2 &max, const glm::vec3 &c = glm::vec3 {1.0})
		: min(min), max(max), color(c) {}
	
	// Upload vertex data to GPU
	void upload(VertexBuffer &vb, IndexBuffer &ib) const {
		// Create vertex data
		std::array <Vertex, 4> vertices {
			Vertex { min, color },
			Vertex { glm::vec2 { min.x, max.y }, color },
			Vertex { max, color },
			Vertex { glm::vec2 { max.x, min.y }, color }
		};

		uint32_t vsize = vb.push_size();
		std::array <uint32_t, 6> indices {
			vsize, vsize + 2, vsize + 1,
			vsize, vsize + 3, vsize + 2
		};
		
		// Upload vertex data
		vb.push_back(vertices);
		ib.push_back(indices);
	}
};

}

}

#endif
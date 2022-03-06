#ifndef RECT_H_
#define RECT_H_

// Standard headers
#include <array>

// GLM headers
#include <glm/glm.hpp>
#include <vulkan/vulkan_core.h>

// Engine headers
#include "../app.hpp"
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

	// TODO: depreciate and replace with NDC structure
	Rect(const glm::vec4 &bounds, const glm::vec3 &c = glm::vec3 {1.0})
		: min(bounds.x, bounds.y), max(bounds.z, bounds.w), color(c) {}
	Rect(const glm::vec2 &min, const glm::vec2 &max, const glm::vec3 &c = glm::vec3 {1.0})
		: min(min), max(max), color(c) {}

	Rect(const App::Window &wctx, float x, float y, float w, float h, const glm::vec3 &c = glm::vec3 {1.0})
		: color(c) {
		float nx = 2 * x / wctx.width - 1;
		float ny = 2 * y / wctx.height - 1;

		float nw = 2 * w / wctx.width;
		float nh = 2 * h / wctx.height;

		min = glm::vec2 {nx, ny};
		max = glm::vec2 {nx + nw, ny + nh};
	}
	
	// Upload vertex data to GPU
	// TODO: a common structure for rendering
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

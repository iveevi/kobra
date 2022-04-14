#include "../../include/gui/gui.hpp"

namespace kobra {

namespace gui {

///////////////////////////
// Vertex static methods //
///////////////////////////

// Get vertex binding description
Vulkan::VB Vertex::vertex_binding()
{
	return Vulkan::VB {
		.binding = 0,
		.stride = sizeof(Vertex),
		.inputRate = VK_VERTEX_INPUT_RATE_VERTEX
	};
}

// Get vertex attribute descriptions
std::vector <Vulkan::VA> Vertex::vertex_attributes()
{
	return {
		Vulkan::VA {
			.location = 0,
			.binding = 0,
			.format = VK_FORMAT_R32G32_SFLOAT,
			.offset = offsetof(Vertex, pos)
		},

		Vulkan::VA {
			.location = 1,
			.binding = 0,
			.format = VK_FORMAT_R32G32B32_SFLOAT,
			.offset = offsetof(Vertex, color)
		}
	};
}

///////////////////////////////////
// Element methods and functions //
///////////////////////////////////


// Wrapper function to render
void _element::render_element(RenderPacket &packet)
{
	// Render this
	render(packet);

	// Render all children
	for (auto &child : children)
		child->render_element(packet);
}

// Bounding box for a list of elements
glm::vec4 get_bounding_box(const std::vector <_element *> &elements)
{
	// Throw on empty list
	if (elements.empty())
		throw std::runtime_error("Empty list of elements");

	// Initialize bounding box
	glm::vec4 bounding_box = glm::vec4 {
		std::numeric_limits <float>::max(),
		std::numeric_limits <float>::max(),
		-std::numeric_limits <float>::max(),
		-std::numeric_limits <float>::max()
	};

	// Loop through all elements
	for (auto &element : elements) {
		// Get the bounding box
		auto bb = element->bounding_box();

		// Update bounding box
		bounding_box.x = std::min(bounding_box.x, bb.x);
		bounding_box.y = std::min(bounding_box.y, bb.y);
		bounding_box.z = std::max(bounding_box.z, bb.z);
		bounding_box.w = std::max(bounding_box.w, bb.w);
	}

	// Return bounding box
	return bounding_box;
}

glm::vec4 get_bounding_box(const std::vector <Element> &elements)
{
	// Throw on empty list
	if (elements.empty())
		throw std::runtime_error("Empty list of elements");

	// Initialize bounding box
	glm::vec4 bounding_box = glm::vec4 {
		std::numeric_limits <float>::max(),
		std::numeric_limits <float>::max(),
		-std::numeric_limits <float>::max(),
		-std::numeric_limits <float>::max()
	};

	// Loop through all elements
	for (auto &element : elements) {
		// Get the bounding box
		auto bb = element->bounding_box();

		// Update bounding box
		bounding_box.x = std::min(bounding_box.x, bb.x);
		bounding_box.y = std::min(bounding_box.y, bb.y);
		bounding_box.z = std::max(bounding_box.z, bb.z);
		bounding_box.w = std::max(bounding_box.w, bb.w);
	}

	// Return bounding box
	return bounding_box;
}

}

}

#ifndef GUI_H_
#define GUI_H_

// GLM headers
#include <glm/glm.hpp>

// Standard headers
#include <memory>
#include <vector>

// Engine headers
#include "../backend.hpp"
#include "../object.hpp"

namespace kobra {

namespace gui {

// Forward declarations
class Layer;

// Vertex data
struct Vertex {
	glm::vec2 pos;
	glm::vec3 color;

	// Get Vulkan info for vertex
	static vk::VertexInputBindingDescription
		vertex_binding();

	static std::vector <vk::VertexInputAttributeDescription>
		vertex_attributes();
};

// Aliases
// using VertexBuffer = BufferManager <Vertex>;
// using IndexBuffer = BufferManager <uint32_t>;

// RenderPacket structure contains
// 	the data needed to render all
// 	the GUI elements in a Layer object
struct RenderPacket {
	const vk::raii::CommandBuffer &cmd;
	const vk::raii::PipelineLayout &sprite_layout;
};

// Latching packet
struct LatchingPacket {
	Layer *layer;
};

// Abstract GUI element type
struct _element : virtual public Object {
	// Virtual destructor
	virtual ~_element() {}

	// Child elements
	std::vector <std::shared_ptr <_element>> children;

	// Pure virtual functions
	virtual void latch(LatchingPacket &packet) = 0;
	virtual void render(RenderPacket &) = 0;

	// Position and bounding box
	virtual glm::vec2 position() const = 0;
	virtual glm::vec4 bounding_box() const = 0;

	// Override object virtual methods for now
	virtual float intersect(const Ray &) const {
		return -1.0f;
	}

	virtual glm::vec3 center() const {
		return glm::vec3(0.0f);
	}

	virtual void save(std::ofstream &) const {
		// Do nothing
	}

	// Wrapper function to render
	void render_element(RenderPacket &);
};

// Aliases
using Element = std::shared_ptr <_element>;

// Bounding box for a list of elements
glm::vec4 get_bounding_box(const std::vector <_element *> &);
glm::vec4 get_bounding_box(const std::vector <Element> &);

}

}

#endif

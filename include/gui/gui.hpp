#ifndef GUI_H_
#define GUI_H_

// GLM headers
#include <glm/glm.hpp>

// Standard headers
#include <memory>
#include <vector>

// Engine headers
#include "../backend.hpp"
#include "../buffer_manager.hpp"

namespace kobra {

// Core definitions for GUI
namespace gui {

// Vertex data
struct Vertex {
	glm::vec2 pos;
	glm::vec3 color;

	// Get Vulkan info for vertex
	static Vulkan::VB vertex_binding();
	static std::vector <Vulkan::VA> vertex_attributes();
};

// Aliases
using VertexBuffer = BufferManager <Vertex>;
using IndexBuffer = BufferManager <uint32_t>;

// RenderPacket structure contains
// 	the data needed to render all
// 	the GUI elements in a Layer object
struct RenderPacket {
	// Rectangles
	struct {
		VertexBuffer *vb;
		IndexBuffer *ib;
	} rects;

	// TODO: text renders so that text can become an object

	// Methods
	void reset();
	void sync();
};

// Abstract GUI element type
struct _element {
	// Virtual destructor
	virtual ~_element() {}

	// Child elements
	std::vector <std::shared_ptr <_element>> children;

	// Pure virtual function to render
	virtual void render(RenderPacket &) = 0;

	// Position and bounding box
	virtual glm::vec2 position() const = 0;
	virtual glm::vec4 bounding_box() const = 0;

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

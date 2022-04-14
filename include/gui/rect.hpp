#ifndef RECT_H_
#define RECT_H_

// Standard headers
#include <array>

// GLM headers
#include <glm/glm.hpp>
#include <vulkan/vulkan_core.h>

// Engine headers
#include "../app.hpp"
#include "../coords.hpp"
#include "gui.hpp"

namespace kobra {

namespace gui {

class Rect : public _element {
	// Store normalized coordinates
	//	(0, 0) is the top-left corner
	//	(1, 1) is the bottom-right corner
	// TODO: store as vec4 instead
	glm::vec2 min = glm::vec2 {0.0f, 0.0f};
	glm::vec2 max = glm::vec2 {0.0f, 0.0f};
	glm::vec3 color = glm::vec3 {0.0f, 0.0f, 0.0f};

	// Vertex and index buffers
	VertexBuffer	_vb;
	IndexBuffer	_ib;

	// Update buffer contents
	void _update_buffers() {
		// Fill buffers
		std::array <uint, 6> indices {
			0, 1, 2,
			2, 3, 0
		};

		std::array <Vertex, 4> vertices {
			Vertex {min,				color},
			Vertex {glm::vec2 {min.x, max.y},	color},
			Vertex {max,				color},
			Vertex {glm::vec2 {max.x, min.y},	color}
		};

		// Upload vertex data
		_vb.reset_push_back();
		_ib.reset_push_back();

		_vb.push_back(vertices);
		_ib.push_back(indices);

		_vb.sync_upload();
		_ib.sync_upload();
	}

	// Initialize buffers
	void _init_buffers(const Vulkan::Context &context) {
		// Settings
		BFM_Settings vb_settings {
			.size = 4,
			.usage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
			.usage_type = BFM_WRITE_ONLY
		};

		BFM_Settings ib_settings {
			.size = 6,
			.usage = VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
			.usage_type = BFM_WRITE_ONLY
		};

		// Initialize buffers
		_vb = VertexBuffer(context, vb_settings);
		_ib = IndexBuffer(context, ib_settings);

		// Update buffers
		_update_buffers();
	}
public:
	// Type name
	static constexpr char object_type[] = "GUI-Rect";

	// Default constructor
	Rect() = default;

	// Constructors
	Rect(const Vulkan::Context &context, const glm::vec4 &bounds, const glm::vec3 &c = glm::vec3 {1.0})
			: Object(object_type), min(bounds.x, bounds.y),
			max(bounds.z, bounds.w), color(c) {
		// Initialize buffers
		_init_buffers(context);
	}

	Rect(const Vulkan::Context &context, const coordinates::Screen &sc,
			const coordinates::Screen &size,
			const glm::vec3 &c = glm::vec3 {1.0})
			: Object(object_type), color(c) {
		// Calulcate bounds
		glm::vec2 ndc = sc.to_ndc();
		glm::vec2 ndc_size = 2.0f * size.to_unit();

		min = ndc;
		max = ndc + ndc_size;

		// Initialize buffers
		_init_buffers(context);
	}

	// Setters
	void set_bounds(const glm::vec4 &bounds) {
		min = glm::vec2 {bounds.x, bounds.y};
		max = glm::vec2 {bounds.z, bounds.w};
		_update_buffers();
	}

	// Virtual methods
	glm::vec2 position() const override {
		return min;
	}

	glm::vec4 bounding_box() const override {
		return glm::vec4 {min.x, min.y, max.x, max.y};
	}

	// Latch to a layer
	void latch(LatchingPacket &lp) override {}

	// Render
	void render(RenderPacket &packet) override {
		// Draw
		VkBuffer	vbuffers[] = {_vb.vk_buffer()};
		VkDeviceSize	offsets[] = {0};

		vkCmdBindVertexBuffers(packet.cmd,
			0, 1, vbuffers, offsets
		);

		vkCmdBindIndexBuffer(packet.cmd,
			_ib.vk_buffer(), 0,
			VK_INDEX_TYPE_UINT32
		);

		vkCmdDrawIndexed(packet.cmd,
			_ib.push_size(),
			1, 0, 0, 0
		);
	}
};

}

}

#endif

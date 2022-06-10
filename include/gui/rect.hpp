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
	BufferData _vertex_buffer = nullptr;
	BufferData _index_buffer = nullptr;

	// Update buffer contents
	void _update_buffers() {
		// Fill buffers
		std::vector <uint> indices {
			0, 1, 2,
			2, 3, 0
		};

		std::vector <Vertex> vertices {
			Vertex {min,				color},
			Vertex {glm::vec2 {min.x, max.y},	color},
			Vertex {max,				color},
			Vertex {glm::vec2 {max.x, min.y},	color}
		};

		// Upload data
		_vertex_buffer.upload(vertices);
		_index_buffer.upload(indices);
	}

	// Initialize buffers
	void _init_buffers(const vk::raii::PhysicalDevice &phdev,
			const vk::raii::Device &device) {
		// Buffer sizes
		vk::DeviceSize vertex_buffer_size = 4 * sizeof(Vertex);
		vk::DeviceSize index_buffer_size = 6 * sizeof(uint);

		// Construct the buffers
		_vertex_buffer = BufferData(phdev, device,
			vertex_buffer_size,
			vk::BufferUsageFlagBits::eVertexBuffer,
			vk::MemoryPropertyFlagBits::eHostVisible
				| vk::MemoryPropertyFlagBits::eHostCoherent
		);

		_index_buffer = BufferData(phdev, device,
			index_buffer_size,
			vk::BufferUsageFlagBits::eIndexBuffer,
			vk::MemoryPropertyFlagBits::eHostVisible
				| vk::MemoryPropertyFlagBits::eHostCoherent
		);

		// Update buffers
		_update_buffers();
	}
public:
	// Type name
	static constexpr char object_type[] = "GUI-Rect";

	// Default constructor
	Rect() = default;

	// Constructors
	Rect(const vk::raii::PhysicalDevice &phdev,
			const vk::raii::Device &device,
			const glm::vec4 &bounds,
			const glm::vec3 &c = glm::vec3 {1.0})
			: Object(object_type),
			min(bounds.x, bounds.y),
			max(bounds.z, bounds.w),
			color(c) {
		// Initialize buffers
		_init_buffers(phdev, device);
	}

	Rect(const vk::raii::PhysicalDevice &phdev,
			const vk::raii::Device &device,
			const coordinates::Screen &sc,
			const coordinates::Screen &size,
			const glm::vec3 &c = glm::vec3 {1.0})
			: Object(object_type), color(c) {
		// Calulcate bounds
		glm::vec2 ndc = sc.to_ndc();
		glm::vec2 ndc_size = 2.0f * size.to_unit();

		min = ndc;
		max = ndc + ndc_size;

		// Initialize buffers
		_init_buffers(phdev, device);
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
	void render(RenderPacket &rp) override {
		// Bind buffers
		rp.cmd.bindVertexBuffers(0, {*_vertex_buffer.buffer}, {0});
		rp.cmd.bindIndexBuffer(*_index_buffer.buffer, 0, vk::IndexType::eUint32);

		// Draw
		rp.cmd.drawIndexed(6, 1, 0, 0, 0);
	}
};

}

}

#endif

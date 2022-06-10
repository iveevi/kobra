#ifndef KOBRA_GUI_SPRITE_H_
#define KOBRA_GUI_SPRITE_H_

// Engine headers
#include "../include/coords.hpp"
// #include "../sampler.hpp"
#include "gui.hpp"

namespace kobra {

namespace gui {

// Sprite class
class Sprite : public _element {
public:
	// Vertex data
	struct Vertex {
		glm::vec2 pos;
		glm::vec2 tex;

		// Get Vulkan info for vertex
		static vk::VertexInputBindingDescription
				vertex_binding() {
			return {
				0, sizeof(Vertex),
				vk::VertexInputRate::eVertex
			};
		}

		static std::vector <vk::VertexInputAttributeDescription>
				vertex_attributes() {
			return {
				vk::VertexInputAttributeDescription {
					0, 0, vk::Format::eR32G32Sfloat,
					offsetof(Vertex, pos)
				},

				vk::VertexInputAttributeDescription {
					1, 0, vk::Format::eR32G32Sfloat,
					offsetof(Vertex, tex)
				}
			};
		}
	};
protected:
	// Image bounds (NDC)
	glm::vec4	_bounds = {0.0f, 0.0f, 0.0f, 0.0f};

	// Vulkan device
	const vk::raii::Device &_device = nullptr;

	// Image data and sampler
	ImageData _image_data = nullptr;
	vk::raii::Sampler _sampler = nullptr;

	// Descriptor set
	vk::raii::DescriptorSet _ds = nullptr;

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
			Vertex {{_bounds.x, _bounds.y}, {0, 0}},
			Vertex {{_bounds.z, _bounds.y}, {1, 0}},
			Vertex {{_bounds.z, _bounds.w}, {1, 1}},
			Vertex {{_bounds.x, _bounds.w}, {0, 1}}
		};

		// Upload to buffers
		_vertex_buffer.upload(vertices);
		_index_buffer.upload(indices);
	}

	// Initialize buffers
	void _init_buffers(const vk::raii::PhysicalDevice &phdev,
			const vk::raii::Device &device) {
		// Create buffers
		auto vertex_buffer_size = sizeof(Vertex) * 4;
		auto index_buffer_size = sizeof(uint) * 6;

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
	// Object type
	static constexpr const char object_type[] = "GUI-Sprite";

	// Default constructor
	Sprite() = default;

	// Constructors
	Sprite(const vk::raii::PhysicalDevice &phdev,
			const vk::raii::Device &device,
			const vk::raii::CommandPool &command_pool,
			const glm::vec4 &bounds,
			const std::string &path = "")
			: Object(object_type),
			_bounds(bounds),
			_device(device) {
		// Initialize buffers
		_init_buffers(phdev, device);

		// Create sampler
		if (!path.empty()) {
			_image_data = make_image(phdev, device,
				command_pool, path,
				vk::ImageTiling::eOptimal,
				vk::ImageUsageFlagBits::eSampled,
				vk::MemoryPropertyFlagBits::eDeviceLocal,
				vk::ImageAspectFlagBits::eColor
			);

			_sampler = make_sampler(device, _image_data);
		}
	}

	Sprite(const vk::raii::PhysicalDevice &phdev,
			const vk::raii::Device &device,
			const vk::raii::CommandPool &command_pool,
			const coordinates::Screen &sc,
			const coordinates::Screen &size,
			const std::string &path = "")
			: Object(object_type),
			_device(device) {
		// Calulcate bounds
		glm::vec2 ndc = sc.to_ndc();
		glm::vec2 ndc_size = 2.0f * size.to_unit();

		_bounds = glm::vec4 {ndc, ndc + ndc_size};

		// Initialize buffers
		_init_buffers(phdev, device);

		// Create sampler
		if (!path.empty()) {
			// TODO: load from TextureManager
			_image_data = make_image(phdev, device,
				command_pool, path,
				vk::ImageTiling::eOptimal,
				vk::ImageUsageFlagBits::eSampled,
				vk::MemoryPropertyFlagBits::eDeviceLocal,
				vk::ImageAspectFlagBits::eColor
			);

			_sampler = make_sampler(device, _image_data);
		}
	}

	// Change image
	void set_image(const vk::raii::PhysicalDevice &phdev,
			const vk::raii::Device &device,
			const vk::raii::CommandPool &command_pool,
			const std::string &path) {
		// Create sampler
		if (!path.empty()) {
			_image_data = make_image(phdev, device,
				command_pool, path,
				vk::ImageTiling::eOptimal,
				vk::ImageUsageFlagBits::eSampled,
				vk::MemoryPropertyFlagBits::eDeviceLocal,
				vk::ImageAspectFlagBits::eColor
			);

			_sampler = make_sampler(device, _image_data);
		}
	}

	// Virtual methods
	glm::vec2 position() const override {
		return glm::vec2(_bounds.x, _bounds.y);
	}

	glm::vec4 bounding_box() const override {
		return _bounds;
	}

	// Latch onto a layer
	void latch(LatchingPacket &) override;

	// Render
	void render(RenderPacket &packet) override {
		// Bind descriptor set
		packet.cmd.bindDescriptorSets(
			vk::PipelineBindPoint::eGraphics,
			*packet.sprite_layout,
			0, {*_ds}, {}
		);

		// Bind vertex and index buffers
		packet.cmd.bindVertexBuffers(0, {*_vertex_buffer.buffer}, {0});
		packet.cmd.bindIndexBuffer(*_index_buffer.buffer, 0, vk::IndexType::eUint32);

		// Draw
		packet.cmd.drawIndexed(6, 1, 0, 0, 0);
	}
};

}

}

#endif

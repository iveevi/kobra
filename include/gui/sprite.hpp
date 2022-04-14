#ifndef KOBRA_GUI_SPRITE_H_
#define KOBRA_GUI_SPRITE_H_

// Engine headers
#include "../sampler.hpp"
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
		static Vulkan::VB vertex_binding() {
			return {
				.binding = 0,
				.stride = sizeof(Vertex),
				.inputRate = VK_VERTEX_INPUT_RATE_VERTEX
			};
		}

		static std::vector <Vulkan::VA> vertex_attributes() {
			return {
				{
					.binding = 0,
					.location = 0,
					.format = VK_FORMAT_R32G32_SFLOAT,
					.offset = offsetof(Vertex, pos)
				},
				{
					.binding = 0,
					.location = 1,
					.format = VK_FORMAT_R32G32_SFLOAT,
					.offset = offsetof(Vertex, tex)
				}
			};
		}
	};

	// Aliases
	using VertexBuffer = BufferManager <Vertex>;
protected:
	glm::vec4	_bounds = {0.0f, 0.0f, 0.0f, 0.0f};

	// Sampler
	Sampler		*_sampler = nullptr;

	// Vulkan context
	Vulkan::Context	_context;

	// Command pool
	VkCommandPool	_command_pool;

	// Descriptor set
	Vulkan::DS	_ds = VK_NULL_HANDLE;

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
			Vertex {{_bounds.x, _bounds.y}, {0, 0}},
			Vertex {{_bounds.z, _bounds.y}, {1, 0}},
			Vertex {{_bounds.z, _bounds.w}, {1, 1}},
			Vertex {{_bounds.x, _bounds.w}, {0, 1}}
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
	void _init_buffers() {
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
		_vb = VertexBuffer(_context, vb_settings);
		_ib = IndexBuffer(_context, ib_settings);

		// Update buffers
		_update_buffers();
	}
public:
	// Object type
	static constexpr const char object_type[] = "GUI-Sprite";

	// Default constructor
	Sprite() = default;

	// Constructors
	Sprite(const Vulkan::Context &context,
			const VkCommandPool &command_pool,
			const glm::vec4 &bounds,
			const std::string &path = "")
			: Object(object_type),
			_bounds(bounds),
			_context(context),
			_command_pool(command_pool) {
		// Initialize buffers
		_init_buffers();

		// Create sampler
		if (!path.empty())
			_sampler = new Sampler(_context, _command_pool, path);
	}

	Sprite(const Vulkan::Context &context,
			const VkCommandPool &command_pool,
			const coordinates::Screen &sc,
			const coordinates::Screen &size,
			const std::string &path = "")
			: Object(object_type),
			_context(context),
			_command_pool(command_pool) {
		// Calulcate bounds
		glm::vec2 ndc = sc.to_ndc();
		glm::vec2 ndc_size = 2.0f * size.to_unit();

		_bounds = glm::vec4 {ndc, ndc + ndc_size};

		// Initialize buffers
		_init_buffers();

		// Create sampler
		if (!path.empty())
			_sampler = new Sampler(_context, _command_pool, path);
	}

	// Change image
	void set_image(const std::string &path) {
		// Create sampler
		if (!path.empty())
			_sampler = new Sampler(_context, _command_pool, path);
	}

	// Virtual methods
	glm::vec2 position() const override {
		return glm::vec2(_bounds.x, _bounds.y);
	}

	glm::vec4 bounding_box() const override {
		return _bounds;
	}

	// Latch onto a layer
	void latch(LatchingPacket &lp) override {
		_ds = lp.layer->serve_sprite_ds();
		if (_sampler)
			_sampler->bind(_ds, 0);
	}

	// Render
	void render(RenderPacket &packet) override {
		// Draw
		VkBuffer	vbuffers[] = {_vb.vk_buffer()};
		VkDeviceSize	offsets[] = {0};

		// Bind descriptor set
		vkCmdBindDescriptorSets(packet.cmd,
			VK_PIPELINE_BIND_POINT_GRAPHICS,
			packet.sprite_layout,
			0, 1, &_ds,
			0, nullptr
		);

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

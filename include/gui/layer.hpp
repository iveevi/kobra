#ifndef KOBRA_GUI_LAYER_H_
#define KOBRA_GUI_LAYER_H_

// Standard headers
#include <map>
#include <vector>

// Engine headers
#include "../app.hpp"
#include "../layer.hpp"
#include "gui.hpp"
#include "text.hpp"

namespace kobra {

namespace gui {

// Contains a set of
//	GUI elements to render
// TODO: derive from element?
class Layer : public kobra::Layer <_element> {
	// Private aliases
	template <class T>
	using str_map = std::map <std::string, T>;

	// Element arranged by pipeline
	str_map <std::vector <ptr>>	_pipeline_map;

	// Set of Text Render objects for each font
	std::vector <TextRender>	_text_renders;

	// Map of font names/aliases to
	//	their respective TextRender indices
	str_map <int>			_font_map;

	// Vulkan structures
	vk::raii::PhysicalDevice	*_physical_device = nullptr;
	vk::raii::Device		*_device = nullptr;
	vk::raii::CommandPool		*_command_pool = nullptr;
	vk::raii::DescriptorPool	*_descriptor_pool = nullptr;

	vk::raii::RenderPass		_render_pass = nullptr;

	// Other layer properties
	vk::Extent2D			_extent;

	// Pipelines
	struct {
		// Vulkan::Pipeline	shapes;
		// Vulkan::Pipeline	sprites;

		// Pipelines
		vk::raii::Pipeline shapes = nullptr;
		vk::raii::Pipeline sprites = nullptr;

		// Pipeline layouts
		vk::raii::PipelineLayout shapes_layout = nullptr;
		vk::raii::PipelineLayout sprites_layout = nullptr;
	} _pipelines;

	// Descriptor set layouts
	vk::raii::DescriptorSetLayout	_dsl_sprites = nullptr;

	// Descriptor set layout bindings
	static const std::vector <DSLB>	_sprites_bindings;

	// Allocation methods
	void _init_vulkan_structures(const vk::AttachmentLoadOp &load,
			const vk::Format &swapchain_format,
			const vk::Format &depth_format) {
		// Create render pass
		_render_pass = make_render_pass(
			*_device,
			swapchain_format,
			depth_format,
			load
		);
	}

	// Hardware resources
	struct {
		BufferData vertex = nullptr;
		BufferData index = nullptr;
	} rects;

	// Allocation methods
	void _alloc_rects(const vk::raii::PhysicalDevice &phdev,
			const vk::raii::Device &device) {
		// Base sizes for both buffers
		vk::DeviceSize vertex_size = 1024 * sizeof(Vertex);
		vk::DeviceSize index_size = 1024 * sizeof(uint32_t);

		// Allocate vertex buffer
		rects.vertex = BufferData(phdev, device,
			vertex_size,
			vk::BufferUsageFlagBits::eVertexBuffer,
			vk::MemoryPropertyFlagBits::eHostVisible
				| vk::MemoryPropertyFlagBits::eHostCoherent
		);

		// Allocate index buffer
		rects.index = BufferData(phdev, device,
			index_size,
			vk::BufferUsageFlagBits::eIndexBuffer,
			vk::MemoryPropertyFlagBits::eHostVisible
				| vk::MemoryPropertyFlagBits::eHostCoherent
		);
	}
public:
	// Default
	Layer() = default;

	// Constructor
	Layer(vk::raii::PhysicalDevice &,
			vk::raii::Device &,
			vk::raii::CommandPool &,
			vk::raii::DescriptorPool &,
			const vk::Extent2D &,
			const vk::Format &,
			const vk::Format &,
			const vk::AttachmentLoadOp & = vk::AttachmentLoadOp::eLoad);

	// Delete copy and assignment
	Layer(const Layer &) = delete;
	Layer &operator=(const Layer &) = delete;

	// Add move and assignment
	Layer(Layer &&) = default;
	Layer &operator=(Layer &&) = default;

	// Add action
	void add_do(const ptr &) override;

	// Serve descriptor sets
	vk::raii::DescriptorSet serve_sprite_ds() {
		auto dsets = vk::raii::DescriptorSets {
			*_device,
			{**_descriptor_pool, *_dsl_sprites}
		};

		return std::move(dsets.front());
	}

	// Adding a scene
	void add_scene(const Scene &scene) override {
		KOBRA_LOG_FUNC(warn) << "Not implemented\n";
	}

	// Load fonts
	void load_font(const std::string &alias, const std::string &path) {
		size_t index = _text_renders.size();
		_text_renders.push_back(
			TextRender(*_physical_device, *_device,
				*_command_pool,
				*_descriptor_pool,
				_render_pass, path,
				_extent.width,
				_extent.height
			)
		);

		_font_map[alias] = index;
	}

	// Get TextRender
	TextRender *text_render(int index) {
		return &_text_renders[index];
	}

	TextRender *text_render(const std::string &alias) {
		return text_render(_font_map.at(alias));
	}

	// Render using command buffer and framebuffer
	void render(const vk::raii::CommandBuffer &cmd,
			const vk::raii::Framebuffer &framebuffer) {
		// Set viewport
		auto viewport = vk::Viewport {
			0.0f, 0.0f,
			static_cast<float>(_extent.width),
			static_cast<float>(_extent.height),
			0.0f, 1.0f
		};

		cmd.setViewport(0, viewport);

		// Set scissor
		auto scissor = vk::Rect2D {
			vk::Offset2D {0, 0},
			_extent
		};

		cmd.setScissor(0, scissor);

		// Start render pass
		std::array <vk::ClearValue, 2> clear_values = {
			vk::ClearValue {
				vk::ClearColorValue {
					std::array <float, 4> {0.0f, 0.0f, 0.0f, 1.0f}
				}
			},
			vk::ClearValue {
				vk::ClearDepthStencilValue {
					1.0f, 0
				}
			}
		};

		cmd.beginRenderPass(
			vk::RenderPassBeginInfo {
				*_render_pass,
				*framebuffer,
				vk::Rect2D {
					vk::Offset2D {0, 0},
					_extent,
				},
				static_cast <uint32_t> (clear_values.size()),
				clear_values.data()
			},
			vk::SubpassContents::eInline
		);

		// Initialize render packet
		RenderPacket rp {
			.cmd = cmd,
			.sprite_layout = _pipelines.sprites_layout,
		};

		// Render all plain shapes
		cmd.bindPipeline(vk::PipelineBindPoint::eGraphics, *_pipelines.shapes);
		for (auto &e : _pipeline_map["shapes"])
			e->render_element(rp);

		// Render all sprites
		cmd.bindPipeline(vk::PipelineBindPoint::eGraphics, *_pipelines.sprites);
		for (auto &e : _pipeline_map["sprites"])
			e->render_element(rp);

		// Render all the text renders
		for (auto &tr : _text_renders)
			tr.render(cmd);

		// End render pass
		cmd.endRenderPass();
	}
};

}

}

#endif

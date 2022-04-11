#ifndef KOBRA_ENGINE_GIZMO_H_
#define KOBRA_ENGINE_GIZMO_H_

// Standard headers
#include <vector>

// Engine headers
#include "../app.hpp"
#include "../backend.hpp"
#include "../camera.hpp"
#include "../raster/mesh.hpp"

namespace kobra {

namespace engine {

// Gizmo class handles displaying
// 	gizmos in the scene (could be multiple)
class Gizmo {
	// Gizmo substructure
	struct _gizmo {
		bool visible = true;

		// Virtual methods
		virtual void set_position(const glm::vec3 &) = 0;
		virtual void render(raster::RenderPacket &) = 0;
	};

	// Transform gizmo
	class TransformGizmo : public _gizmo {
		raster::Mesh	*x_box;
		raster::Mesh	*y_box;
		raster::Mesh	*z_box;
	public:
		// Constructor
		TransformGizmo(const Vulkan::Context &context) {
			glm::vec3 pos {0.0f};
			x_box = new raster::Mesh(context, Mesh::make_box(pos, {1, 0.01, 0.01}));
			y_box = new raster::Mesh(context, Mesh::make_box(pos, {0.01, 1, 0.01}));
			z_box = new raster::Mesh(context, Mesh::make_box(pos, {0.01, 0.01, 1}));

			x_box->material().albedo = {1, 0, 0};
			y_box->material().albedo = {0, 1, 0};
			z_box->material().albedo = {0, 0, 1};
		}

		// Destructor
		~TransformGizmo() {
			delete x_box;
			delete y_box;
			delete z_box;
		}

		// Set position of gizmo
		void set_position(const glm::vec3 &pos) override {
			x_box->transform().position = pos;
			y_box->transform().position = pos;
			z_box->transform().position = pos;
		}

		// Render gizmo
		void render(raster::RenderPacket &packet) override {
			if (visible) {
				x_box->draw(packet);
				y_box->draw(packet);
				z_box->draw(packet);
			}
		}
	};
public:
	// Public aliases
	using Handle = std::shared_ptr <_gizmo>;
private:
	// List of gizmos
	std::vector <Handle>	_gizmos;

	// Vulkan structures
	Vulkan::Context		_context;
	VkRenderPass		_render_pass;
	Vulkan::Pipeline	_pipeline;
	VkExtent2D		_extent;

	// Active camera
	Camera			_camera;

	// Initialization
	void _init_vulkan_structures(const Vulkan::Swapchain &swapchain) {
		// Create render pass
		// TODO: context method for this
		_render_pass = _context.vk->make_render_pass(
			_context.phdev,
			_context.device,
			swapchain,
			VK_ATTACHMENT_LOAD_OP_LOAD,
			VK_ATTACHMENT_STORE_OP_STORE
		);

		// Load shaders
		auto shaders = _context.make_shaders({
			"shaders/bin/raster/vertex.spv",
			"shaders/bin/raster/plain_color_frag.spv"
		});

		// Push constants
		VkPushConstantRange pcr {
			.stageFlags = VK_SHADER_STAGE_VERTEX_BIT,
			.offset = 0,
			.size = sizeof(typename raster::Mesh::MVP)
		};

		// Creation info
		Vulkan::PipelineInfo info {
			.swapchain = swapchain,
			.render_pass = _render_pass,

			.vert = shaders[0],
			.frag = shaders[1],

			.dsls = {},

			.vertex_binding = Vertex::vertex_binding(),
			.vertex_attributes = Vertex::vertex_attributes(),

			.push_consts = 1,
			.push_consts_range = &pcr,

			.depth_test = false,

			.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,

			// TODO: make a swapchain method to get viewport
			.viewport {
				.width = (int) swapchain.extent.width,
				.height = (int) swapchain.extent.height,
				.x = 0,
				.y = 0
			}
		};

		_pipeline = _context.make_pipeline(info);
	}
public:
	// Default constructor
	Gizmo() = default;

	// Constructor
	Gizmo(const App::Window &wctx, const VkAttachmentLoadOp &load)
			: _context(wctx.context),
			_extent(wctx.swapchain.extent) {
		// Initialize vulkan structures
		_init_vulkan_structures(wctx.swapchain);
	}

	// Serve gizmos
	Handle transform_gizmo() {
		// Create gizmo
		auto gizmo = std::make_shared <TransformGizmo> (_context);

		// Add to list
		_gizmos.push_back(gizmo);

		// Return handle
		return gizmo;
	}

	// Set camera
	void set_camera(const Camera &camera) {
		_camera = camera;
	}

	// Render all gizmos
	void render(const VkCommandBuffer &cmd, const VkFramebuffer &framebuffer) {
		// Start gizmo render pass
		VkClearValue clear_colors[] = {
			{.color = {0.0f, 0.0f, 0.0f, 1.0f}},
			{.depthStencil = {1.0f, 0}}
		};

		VkRenderPassBeginInfo render_pass_info = {
			.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
			.renderPass = _render_pass,
			// TODO: should each Vulkan::Pipeline have a refernce to its render pass?
			.framebuffer = framebuffer,
			.renderArea = {
				.offset = { 0, 0 },
				.extent = _extent
			},
			.clearValueCount = 2,
			.pClearValues = clear_colors
		};

		vkCmdBeginRenderPass(cmd,
			&render_pass_info,
			VK_SUBPASS_CONTENTS_INLINE
		);

		// Bind pipeline
		vkCmdBindPipeline(cmd,
			VK_PIPELINE_BIND_POINT_GRAPHICS,
			_pipeline.pipeline
		);

		// Initialize render packet
		raster::RenderPacket packet {
			.cmd = cmd,

			.pipeline_layout = _pipeline.layout,

			// TODO: warn on null camera
			.view = _camera.view(),
			.proj = _camera.perspective()
		};

		// Render gizmos
		for (auto &gizmo : _gizmos)
			gizmo->render(packet);

		// End render pass
		vkCmdEndRenderPass(cmd);
	}
};

}

}

#endif

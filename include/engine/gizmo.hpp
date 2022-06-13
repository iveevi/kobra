#ifndef KOBRA_ENGINE_GIZMO_H_
#define KOBRA_ENGINE_GIZMO_H_

// Standard headers
#include <vector>

// Engine headers
#include "../app.hpp"
#include "../backend.hpp"
#include "../camera.hpp"
#include "../object.hpp"
#include "../raster/mesh.hpp"

namespace kobra {

namespace engine {

// Gizmo class handles displaying
// 	gizmos in the scene (could be multiple)
class Gizmo {
	// Gizmo substructure
	class _gizmo {
	protected:
		ObjectPtr	object;
	public:
		bool		visible = true;

		// Bind the gizmo to an object
		void bind(ObjectPtr object) {
			this->object = object;
		}

		// Get the object the gizmo is bound to
		ObjectPtr get_object() {
			return object;
		}

		// Deselect
		void deselect() {
			object = nullptr;
			this->deactivate();
		}

		// Virtual methods
		virtual const glm::vec3 &get_position() const = 0;
		virtual bool handle_select(const glm::mat4 &, float, float, float, float) = 0;
		virtual void handle_drag(const glm::mat4 &, float, float, float, float) = 0;
		virtual void deactivate() = 0;
		virtual void set_position(const glm::vec3 &) = 0;
		virtual void render(raster::RenderPacket &) = 0;
	};

	// Transform gizmo
	class TransformGizmo : public _gizmo {
		raster::Mesh	*x_box;
		raster::Mesh	*y_box;
		raster::Mesh	*z_box;

		// TODO: make protected
		glm::vec3	pos {0.0f};

		// Currently active axis
		int		active_axis = -1;
	public:
		// Constructor
		TransformGizmo(const vk::raii::PhysicalDevice &phdev_,
				const vk::raii::Device &device_) {
			x_box = new raster::Mesh(phdev_, device_, Mesh::make_box(pos, {0.5, 0.02, 0.02}));
			y_box = new raster::Mesh(phdev_, device_, Mesh::make_box(pos, {0.02, 0.5, 0.02}));
			z_box = new raster::Mesh(phdev_, device_, Mesh::make_box(pos, {0.02, 0.02, 0.5}));

			x_box->material().Kd = {1, 0, 0};
			y_box->material().Kd = {0, 1, 0};
			z_box->material().Kd = {0, 0, 1};

			set_position({0, 0, 0});
		}

		// Destructor
		~TransformGizmo() {
			delete x_box;
			delete y_box;
			delete z_box;
		}

		// Get position of gizmo
		const glm::vec3 &get_position() const override {
			return pos;
		}

		// Handle initial selection
		bool handle_select(const glm::mat4 &proj, float x, float y, float w, float h) override {
			glm::vec4 a = {pos, 1};
			glm::vec4 b = {pos + glm::vec3(1, 0, 0), 1};
			glm::vec4 c = {pos + glm::vec3(0, 1, 0), 1};
			glm::vec4 d = {pos + glm::vec3(0, 0, 1), 1};

			auto project = [&](const glm::vec4 &v) -> glm::vec2 {
				glm::vec4 v_ = proj * v;
				v_ = v_ / v_.w;

				glm::vec2 v_2 = {v_.x, v_.y};
				v_2 = v_2 * 0.5f + 0.5f;

				v_2 *= glm::vec2 {w, h};

				return v_2;
			};

			glm::vec2 a_ = project(a);
			glm::vec2 b_ = project(b);
			glm::vec2 c_ = project(c);
			glm::vec2 d_ = project(d);

			float d_x = distance(a_, b_, {x, y});
			float d_y = distance(a_, c_, {x, y});
			float d_z = distance(a_, d_, {x, y});

			// Activate axis based on the smallest distance
			bool activate_x = (d_x < 20) && (d_x < d_y) && (d_x < d_z);
			bool activate_y = (d_y < 20) && (d_y < d_x) && (d_y < d_z);
			bool activate_z = (d_z < 20) && (d_z < d_x) && (d_z < d_y);

			if (activate_x)
				active_axis = 0;
			else if (activate_y)
				active_axis = 1;
			else if (activate_z)
				active_axis = 2;
			else
				active_axis = -1;

			return (active_axis != -1);
		}

		// Handle dragging
		void handle_drag(const glm::mat4 &proj, float dx, float dy, float w, float h) override {
			if (active_axis == -1)
				return;

			glm::vec4 a = {pos, 1};
			glm::vec4 b = {pos + glm::vec3(1, 0, 0), 1};
			glm::vec4 c = {pos + glm::vec3(0, 1, 0), 1};
			glm::vec4 d = {pos + glm::vec3(0, 0, 1), 1};

			// TODO: static helper method
			auto project = [&](const glm::vec4 &v) -> glm::vec2 {
				glm::vec4 v_ = proj * v;
				v_ = v_ / v_.w;

				glm::vec2 v_2 = {v_.x, v_.y};
				v_2 = v_2 * 0.5f + 0.5f;

				v_2 *= glm::vec2 {w, h};

				return v_2;
			};

			// TODO: only do these calculation for the relevant axis
			glm::vec2 a_ = project(a);
			glm::vec2 b_ = project(b);
			glm::vec2 c_ = project(c);
			glm::vec2 d_ = project(d);

			float scale_x = glm::length(b_ - a_);
			float scale_y = glm::length(c_ - a_);
			float scale_z = glm::length(d_ - a_);

			float dd = 0;
			if (active_axis == 0 && scale_x > 0) {
				dd = glm::dot(glm::normalize(b_ - a_), glm::vec2 {dx, dy});
				dd /= scale_x;
			} else if (active_axis == 1 && scale_y > 0) {
				dd = glm::dot(glm::normalize(c_ - a_), glm::vec2 {dx, dy});
				dd /= scale_y;
			} else if (active_axis == 2 && scale_z > 0) {
				dd = glm::dot(glm::normalize(d_ - a_), glm::vec2 {dx, dy});
				dd /= scale_z;
			}

			// Move the bound object
			object->transform().position[active_axis] += dd;

			// Update position and boxes
			pos[active_axis] += dd;
			set_position(pos);
		}

		// Deactivate gizmo
		void deactivate() override {
			active_axis = -1;
		}

		// Set position of gizmo
		void set_position(const glm::vec3 &x) override {
			pos = x;
			x_box->transform().position = {pos.x + 0.5, pos.y, pos.z};
			y_box->transform().position = {pos.x, pos.y + 0.5, pos.z};
			z_box->transform().position = {pos.x, pos.y, pos.z + 0.5};
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
	std::vector <Handle>		_gizmos;

	// Vulkan structures
	const vk::raii::PhysicalDevice 	&_physical_device = nullptr;
	const vk::raii::Device		&_device = nullptr;
	vk::raii::RenderPass		_render_pass = nullptr;
	vk::raii::Pipeline		_pipeline = nullptr;
	vk::raii::PipelineLayout 	_pipeline_layout = nullptr;
	vk::Extent2D			_extent;

	// Active camera
	Camera			_camera;

	// Initialization
	void _init_vulkan_structures(const vk::Format &swapchain_format,
			const vk::Format &depth_format,
			const vk::AttachmentLoadOp &load) {
		// Create render pass
		_render_pass = make_render_pass(_device, swapchain_format, depth_format, load);

		// Load shaders
		auto shaders = make_shader_modules(_device, {
			"shaders/bin/raster/vertex.spv",
			"shaders/bin/raster/plain_color_frag.spv"
		});

		// Push constants
		vk::PushConstantRange pcr {
			vk::ShaderStageFlagBits::eVertex,
			0, sizeof(raster::Mesh::MVP)
		};

		// Pipeline layout
		_pipeline_layout = vk::raii::PipelineLayout {
			_device,
			{{}, {}, pcr}
		};

		// Pipeline cache
		vk::raii::PipelineCache pc {_device, nullptr};

		// Create the graphics pipeline
		GraphicsPipelineInfo grp_info {
			.device = _device,
			.render_pass = _render_pass,

			.vertex_shader = std::move(shaders[0]),
			.fragment_shader = std::move(shaders[1]),

			.vertex_binding = Vertex::vertex_binding(),
			.vertex_attributes = Vertex::vertex_attributes(),

			.pipeline_layout = _pipeline_layout,
			.pipeline_cache = pc,

			.depth_test = true,
			.depth_write = true
		};

		_pipeline = make_graphics_pipeline(grp_info);
	}
public:
	// Default constructor
	Gizmo() = default;

	// Constructor
	Gizmo(const vk::raii::PhysicalDevice &phdev_,
			const vk::raii::Device &device_,
			const vk::Extent2D &extent_,
			const vk::Format &swapchain_format,
			const vk::Format &depth_format,
			const vk::AttachmentLoadOp &load)
			: _physical_device(phdev_),
			_device(device_),
			_extent(extent_) {
		// Initialize vulkan structures
		_init_vulkan_structures(swapchain_format, depth_format, load);
	}

	// Serve gizmos
	Handle transform_gizmo() {
		// Create gizmo
		auto gizmo = std::make_shared <TransformGizmo> (_physical_device, _device);

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
	void render(const vk::raii::CommandBuffer &cmd, const vk::raii::Framebuffer &framebuffer) {
		// Start gizmo render pass
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

		// Bind pipeline
		cmd.bindPipeline(vk::PipelineBindPoint::eGraphics, *_pipeline);

		// Initialize render packet
		raster::RenderPacket packet {
			.cmd = cmd,

			.pipeline_layout = _pipeline_layout,

			// TODO: warn on null camera
			.view = _camera.view(),
			.proj = _camera.perspective()
		};

		// Render gizmos
		for (auto &gizmo : _gizmos)
			gizmo->render(packet);

		// End render pass
		cmd.endRenderPass();
	}
};

}

}

#endif

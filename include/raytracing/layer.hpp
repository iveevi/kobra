#ifndef KOBRA_RT_LAYER_H_
#define KOBRA_RT_LAYER_H_

// Standard headers
#include <vector>
#include <vulkan/vulkan_core.h>

// Engine headers
#include "../../shaders/rt/bindings.h"
#include "../app.hpp"
#include "../backend.hpp"
#include "../bbox.hpp"
#include "../bvh.hpp"
#include "../camera.hpp"
#include "../layer.hpp"
#include "../logger.hpp"
#include "../mesh.hpp"
#include "../sphere.hpp"
#include "batch.hpp"
#include "rt.hpp"

namespace kobra {

namespace rt {

// Layer class
class Layer : public kobra::Layer <rt::_element> {
	// Private aliases
	using DSLBinding = VkDescriptorSetLayoutBinding;
	using DSLBindings = std::vector <DSLBinding>;

	// All of the layer's cameras
	std::vector <Camera>	_cameras;

	// Active camera
	Camera			*_active_camera = nullptr;

	// Vulkan context
	Vulkan::Context		_context;

	// Descriptor pool
	VkDescriptorPool	_descriptor_pool;

	// Command pool
	VkCommandPool		_command_pool;

	// Swapchain extent
	VkExtent2D		_extent;

	// Render pass
	VkRenderPass		_render_pass;

	// Pipelines
	struct {
		Vulkan::Pipeline mesh;
		Vulkan::Pipeline postproc;
	} _pipelines;

	// Descriptor set bindings
	static const DSLBindings _mesh_compute_bindings;
	static const DSLBindings _postproc_bindings;

	VkDescriptorSetLayout	_mesh_dsl;
	VkDescriptorSetLayout	_postproc_dsl;

	// Descriptor sets
	VkDescriptorSet		_mesh_ds;
	VkDescriptorSet		_postproc_ds;

	// Barrier
	// VkImageMemoryBarrier	_barrier;

	// Initialize mesh compute pipeline
	void _init_mesh_compute_pipeline();
	void _init_postproc_pipeline(const Vulkan::Swapchain &);

	// Initialize all pipelines
	void _init_pipelines(const Vulkan::Swapchain &swapchain) {
		// First, create the DSLs
		_mesh_dsl = _context.vk->make_descriptor_set_layout(
			_context.device,
			_mesh_compute_bindings
		);

		_postproc_dsl = _context.vk->make_descriptor_set_layout(
			_context.device,
			_postproc_bindings
		);

		// Then, create the descriptor sets
		_mesh_ds = _context.vk->make_descriptor_set(
			_context.device,
			_descriptor_pool,
			_mesh_dsl
		);

		_postproc_ds = _context.vk->make_descriptor_set(
			_context.device,
			_descriptor_pool,
			_postproc_dsl
		);

		// All pipelines
		_init_mesh_compute_pipeline();
		_init_postproc_pipeline(swapchain);
	}

	// Data
	BufferManager <uint>	_pixels;
	Buffer4f		_vertices;
	Buffer4f		_triangles;
	Buffer4f		_materials;
	Buffer4f		_lights;		// TODO: make as a 4u buffer, less casting overall in the shader
	BufferManager <uint>	_light_indices;

	Buffer4m		_transforms;


	//////////////
	// Samplers //
	//////////////

	// Empty sampler
	Sampler			_empty_sampler;

	// Environment sampler
	Sampler			_env_sampler;

	// Vector of image descriptors
	ImageDescriptors	_albedo_image_descriptors;
	ImageDescriptors	_normal_image_descriptors;

	// Update descriptor set for albedo
	void _update_samplers(const ImageDescriptors &ids, uint binding) {
		// Update descriptor set
		VkWriteDescriptorSet descriptor_write {
			.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
			.dstSet = _mesh_ds,
			.dstBinding = binding,
			.dstArrayElement = 0,
			.descriptorCount = (uint) ids.size(),
			.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
			.pImageInfo = ids.data()
		};

		vkUpdateDescriptorSets(_context.vk_device(),
			1, &descriptor_write,
			0, nullptr
		);
	}

	// BVH
	BVH			_bvh;

	// Batching variables
	int			_xbatch = 0;
	int			_ybatch = 0;

	int			_xbatch_size = 50;
	int			_ybatch_size = 50;

	// Get list of bboxes for each triangle
	std::vector <BoundingBox> _get_bboxes() const {
		std::vector <BoundingBox> bboxes;
		bboxes.reserve(_triangles.size());

		const auto &vertices = _vertices.vector();
		const auto &triangles = _triangles.vector();

		for (size_t i = 0; i < _triangles.push_size(); i++) {
			const auto &triangle = triangles[i];

			float ia = triangle.data.x;
			float ib = triangle.data.y;
			float ic = triangle.data.z;
			float id = triangle.data.w;

			uint a = *(reinterpret_cast <uint *> (&ia));
			uint b = *(reinterpret_cast <uint *> (&ib));
			uint c = *(reinterpret_cast <uint *> (&ic));
			uint d = *(reinterpret_cast <uint *> (&id));

			// If a == b == c, its a sphere
			if (a == b && b == c) {
				glm::vec4 center = vertices[2 * a].data;
				float radius = center.w;

				glm::vec4 min = center - glm::vec4(radius);
				glm::vec4 max = center + glm::vec4(radius);

				bboxes.push_back(BoundingBox {min, max});
			} else {
				glm::vec4 va = vertices[2 * a].data;
				glm::vec4 vb = vertices[2 * b].data;
				glm::vec4 vc = vertices[2 * c].data;

				glm::vec4 min = glm::min(va, glm::min(vb, vc));
				glm::vec4 max = glm::max(va, glm::max(vb, vc));

				bboxes.push_back(BoundingBox {min, max});
			}
		}

		return bboxes;
	}
public:
	// Default constructor
	Layer() = default;

	// Constructor
	Layer(const App::Window &wctx)
			: _context(wctx.context),
			_extent(wctx.swapchain.extent),
			_descriptor_pool(wctx.descriptor_pool),
			_command_pool(wctx.command_pool) {
		// Create the render pass
		// TODO: context method
		_render_pass = _context.vk->make_render_pass(
			_context.phdev,
			_context.device,
			wctx.swapchain,
			VK_ATTACHMENT_LOAD_OP_CLEAR,
			VK_ATTACHMENT_STORE_OP_STORE
		);

		// Initialize pipelines
		_init_pipelines(wctx.swapchain);

		// Allocate buffers
		size_t pixels = wctx.swapchain.extent.width
			* wctx.swapchain.extent.height;

		BFM_Settings pixel_settings {
			.size = pixels,
			.usage_type = BFM_READ_ONLY,
			.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
		};

		BFM_Settings viewport_settings {
			.size = 2,
			.usage_type = BFM_WRITE_ONLY,
			.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
		};

		BFM_Settings write_only_settings {
			.size = 1024,
			.usage_type = BFM_WRITE_ONLY,
			.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
		};

		_pixels = BufferManager <uint> (_context, pixel_settings);

		_vertices = Buffer4f(_context, write_only_settings);
		_triangles = Buffer4f(_context, write_only_settings);
		_materials = Buffer4f(_context, write_only_settings);
		_lights = Buffer4f(_context, write_only_settings);
		_light_indices = BufferManager <uint> (_context, write_only_settings);
		_transforms = Buffer4m(_context, write_only_settings);

		// Initial (blank) binding
		_vertices.bind(_mesh_ds, MESH_BINDING_VERTICES);
		_triangles.bind(_mesh_ds, MESH_BINDING_TRIANGLES);
		_materials.bind(_mesh_ds, MESH_BINDING_MATERIALS);
		_transforms.bind(_mesh_ds, MESH_BINDING_TRANSFORMS);
		_lights.bind(_mesh_ds, MESH_BINDING_LIGHTS);
		_light_indices.bind(_mesh_ds, MESH_BINDING_LIGHT_INDICES);

		// Rebind to descriptor sets
		_bvh = BVH(_context, _get_bboxes());
		_bvh.bind(_mesh_ds, MESH_BINDING_BVH);

		// Bind to descriptor sets
		_pixels.bind(_mesh_ds, MESH_BINDING_PIXELS);
		_pixels.bind(_postproc_ds, MESH_BINDING_PIXELS);

		/////////////////////////////////////////////
		// Fill sampler arrays with blank samplers //
		/////////////////////////////////////////////

		// Initialize samplers
		_empty_sampler = Sampler::blank_sampler(_context, _command_pool);
		_env_sampler = Sampler::blank_sampler(_context, _command_pool);

		// Binding environment sampler
		_env_sampler.bind(_mesh_ds, MESH_BINDING_ENVIRONMENT);

		// Albedos
		while (_albedo_image_descriptors.size() < MAX_TEXTURES)
			_albedo_image_descriptors.push_back(_empty_sampler.get_image_info());

		_update_samplers(_albedo_image_descriptors, MESH_BINDING_ALBEDOS);

		// Normals
		while (_normal_image_descriptors.size() < MAX_TEXTURES)
			_normal_image_descriptors.push_back(_empty_sampler.get_image_info());

		_update_samplers(_normal_image_descriptors, MESH_BINDING_NORMAL_MAPS);
	}

	// Adding elements
	void add_do(const ptr &e) override {
		if (_pipelines.mesh.pipeline == VK_NULL_HANDLE) {
			KOBRA_LOG_FUNC(warn) << "rt::Layer is not yet initialized\n";
			return;
		}

		LatchingPacket lp {
			.vertices = &_vertices,
			.triangles = &_triangles,
			.materials = &_materials,
			.transforms = &_transforms,
			.lights = &_lights,
			.light_indices = &_light_indices,

			.albedo_samplers = _albedo_image_descriptors,
			.normal_samplers = _normal_image_descriptors,
		};

		e->latch(lp, _elements.size());

		// Flush the vertices and triangles
		_vertices.sync_upload();
		_triangles.sync_upload();
		_materials.sync_upload();
		_transforms.sync_upload();
		_lights.sync_upload();
		_light_indices.sync_upload();

		// Rebind to descriptor sets
		// TODO: method
		_vertices.bind(_mesh_ds, MESH_BINDING_VERTICES);
		_triangles.bind(_mesh_ds, MESH_BINDING_TRIANGLES);
		_materials.bind(_mesh_ds, MESH_BINDING_MATERIALS);
		_transforms.bind(_mesh_ds, MESH_BINDING_TRANSFORMS);
		_lights.bind(_mesh_ds, MESH_BINDING_LIGHTS);
		_light_indices.bind(_mesh_ds, MESH_BINDING_LIGHT_INDICES);

		// Update sampler descriptors
		_update_samplers(_albedo_image_descriptors, MESH_BINDING_ALBEDOS);
		_update_samplers(_normal_image_descriptors, MESH_BINDING_NORMAL_MAPS);

		// Update the BVH
		_bvh = BVH(_context, _get_bboxes());

		// Rebind to descriptor sets
		_bvh.bind(_mesh_ds, MESH_BINDING_BVH);
	}

	// Adding scenes
	void add_scene(const Scene &scene) override;

	// Set environment map
	void set_environment_map(const Texture &env) {
		// Create texture packet
		TexturePacket tp = make_texture(_context,
			_command_pool,
			env,
			VK_FORMAT_R8G8B8A8_UNORM,
			VK_IMAGE_USAGE_SAMPLED_BIT
				| VK_IMAGE_USAGE_TRANSFER_DST_BIT,
			VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
		);

		// TODO: do this inside make_texture method automatically
		// TODO: constructor for texture packet instead of this garbage
		tp.transition_manual(_context,
			_command_pool,
			VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
			VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
			VK_PIPELINE_STAGE_TRANSFER_BIT,
			VK_PIPELINE_STAGE_TRANSFER_BIT
		);

		// Construct and rebind
		_env_sampler = Sampler(_context, tp);
		_env_sampler.bind(_mesh_ds, MESH_BINDING_ENVIRONMENT);
	}

	// Clearning all data
	void clear() override {
		// Call parents clear
		kobra::Layer <_element> ::clear();

		// Clear all the buffers
		// _pixels.clear();
		_vertices.clear();
		_triangles.clear();
		_materials.clear();
		_lights.clear();
		_light_indices.clear();
		_transforms.clear();
		_bvh.clear();
	}

	// Number of triangles
	size_t triangle_count() const {
		return _triangles.push_size();
	}

	// Number of cameras
	size_t camera_count() const {
		return _cameras.size();
	}

	// Add a camera to the layer
	void add_camera(const Camera &camera) {
		_cameras.push_back(camera);
	}

	// Active camera
	Camera *active_camera() {
		return _active_camera;
	}

	// Activate a camera
	Camera *activate_camera(size_t index) {
		if (index < _cameras.size()) {
			_active_camera = &_cameras[index];
		} else {
			KOBRA_LOG_FUNC(warn) << "Camera index out of range ["
				<< index << "/" << _cameras.size() << "]";
		}

		return _active_camera;
	}

	// Set active camera
	void set_active_camera(const Camera &camera) {
		// If active camera has not been set
		if (_active_camera == nullptr) {
			if (_cameras.empty())
				_cameras.push_back(camera);

			_active_camera = &_cameras[0];
		}

		*_active_camera = camera;
	}

	// Get pixel buffer
	const BufferManager <uint> &pixels() {
		return _pixels;
	}

	// Render
	void render(const VkCommandBuffer &,
			const VkFramebuffer &,
			const BatchIndex &);
};

}

}

#endif

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

	// Initialize pipelines
	void _init_mesh_compute_pipeline();
	void _init_postproc_pipeline(const Vulkan::Swapchain &);
	void _init_pipelines(const Vulkan::Swapchain &);

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

	// Update descriptor sets for samplers
	void _update_samplers(const ImageDescriptors &, uint);

	// BVH
	BVH			_bvh;

	// Batching variables
	int			_xbatch = 0;
	int			_ybatch = 0;

	int			_xbatch_size = 50;
	int			_ybatch_size = 50;

	// Get list of bboxes for each triangle
	std::vector <BoundingBox> _get_bboxes() const;
public:
	// Default constructor
	Layer() = default;

	// Constructor
	Layer(const App::Window &);

	// Adding elements
	void add_do(const ptr &) override;
	void add_scene(const Scene &) override;

	// Set environment map
	void set_environment_map(const Texture &);

	// Clearning all data
	void clear() override;

	// Count methods
	size_t triangle_count() const;
	size_t camera_count() const;
	
	// Camera things
	void add_camera(const Camera &);
	Camera *active_camera();
	Camera *activate_camera(size_t);
	void set_active_camera(const Camera &);

	// Other getters
	const BufferManager <uint> &pixels();

	// Render
	void render(const VkCommandBuffer &,
			const VkFramebuffer &,
			const BatchIndex &);
};

}

}

#endif

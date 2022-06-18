#ifndef KOBRA_RT_LAYER_H_
#define KOBRA_RT_LAYER_H_

// Standard headers
#include <vector>

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
#include "bvh.hpp"
#include "rt.hpp"

namespace kobra {

namespace rt {

// Layer class
class Layer : public kobra::Layer <rt::_element> {
public:
	// Current mode
	enum Mode {
		NORMALS,
		HEATMAP,
		FAST_PATH_TRACER,
		PATH_TRACER,
		MIS_PATH_TRACER,
		BIDIRECTIONAL_PATH_TRACE
	};
protected:
	// Multiprocessing compute batches
	const int PARALLELIZATION = 12;

	// Camera things
	std::vector <Camera>		_cameras;
	Camera				*_active_camera = nullptr;

	// Vulkan structures
	const vk::raii::PhysicalDevice	&_physical_device = nullptr;
	const vk::raii::Device		&_device = nullptr;
	const vk::raii::CommandPool	&_command_pool = nullptr;
	const vk::raii::DescriptorPool	&_descriptor_pool = nullptr;

	vk::raii::RenderPass		_render_pass = nullptr;
	vk::Extent2D			_extent;

	// Resources for multiprocessing
	std::vector <vk::raii::Queue>	_queues;
	std::vector <vk::raii::CommandBuffer>
					_command_buffers;
	std::vector <vk::raii::Fence>	_fences;

	// Pipelines
	struct {
		// Pipelines
		vk::raii::Pipeline normals = nullptr;
		vk::raii::Pipeline heatmap = nullptr;
		vk::raii::Pipeline fast_path_tracer = nullptr;
		vk::raii::Pipeline path_tracer = nullptr;
		vk::raii::Pipeline mis_path_tracer = nullptr;
		vk::raii::Pipeline bidirectional_path_tracer = nullptr;
		vk::raii::Pipeline postprocess = nullptr;

		// Common pipeline layouts
		vk::raii::PipelineLayout raytracing_layout = nullptr;
		vk::raii::PipelineLayout postprocess_layout = nullptr;
	} _pipelines;

	// Current mode
	Mode _mode = PATH_TRACER;

	// Get active pipeline
	const vk::raii::Pipeline &_active_pipeline() {
		switch (_mode) {
		case NORMALS:
			return _pipelines.normals;
		case HEATMAP:
			return _pipelines.heatmap;
		case FAST_PATH_TRACER:
			return _pipelines.fast_path_tracer;
		case PATH_TRACER:
			return _pipelines.path_tracer;
		case MIS_PATH_TRACER:
			return _pipelines.mis_path_tracer;
		case BIDIRECTIONAL_PATH_TRACE:
			return _pipelines.bidirectional_path_tracer;
		default:
			break;
		}

		throw std::runtime_error("Invalid mode");
		return _pipelines.normals;
	}

	// Descriptor set layouts
	vk::raii::DescriptorSetLayout 	_dsl_raytracing = nullptr;
	vk::raii::DescriptorSetLayout	_dsl_postprocess = nullptr;

	// Descriptor sets
	vk::raii::DescriptorSet		_dset_raytracing = nullptr;
	vk::raii::DescriptorSet		_dset_postprocess = nullptr;

	// Descriptor set bindings
	static const std::vector <DSLB>	_raytracing_bindings;
	static const std::vector <DSLB>	_postproc_bindings;

	// Initialize pipelines
	void _init_compute_pipelines();
	void _init_postprocess_pipeline();
	void _init_pipelines();

	// Device buffer data
	struct {
		BufferData		pixels = nullptr;
		
		BufferData		vertices = nullptr;
		BufferData		triangles = nullptr;
		BufferData		materials = nullptr;
		
		BufferData		lights = nullptr;
		BufferData		light_indices = nullptr;
		
		BufferData		transforms = nullptr;

		// Bind to respective descriptor set
		void bind(const vk::raii::Device &device,
				const vk::raii::DescriptorSet &dset_raytracing) {
			bind_ds(device,
				dset_raytracing, vertices,
				vk::DescriptorType::eStorageBuffer,
				MESH_BINDING_VERTICES
			);

			bind_ds(device,
				dset_raytracing, triangles,
				vk::DescriptorType::eStorageBuffer,
				MESH_BINDING_TRIANGLES
			);

			bind_ds(device,
				dset_raytracing, materials,
				vk::DescriptorType::eStorageBuffer,
				MESH_BINDING_MATERIALS
			);

			bind_ds(device,
				dset_raytracing, lights,
				vk::DescriptorType::eStorageBuffer,
				MESH_BINDING_LIGHTS
			);

			bind_ds(device,
				dset_raytracing, light_indices,
				vk::DescriptorType::eStorageBuffer,
				MESH_BINDING_LIGHT_INDICES
			);

			bind_ds(device,
				dset_raytracing, transforms,
				vk::DescriptorType::eStorageBuffer,
				MESH_BINDING_TRANSFORMS
			);
		}
	} _dev;

	// Host buffer data
	struct {
		std::vector <aligned_vec4>	vertices;
		std::vector <aligned_vec4>	triangles;
		std::vector <aligned_vec4>	materials;

		std::vector <aligned_vec4>	lights;
		std::vector <uint>		light_indices;
		
		std::vector <aligned_mat4>	transforms; // TODO: is this needed?
	} _host;

	// Samplers and their images
	struct {
		// Original image data
		ImageData		empty_image = nullptr;
		ImageData		environment_image = nullptr;
		ImageData		result_image = nullptr;

		// Corresponding samplers
		// TODO: wrapper Sampler class with imagedata & sampler
		vk::raii::Sampler	empty = nullptr;
		vk::raii::Sampler	environment = nullptr;
		vk::raii::Sampler	result = nullptr;
	} _samplers;

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

	// Initialization status
	bool			_initialized = false;

	// Get list of bboxes for each triangle
	std::vector <BoundingBox> _get_bboxes() const;

	// Launch an RT batch kernel
	void _launch_kernel(uint32_t, const Batch &, const BatchIndex &bi);
public:
	// Default constructor
	Layer() = default;

	// Constructor
	// TODO: these arguments should be wrapped in a struct
	Layer(const vk::raii::PhysicalDevice &,
			const vk::raii::Device &,
			const vk::raii::CommandPool &,
			const vk::raii::DescriptorPool &,
			const vk::Extent2D &,
			const vk::Format &,
			const vk::Format &,
			const vk::AttachmentLoadOp & = vk::AttachmentLoadOp::eLoad);

	// Adding elements
	void add_do(const ptr &) override;
	void add_scene(const Scene &) override;

	// Set current mode
	void set_mode(Mode mode) {
		_mode = mode;
	}

	// Set environment map
	void set_environment_map(ImageData &&);

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
	const BufferData &pixels();

	// Display memory footprint
	void display_memory_footprint() const;

	// Render
	void render(const vk::raii::CommandBuffer &,
			const vk::raii::Framebuffer &,
			const vk::Extent2D &,
			Batch &,
			BatchIndex &);
};

}

}

#endif

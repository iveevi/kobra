#ifndef KOBRA_LAYERS_RAYTRACER_H_
#define KOBRA_LAYERS_RAYTRACER_H_

// Standard headers
#include <vector>

// Engine headers
#include "../ecs.hpp"
#include "../backend.hpp"
#include "../../shaders/rt/bindings.h"

namespace kobra {

namespace layers {

class Raytracer {
	// Vulkan context
	Context _ctx;

	// Other Vulkan structures
	vk::raii::RenderPass		_render_pass = nullptr;

	vk::raii::PipelineLayout	_ppl_raytracing = nullptr;
	vk::raii::PipelineLayout	_ppl_postprocess = nullptr;

	vk::raii::Pipeline		_p_raytracing = nullptr;
	vk::raii::Pipeline		_p_postprocess = nullptr;

	// Device buffer data
	struct DeviceBufferData;
	struct {
		BufferData		pixels = nullptr;

		BufferData		bvh = nullptr;

		BufferData		vertices = nullptr;
		BufferData		triangles = nullptr;
		BufferData		materials = nullptr;

		BufferData		area_lights = nullptr;

		// TODO: We have a use for this now
		BufferData		transforms = nullptr;

		// Bind to respective descriptor set
		void bind(const vk::raii::Device &device,
				const vk::raii::DescriptorSet &dset_raytracing) {
			bind_ds(device,
				dset_raytracing, bvh,
				vk::DescriptorType::eStorageBuffer,
				MESH_BINDING_BVH
			);

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
				dset_raytracing, area_lights,
				vk::DescriptorType::eStorageBuffer,
				MESH_BINDING_AREA_LIGHTS
			);

			bind_ds(device,
				dset_raytracing, transforms,
				vk::DescriptorType::eStorageBuffer,
				MESH_BINDING_TRANSFORMS
			);
		}
	} _dev;

	// Descriptor set, layout and bindnigs
	vk::raii::DescriptorSet		_ds_raytracing = nullptr;
	vk::raii::DescriptorSet		_ds_postprocess = nullptr;

	vk::raii::DescriptorSetLayout	_dsl_raytracing = nullptr;
	vk::raii::DescriptorSetLayout	_dsl_postprocess = nullptr;

	static const std::vector <DSLB> _dslb_raytracing;
	static const std::vector <DSLB> _dslb_postprocess;

	// TODO: image2D instead of buffer

	// Samplers and their images
	// TODO: cleaner workaround?
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

	using ImageDescriptors = std::vector <vk::DescriptorImageInfo>;

	// Vector of image descriptors
	ImageDescriptors	_albedo_image_descriptors;
	ImageDescriptors	_normal_image_descriptors;

	// Reference to SyncQueue
	SyncQueue			*_sync_queue;

	// Accumulation status
	// TODO: struct, then methods to reset
	int		_accumulated = 0;
	Transform	_ptransform;
	int		_skip = 3;	// Skip batch size (per dim)
	int		_offsetx = 0;
	int		_offsety = 0;

	// TODO: the following should be kept in a cache structure
	std::vector <Transform> _p_light_transforms;
	std::vector <const kobra::Raytracer *> _p_raytracers;

	// Helper functions
	void _initialize_vuklan_structures(const vk::AttachmentLoadOp &);
	std::vector <BoundingBox> _get_bboxes(const kobra::Raytracer::HostBuffers &) const;
	void _update_samplers(const ImageDescriptors &, uint32_t);
public:
	// Default constructor
	Raytracer() = default;

	// Constructors
	Raytracer(const Context &, SyncQueue *, const vk::AttachmentLoadOp &);

	// Methods
	void environment_map(const std::string &);

	// Render
	void render(const vk::raii::CommandBuffer &,
			const vk::raii::Framebuffer &,
			const ECS &, const RenderArea & = {{-1, -1}, {-1, -1}});
};

}

}

#endif

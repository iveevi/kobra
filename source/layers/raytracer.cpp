// Engine headers
#include "../../include/layers/raytracer.hpp"
#include "../../include/profiler.hpp"
#include "../../include/texture_manager.hpp"

namespace kobra {

namespace layers {

//////////////////////
// Static variables //
//////////////////////

const std::vector <DSLB> Raytracer::_dslb_raytracing {
	DSLB {
		MESH_BINDING_PIXELS,
		vk::DescriptorType::eStorageBuffer,
		1, vk::ShaderStageFlagBits::eCompute,
	},

	DSLB {
		MESH_BINDING_VERTICES,
		vk::DescriptorType::eStorageBuffer,
		1, vk::ShaderStageFlagBits::eCompute,
	},

	DSLB {
		MESH_BINDING_TRIANGLES,
		vk::DescriptorType::eStorageBuffer,
		1, vk::ShaderStageFlagBits::eCompute,
	},

	DSLB {
		MESH_BINDING_TRANSFORMS,
		vk::DescriptorType::eStorageBuffer,
		1, vk::ShaderStageFlagBits::eCompute,
	},

	DSLB {
		MESH_BINDING_BVH,
		vk::DescriptorType::eStorageBuffer,
		1, vk::ShaderStageFlagBits::eCompute,
	},

	// Materials buffer
	DSLB {
		MESH_BINDING_MATERIALS,
		vk::DescriptorType::eStorageBuffer,
		1, vk::ShaderStageFlagBits::eCompute,
	},

	DSLB {
		MESH_BINDING_AREA_LIGHTS,
		vk::DescriptorType::eStorageBuffer,
		1, vk::ShaderStageFlagBits::eCompute,
	},

	// Lights buffer
	DSLB {
		MESH_BINDING_LIGHTS,
		vk::DescriptorType::eStorageBuffer,
		1, vk::ShaderStageFlagBits::eCompute,
	},

	// Light indices
	DSLB {
		MESH_BINDING_LIGHT_INDICES,
		vk::DescriptorType::eStorageBuffer,
		1, vk::ShaderStageFlagBits::eCompute,
	},

	// Texture samplers
	DSLB {
		MESH_BINDING_ALBEDOS,
		vk::DescriptorType::eCombinedImageSampler,
		MAX_TEXTURES, vk::ShaderStageFlagBits::eCompute,
	},

	DSLB {
		MESH_BINDING_NORMAL_MAPS,
		vk::DescriptorType::eCombinedImageSampler,
		MAX_TEXTURES, vk::ShaderStageFlagBits::eCompute,
	},

	DSLB {
		MESH_BINDING_ENVIRONMENT,
		vk::DescriptorType::eCombinedImageSampler,
		1, vk::ShaderStageFlagBits::eCompute,
	},

	DSLB {
		MESH_BINDING_OUTPUT,
		vk::DescriptorType::eStorageBuffer,
		1, vk::ShaderStageFlagBits::eCompute,
	},
};

const std::vector <DSLB> Raytracer::_dslb_postprocess = {
	DSLB {
		0, vk::DescriptorType::eCombinedImageSampler,
		1, vk::ShaderStageFlagBits::eFragment
	}
};

////////////////////
// Aux structures //
////////////////////

struct _area_light {
	alignas(16) glm::vec3 a;
	alignas(16) glm::vec3 ab;
	alignas(16) glm::vec3 ac;
	alignas(16) glm::vec3 color;
	float power;
};

struct _area_light_info {
	int count;

	_area_light lights[32];
};

struct PushConstants {
	alignas(16)
	uint width;
	uint height;

	uint skip;
	uint xoffset;
	uint yoffset;

	uint triangles;
	uint lights;
	uint samples_per_pixel;
	uint samples_per_surface;
	uint samples_per_light;

	uint accumulate;
	uint present;
	uint total;

	float time;

	aligned_vec4 camera_position;
	aligned_vec4 camera_forward;
	aligned_vec4 camera_up;
	aligned_vec4 camera_right;

	aligned_vec4 camera_tunings;
};

struct Viewport {
	uint width;
	uint height;
};

/////////////////
// Constructor //
/////////////////

Raytracer::Raytracer(const Context &ctx, SyncQueue *sq, const vk::AttachmentLoadOp &load)
		: _ctx(ctx), _sync_queue(sq)
{
	_initialize_vuklan_structures(load);

	// Create buffers
	const auto &phdev = *_ctx.phdev;
	const auto &device = *_ctx.device;

	vk::DeviceSize uint_size = 1024 * sizeof(uint);
	vk::DeviceSize vec4_size = 1024 * sizeof(aligned_vec4);
	vk::DeviceSize mat4_size = 1024 * sizeof(aligned_mat4);
	vk::DeviceSize pixels_size = _ctx.extent.width * _ctx.extent.height * sizeof(uint);

	auto usage = vk::BufferUsageFlagBits::eStorageBuffer;
	auto mem_props = vk::MemoryPropertyFlagBits::eDeviceLocal
		| vk::MemoryPropertyFlagBits::eHostCoherent
		| vk::MemoryPropertyFlagBits::eHostVisible;

	_dev.pixels = BufferData(
		phdev, device, pixels_size,
		usage | vk::BufferUsageFlagBits::eTransferSrc, mem_props
	);

	_dev.bvh = BufferData(phdev, device, vec4_size, usage, mem_props);
	_dev.vertices = BufferData(phdev, device, vec4_size, usage, mem_props);
	_dev.triangles = BufferData(phdev, device, vec4_size, usage, mem_props);
	_dev.materials = BufferData(phdev, device, vec4_size, usage, mem_props);

	_dev.area_lights = BufferData(phdev, device, sizeof(_area_light_info), usage, mem_props);

	_dev.transforms = BufferData(phdev, device, mat4_size, usage, mem_props);

	// Bind to descriptor sets
	bind_ds(*_ctx.device, _ds_raytracing, _dev.pixels, vk::DescriptorType::eStorageBuffer, MESH_BINDING_PIXELS);

	_dev.bind(*_ctx.device, _ds_raytracing);

	// Sampler source images
	_samplers.empty_image = ImageData::blank(phdev, device);
	_samplers.environment_image = ImageData::blank(phdev, device);

	// Initialize samplers
	_samplers.empty = make_sampler(device, _samplers.empty_image);
	_samplers.environment = make_sampler(device, _samplers.environment_image);

	// Output image and sampler
	_samplers.result_image = ImageData(
		phdev, device,
		vk::Format::eR8G8B8A8Unorm, _ctx.extent,
		vk::ImageTiling::eOptimal,
		vk::ImageUsageFlagBits::eSampled
			| vk::ImageUsageFlagBits::eTransferDst,
		vk::ImageLayout::ePreinitialized,
		vk::MemoryPropertyFlagBits::eDeviceLocal,
		vk::ImageAspectFlagBits::eColor
	);

	_samplers.result = make_sampler(device, _samplers.result_image);

	// Transition image layout
	auto tmp_cmd = make_command_buffer(device, *_ctx.command_pool);
	auto queue = vk::raii::Queue {device, 0, 0};

	{
		tmp_cmd.begin({});
		_samplers.environment_image.transition_layout(tmp_cmd, vk::ImageLayout::eShaderReadOnlyOptimal);
		_samplers.empty_image.transition_layout(tmp_cmd, vk::ImageLayout::eShaderReadOnlyOptimal);
		tmp_cmd.end();
	}

	queue.submit(
		vk::SubmitInfo {
			0, nullptr, nullptr, 1, &*tmp_cmd
		},
		nullptr
	);

	queue.waitIdle();

	// Binding environment and result sampler
	bind_ds(device,
		_ds_raytracing,
		_samplers.environment,
		_samplers.environment_image,
		MESH_BINDING_ENVIRONMENT
	);

	bind_ds(device,
		_ds_raytracing,
		_dev.pixels,
		vk::DescriptorType::eStorageBuffer,
		MESH_BINDING_PIXELS
	);

	bind_ds(device,
		_ds_postprocess,
		_samplers.result,
		_samplers.result_image,
		MESH_BINDING_PIXELS
	);

	// Albedos
	while (_albedo_image_descriptors.size() < MAX_TEXTURES) {
		_albedo_image_descriptors.push_back(
			vk::DescriptorImageInfo(
				*_samplers.empty,
				*_samplers.empty_image.view,
				vk::ImageLayout::eShaderReadOnlyOptimal
			)
		);
	}

	_update_samplers(_albedo_image_descriptors, MESH_BINDING_ALBEDOS);

	// Normals
	while (_normal_image_descriptors.size() < MAX_TEXTURES) {
		_normal_image_descriptors.push_back(
			vk::DescriptorImageInfo(
				*_samplers.empty,
				*_samplers.empty_image.view,
				vk::ImageLayout::eShaderReadOnlyOptimal
			)
		);
	}

	_update_samplers(_normal_image_descriptors, MESH_BINDING_NORMAL_MAPS);
}

/////////////
// Methods //
/////////////

// TODO: also HDR maps
void Raytracer::environment_map(const std::string &path)
{
	// Reset accumulation status
	_accumulated = 0;
	_offsetx = 0;
	_offsety = 0;

	// Load environment map
	// TODO: remove sampler stuff and rely on texturemanage
	/* _samplers.environment_image = TextureManager::load_texture(
		*_ctx.phdev, *_ctx.device, path); */

	TextureManager::bind(
		*_ctx.phdev, *_ctx.device,
		_ds_raytracing,
		path, MESH_BINDING_ENVIRONMENT
	);
}

////////////
// Render //
////////////

void Raytracer::render(const vk::raii::CommandBuffer &cmd,
		const vk::raii::Framebuffer &framebuffer,
		const ECS &ecs, const RenderArea &ra)
{
	// Check if we need to resize pixel buffer
	uint32_t csize = ra.pixels() * sizeof(uint);
	if (_dev.pixels.size != csize) {
		KOBRA_LOG_FUNC(warn) << "Pixel buffer size changed, resizing...\n";

		// Resize buffer
		_dev.pixels.resize(csize);

		// Resize result image
		_samplers.result_image = ImageData(
			*_ctx.phdev, *_ctx.device,
			vk::Format::eR8G8B8A8Unorm, {ra.width(), ra.height()},
			vk::ImageTiling::eOptimal,
			vk::ImageUsageFlagBits::eSampled
				| vk::ImageUsageFlagBits::eTransferDst,
			vk::ImageLayout::ePreinitialized,
			vk::MemoryPropertyFlagBits::eDeviceLocal,
			vk::ImageAspectFlagBits::eColor
		);

		_samplers.result = make_sampler(*_ctx.device, _samplers.result_image);

		// Rebind
		_sync_queue->push(
			[&]() {
				bind_ds(*_ctx.device,
					_ds_raytracing, _dev.pixels,
					vk::DescriptorType::eStorageBuffer,
					MESH_BINDING_PIXELS
				);

				bind_ds(*_ctx.device,
					_ds_postprocess, _samplers.result,
					_samplers.result_image,
					MESH_BINDING_PIXELS
				);
			}
		);

		// Return and wait for the next frame
		return;
	}

	// Profiler
	Profiler profiler;

	// Initialization phase
	Camera camera;
	Transform camera_transform;
	bool found_camera = false;

	kobra::Raytracer::HostBuffers host_buffers {
		.albedo_textures = _albedo_image_descriptors,
		.normal_textures = _normal_image_descriptors,
		.id = 1
	};

	int light_index = 0;
	int raytracers_index = 0;
	bool dirty_lights = false;
	bool dirty_raytracers = false;
	std::vector <Transform> light_transforms;
	std::vector <const kobra::Raytracer *> raytracers;
	std::vector <Transform> raytracer_transforms;

	_area_light_info alight_info {.count = 0};

	profiler.frame("Raytracer frame");
	profiler.frame("Iterating through entities");

	// TODO: this loop ccan be parallelised
	for (int i = 0; i < ecs.size(); i++) {
		// TODO: how to avoid constructing BVH every single
		// frame?
		// GPU construction?
		// NOTE: CudaRaytracer will rely on OptiX for this

		// Deal with camera component
		if (ecs.exists <Camera> (i)) {
			camera = ecs.get <Camera> (i);
			camera_transform = ecs.get <Transform> (i);
			found_camera = true;
		}

		if (ecs.exists <kobra::Raytracer> (i)) {
			// TODO: account for changing transforms
			const kobra::Raytracer *raytracer = &ecs.get <kobra::Raytracer> (i);

			if (raytracers_index >= _p_raytracers.size())
				dirty_raytracers = true;
			else if (_p_raytracers[raytracers_index] != raytracer)
				dirty_raytracers = true;

			raytracers.push_back(raytracer);

			const Transform &transform = ecs.get <Transform> (i);

			raytracer_transforms.push_back(transform);
			raytracers_index++;
		}

		// Deal with light
		// TODO: methods...
		if (ecs.exists <Light> (i)) {
			const Light &light = ecs.get <Light> (i);

			const Transform &transform = ecs.get <Transform> (i);

			// Check if lights have moved
			// TODO: also check if color has changed (i.e. same
			// strategy as for raytracers)
			if (light_index >= _p_light_transforms.size())
				dirty_lights = true;
			else if (transform != _p_light_transforms[light_index])
				dirty_lights = true;

			light_transforms.push_back(transform);
			light_index++;

			// Area light
			if (light.type == Light::Type::eArea) {
				profiler.frame("Serializing area light");

				// New vertices (square 1x1 in center)
				glm::vec3 a {-0.5f, 0, -0.5f};
				glm::vec3 b {0.5f, 0, -0.5f};
				glm::vec3 c {-0.5f, 0, 0.5f};

				a = transform.apply(a);
				b = transform.apply(b);
				c = transform.apply(c);

				// TODO: this only applies if the light is an area light
				_area_light alight;
				alight.a = a;
				alight.ab = b - a;
				alight.ac = c - a;
				alight.color = light.color;
				alight.power = light.power;

				alight_info.lights[alight_info.count++] = alight;

				profiler.end();
			}
		}
	}

	profiler.end();

	if (!found_camera) {
		// Actually just skip
		throw std::runtime_error("No camera found");
	}

	// Upload to device buffers
	bool rebinding = false;

	profiler.frame("Updating buffers");
	if (dirty_raytracers) {
		KOBRA_LOG_FILE(warn) << "Raytracer components have been modified, rebuilding...\n";

		profiler.frame("Rebuilding raytracers");

		profiler.frame("Serializing");

		for (int i = 0; i < raytracers.size(); i++) {
			profiler.frame("Serializing raytracer component");

			raytracers[i]->serialize({_ctx.phdev, _ctx.device},
				raytracer_transforms[i],
				host_buffers
			);

			profiler.end();
		}

		profiler.end();

		KOBRA_LOG_FILE(notify) << "Uploading data to device buffers...\n";
		rebinding |= _dev.vertices.upload(host_buffers.vertices, 0);
		rebinding |= _dev.triangles.upload(host_buffers.triangles, 0);
		rebinding |= _dev.materials.upload(host_buffers.materials, 0);
		rebinding |= _dev.transforms.upload(host_buffers.transforms, 0);

		// TODO: some way to combine bvhs of multiple objects (id
		// clashing...)
		profiler.frame("Constructing BVH");
		auto bboxes = _get_bboxes(host_buffers);
		auto bvh = partition(bboxes);

		serialize(host_buffers.bvh, bvh);
		profiler.end();

		rebinding |= _dev.bvh.upload(host_buffers.bvh, 0);

		profiler.end();
	}

	rebinding |= _dev.area_lights.upload(&alight_info, sizeof(alight_info));
	profiler.end();

	profiler.end();
	// std::cout << profiler.pretty(profiler.pop()) << std::endl;

	if (rebinding) {
		KOBRA_LOG_FILE(warn) << "Rebinding device buffers\n";
		_sync_queue->push(
			[&]() {
				_dev.bind(*_ctx.device, _ds_raytracing);
			}
		);

		// TODO: check if desciptors actually changed
		_sync_queue->push(
			[&]() {
				_update_samplers(_albedo_image_descriptors, MESH_BINDING_ALBEDOS);
				_update_samplers(_normal_image_descriptors, MESH_BINDING_NORMAL_MAPS);
			}
		);

		return;
	}

	// Compute shader
	cmd.bindPipeline(vk::PipelineBindPoint::eCompute, *_p_raytracing);

	// Time as float
	unsigned int time = static_cast <unsigned int>
		(std::chrono::duration_cast
			<std::chrono::milliseconds>
			(std::chrono::system_clock::now().time_since_epoch()).count());

	// Dirty means reset samples
	bool dirty = (_ptransform != camera_transform);
	if (dirty || dirty_lights) {
		_accumulated = 0;
		_offsetx = 0;
		_offsety = 0;
	}

	_ptransform = camera_transform;

	_p_light_transforms = light_transforms;
	_p_raytracers = raytracers;

	// TODO: using progressive rendering, we can skip pixels (every
	// other) and then increment samples after each complete pass.

	// Prepare push constants
	// TODO: method for generating this structure
	PushConstants pc {
		.width = ra.width(),
		.height = ra.height(),

		// TODO: pass skpi size
		.skip = (uint) _skip,
		.xoffset = (uint) _offsetx,
		.yoffset = (uint) _offsety,

		.triangles = (uint) host_buffers.triangles.size(),
		.lights = (uint) host_buffers.light_indices.size(),

		// TODO: still unable to do large number of samples
		.samples_per_pixel = 1,
		.samples_per_surface = 1,
		.samples_per_light = 1,

		.accumulate = 0,
		.present = (uint) _accumulated,
		.total = 1,

		// Pass current time (as float)
		.time = (float) time,

		// TODO: uvw instead?
		.camera_position = camera_transform.position,
		.camera_forward = camera_transform.forward(),
		.camera_up = camera_transform.up(),
		.camera_right = camera_transform.right(),

		.camera_tunings = glm::vec4 {
			tan(glm::radians(camera.fov) / 2.0f),
			camera.aspect,
			0, 0
		}
	};

	// Bind descriptor set
	cmd.bindDescriptorSets(
		vk::PipelineBindPoint::eCompute,
		*_ppl_raytracing,
		0, { *_ds_raytracing }, {}
	);

	// Push constants
	cmd.pushConstants <PushConstants> (
		*_ppl_raytracing,
		vk::ShaderStageFlagBits::eCompute,
		0, pc
	);

	// Dispatch the compute shader
	cmd.dispatch(_ctx.extent.width/_skip, _ctx.extent.height/_skip, 1);

	// Update accumulation status'
	_offsetx++;
	if (_offsetx >= _skip) {
		_offsetx = 0;
		_offsety++;
	}

	if (_offsety >= _skip) {
		_offsety = 0;
		_accumulated++;
	}

	// Transition the result image to transfer destination
	// TODO: pass extent
	_samplers.result_image.transition_layout(cmd, vk::ImageLayout::eTransferDstOptimal);

	copy_data_to_image(cmd,
		_dev.pixels.buffer,
		_samplers.result_image.image,
		_samplers.result_image.format,
		ra.width(), ra.height()
	);

	// Transition image back to shader read
	_samplers.result_image.transition_layout(cmd, vk::ImageLayout::eShaderReadOnlyOptimal);

	// Apply render area
	ra.apply(cmd, _ctx.extent);

	// Clear colors
	std::array <vk::ClearValue, 2> clear_values {
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

	// Start the render pass
	cmd.beginRenderPass(
		vk::RenderPassBeginInfo {
			*_render_pass,
			*framebuffer,
			vk::Rect2D {
				vk::Offset2D {0, 0},
				_ctx.extent
			},
			static_cast <uint32_t> (clear_values.size()),
			clear_values.data()
		},
		vk::SubpassContents::eInline
	);

	// Post process pipeline
	cmd.bindPipeline(
		vk::PipelineBindPoint::eGraphics,
		*_p_postprocess
	);

	// Bind descriptor set
	cmd.bindDescriptorSets(
		vk::PipelineBindPoint::eGraphics,
		*_ppl_postprocess,
		0, {*_ds_postprocess}, {}
	);

	// Draw and end
	cmd.draw(6, 1, 0, 0);
	cmd.endRenderPass();
}

/////////////////////
// Private methods //
/////////////////////

void Raytracer::_initialize_vuklan_structures(const vk::AttachmentLoadOp &load)
{
	// Create render pass
	_render_pass = make_render_pass(*_ctx.device,
		_ctx.swapchain_format,
		_ctx.depth_format, load
	);

	// Create descriptor set layouts
	_dsl_raytracing = make_descriptor_set_layout(*_ctx.device, _dslb_raytracing);
	_dsl_postprocess = make_descriptor_set_layout(*_ctx.device, _dslb_postprocess);

	// Create descriptor sets
	std::array <vk::DescriptorSetLayout, 2> dsls {
		*_dsl_raytracing, *_dsl_postprocess
	};

	auto dsets = vk::raii::DescriptorSets {
		*_ctx.device,
		{**_ctx.descriptor_pool, dsls}
	};

	_ds_raytracing = std::move(dsets.front());
	_ds_postprocess = std::move(dsets.back());

	// Load all shaders
	auto shaders = make_shader_modules(*_ctx.device, {
		"shaders/bin/generic/progressive_path_tracer.spv",
		"shaders/bin/generic/postproc_vert.spv",
		"shaders/bin/generic/postproc_frag.spv"
	});

	// RT compute pipeline
	auto pcr = vk::PushConstantRange {
		vk::ShaderStageFlagBits::eCompute,
		0, sizeof(PushConstants)
	};

	_ppl_raytracing = vk::raii::PipelineLayout {
		*_ctx.device,
		{{}, *_dsl_raytracing, pcr}
	};

	vk::PipelineShaderStageCreateInfo rt_stage {
		vk::PipelineShaderStageCreateFlags(),
		vk::ShaderStageFlagBits::eCompute,
		*shaders[0], "main"
	};

	vk::ComputePipelineCreateInfo rt_pipeline_info {
		{}, {}, *_ppl_raytracing, nullptr
	};

	rt_pipeline_info.stage = rt_stage;

	_p_raytracing = vk::raii::Pipeline {
		*_ctx.device,
		vk::raii::PipelineCache {
			*_ctx.device,
			vk::PipelineCacheCreateInfo {}
		},
		rt_pipeline_info
	};

	// Postprocess pipeline
	pcr = vk::PushConstantRange {
		vk::ShaderStageFlagBits::eVertex,
		0, sizeof(Viewport)
	};

	_ppl_postprocess = vk::raii::PipelineLayout {
		*_ctx.device,
		{{}, *_dsl_postprocess, pcr}
	};
		
	GraphicsPipelineInfo grp_info(*_ctx.device, _render_pass,
		std::move(shaders[1]), nullptr,
		std::move(shaders[2]), nullptr,
		{}, {},
		_ppl_postprocess, vk::raii::PipelineCache {
			*_ctx.device,
			vk::PipelineCacheCreateInfo {}
		}
	);

	_p_postprocess = make_graphics_pipeline(grp_info);
}

std::vector <BoundingBox> Raytracer::_get_bboxes(const kobra::Raytracer::HostBuffers &hb) const
{
	std::vector <BoundingBox> bboxes;
	bboxes.reserve(hb.triangles.size());

	const auto &vertices = hb.vertices;
	const auto &triangles = hb.triangles;

	for (size_t i = 0; i < hb.triangles.size(); i++) {
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
			glm::vec4 center = vertices[VERTEX_STRIDE * a].data;
			float radius = center.w;

			glm::vec4 min = center - glm::vec4(radius);
			glm::vec4 max = center + glm::vec4(radius);

			bboxes.push_back(BoundingBox {min, max, int(d)});
		} else {
			glm::vec4 va = vertices[VERTEX_STRIDE * a].data;
			glm::vec4 vb = vertices[VERTEX_STRIDE * b].data;
			glm::vec4 vc = vertices[VERTEX_STRIDE * c].data;

			glm::vec4 min = glm::min(va, glm::min(vb, vc));
			glm::vec4 max = glm::max(va, glm::max(vb, vc));

			bboxes.push_back(BoundingBox {min, max, int(d)});
		}
	}

	return bboxes;
}

void Raytracer::_update_samplers(const ImageDescriptors &descriptors, uint32_t binding)
{
	// Update descriptor set
	vk::WriteDescriptorSet dset_write {
		*_ds_raytracing,
		binding, 0,
		vk::DescriptorType::eCombinedImageSampler,
		descriptors
	};

	_ctx.device->updateDescriptorSets(dset_write, nullptr);
}

}

}

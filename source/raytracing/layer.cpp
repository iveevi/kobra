// Vulkan headers
#include <vulkan/vulkan_core.h>

// Engine headers
#include "../../include/raytracing/layer.hpp"
#include "../../include/raytracing/sphere.hpp"
#include "../../include/raytracing/mesh.hpp"

namespace kobra {

namespace rt {

/////////////////////////////
// Static member variables //
/////////////////////////////

const std::vector <DSLB> Layer::_raytracing_bindings {
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

const std::vector <DSLB> Layer::_postproc_bindings = {
	DSLB {
		0, vk::DescriptorType::eCombinedImageSampler,
		1, vk::ShaderStageFlagBits::eFragment
	}
};

//////////////////////////////
// Private helper functions //
//////////////////////////////

void Layer::_init_compute_pipelines()
{
	// Push constants
	auto pcr = vk::PushConstantRange {
		vk::ShaderStageFlagBits::eCompute,
		0, sizeof(PushConstants)
	};

	// Common pipeline layout
	_pipelines.raytracing_layout = vk::raii::PipelineLayout {
		_device,
		{{}, *_dsl_raytracing, pcr}
	};

	// Get all the shaders
	auto shaders = make_shader_modules(_device, {
		"shaders/bin/generic/normal.spv",
		"shaders/bin/generic/heatmap.spv",
		"shaders/bin/generic/fast_path_tracer.spv",
		"shaders/bin/generic/pbr_path_tracer.spv",
		"shaders/bin/generic/mis_path_tracer.spv",
		"shaders/bin/generic/bidirectional_path_tracer.spv"
	});

	// Shader stages for each pipeline
	std::array <vk::PipelineShaderStageCreateInfo, 6> shader_stages {
		vk::PipelineShaderStageCreateInfo {
			{}, vk::ShaderStageFlagBits::eCompute,
			*shaders[0], "main"
		},

		vk::PipelineShaderStageCreateInfo {
			{}, vk::ShaderStageFlagBits::eCompute,
			*shaders[1], "main"
		},

		vk::PipelineShaderStageCreateInfo {
			{}, vk::ShaderStageFlagBits::eCompute,
			*shaders[2], "main"
		},

		vk::PipelineShaderStageCreateInfo {
			{}, vk::ShaderStageFlagBits::eCompute,
			*shaders[3], "main"
		},

		vk::PipelineShaderStageCreateInfo {
			{}, vk::ShaderStageFlagBits::eCompute,
			*shaders[4], "main"
		},

		vk::PipelineShaderStageCreateInfo {
			{}, vk::ShaderStageFlagBits::eCompute,
			*shaders[5], "main"
		}
	};

	// Compute pipeline creation info
	vk::ComputePipelineCreateInfo ppl_info {
		{}, {},
		*_pipelines.raytracing_layout,
		nullptr,
	};

	// Lambda to create the pipeline
	auto ppl_maker = [&](const vk::PipelineShaderStageCreateInfo &shader_stage) {
		// Set the shader stage
		ppl_info.stage = shader_stage;

		// Pipeline to return
		return vk::raii::Pipeline {
			_device,
			vk::raii::PipelineCache {
				_device,
				vk::PipelineCacheCreateInfo {}
			},
			ppl_info
		};
	};

	// Create the pipelines
	_pipelines.normals = ppl_maker(shader_stages[0]);
	_pipelines.heatmap = ppl_maker(shader_stages[1]);
	_pipelines.fast_path_tracer = ppl_maker(shader_stages[2]);
	_pipelines.path_tracer = ppl_maker(shader_stages[3]);
	_pipelines.mis_path_tracer = ppl_maker(shader_stages[4]);
	_pipelines.bidirectional_path_tracer = ppl_maker(shader_stages[5]);
}

void Layer::_init_postprocess_pipeline()
{
	// Load the shaders
	auto shaders = make_shader_modules(_device, {
		"shaders/bin/generic/postproc_vert.spv",
		"shaders/bin/generic/postproc_frag.spv"
	});

	// Push constants
	auto pcr = vk::PushConstantRange {
		vk::ShaderStageFlagBits::eVertex,
		0, sizeof(PC_Viewport) // TODO: Why is this needed?
	};

	// Create the pipeline layout
	_pipelines.postprocess_layout = vk::raii::PipelineLayout {
		_device,
		{{}, *_dsl_postprocess, pcr}
	};

	// Create pipeline cache
	auto pc = vk::raii::PipelineCache {
		_device,
		vk::PipelineCacheCreateInfo {}
	};

	// Create the graphics pipeline
	auto grp_info = GraphicsPipelineInfo {
		.device = _device,
		.render_pass = _render_pass,

		.vertex_shader = std::move(shaders[0]),
		.fragment_shader = std::move(shaders[1]),

		.no_bindings = true,
		.vertex_binding = {},
		.vertex_attributes = {},

		.pipeline_layout = _pipelines.postprocess_layout,
		.pipeline_cache = pc,

		.depth_test = true,
		.depth_write = true
	};

	_pipelines.postprocess = make_graphics_pipeline(grp_info);
}

// Initialize all pipelines
void Layer::_init_pipelines()
{
	// First, create the descriptor set layouts
	_dsl_raytracing = make_descriptor_set_layout(
		_device,
		_raytracing_bindings
	);

	_dsl_postprocess = make_descriptor_set_layout(
		_device,
		_postproc_bindings
	);

	// Then, create the descriptor sets
	auto dset1 = vk::raii::DescriptorSets {
		_device,
		{*_descriptor_pool, *_dsl_raytracing},
	};

	auto dset2 = vk::raii::DescriptorSets {
		_device,
		{*_descriptor_pool, *_dsl_postprocess},
	};

	_dset_raytracing = std::move(dset1.front());
	_dset_postprocess = std::move(dset2.front());

	// All pipelines
	_init_compute_pipelines();
	_init_postprocess_pipeline();
}

// Update descriptor sets for samplers
void Layer::_update_samplers(const ImageDescriptors &descriptors, uint32_t binding)
{
	// Update descriptor set
	vk::WriteDescriptorSet dset_write {
		*_dset_raytracing,
		binding, 0,
		vk::DescriptorType::eCombinedImageSampler,
		descriptors
	};

	_device.updateDescriptorSets(dset_write, nullptr);
}

// Get list of bboxes for each triangle
std::vector <BoundingBox> Layer::_get_bboxes() const
{
	std::vector <BoundingBox> bboxes;
	bboxes.reserve(_host.triangles.size());

	const auto &vertices = _host.vertices;
	const auto &triangles = _host.triangles;

	for (size_t i = 0; i < _host.triangles.size(); i++) {
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

////////////////////
// Public methods //
////////////////////

// Constructor
Layer::Layer(const vk::raii::PhysicalDevice &phdev,
		const vk::raii::Device &device,
		const vk::raii::CommandPool &command_pool,
		const vk::raii::DescriptorPool &descriptor_pool,
		const vk::Extent2D &extent,
		const vk::Format &swapchain_format,
		const vk::Format &depth_format,
		const vk::AttachmentLoadOp &load)
		: _physical_device(phdev), _device(device),
		_command_pool(command_pool),
		_descriptor_pool(descriptor_pool),
		_extent(extent)
{
	// Organize queues
	// TODO: helper function
	auto queue_families = _physical_device.getQueueFamilyProperties();
	auto graphics_familiy = find_graphics_queue_family(_physical_device);

	// TODO: singleton queue tracker to track which queues are in use

	KOBRA_LOG_FILE(notify) << "Queue families:\n";
	std::cout << dev_info(_physical_device) << std::endl;

	std::cout << "Scanning for compute queues:\n";
	std::cout << "\t# of queues: " << queue_families.size() << std::endl;
	int count = 0;
	for (uint32_t i = 0; i < queue_families.size(); i++) {
		const auto &qf = queue_families[i];
		std::cout << "\t" << qf.queueCount << " queues in family " << i << std::endl;
		if (qf.queueCount > 0 && (qf.queueFlags & vk::QueueFlagBits::eCompute)){
			count += qf.queueCount;

			// TODO: acquire queue method or something
			int j = 0;
			while (_queues.size() < PARALLELIZATION) {
				// TODO: dont hardcode..., check which queues
				// are already in use...
				if (i == graphics_familiy && j == 0) {
					j++;
					continue;
				}

				_queues.emplace_back(_device, i, j++);
				_command_pools.emplace_back(
					device, vk::CommandPoolCreateInfo {
						vk::CommandPoolCreateFlagBits::eResetCommandBuffer,
						i
					}
				);

				_command_buffers.emplace_back(
					make_command_buffer(
						_device,
						_command_pools.back()
					)
				);

				_fences.emplace_back(_device, vk::FenceCreateInfo {
					vk::FenceCreateFlagBits::eSignaled
				});
			}
		}
	}

	KOBRA_LOG_FILE(notify) << "Found " << count << " compute queues\n";

	// Create the render pass
	_render_pass = make_render_pass(_device,
		swapchain_format,
		depth_format, load
	);

	// Initialize pipelines
	_init_pipelines();

	// Allocate buffers
	vk::DeviceSize uint_size = 1024 * sizeof(uint);
	vk::DeviceSize vec4_size = 1024 * sizeof(aligned_vec4);
	vk::DeviceSize mat4_size = 1024 * sizeof(aligned_mat4);
	vk::DeviceSize pixels_size = _extent.width * _extent.height * sizeof(uint);

	auto usage = vk::BufferUsageFlagBits::eStorageBuffer;
	auto mem_props = vk::MemoryPropertyFlagBits::eDeviceLocal
		| vk::MemoryPropertyFlagBits::eHostCoherent
		| vk::MemoryPropertyFlagBits::eHostVisible;

	_dev.pixels = BufferData(
		_physical_device, _device, pixels_size,
		usage | vk::BufferUsageFlagBits::eTransferSrc, mem_props
	);

	_dev.vertices = BufferData(phdev, _device, vec4_size, usage, mem_props);
	_dev.triangles = BufferData(phdev, _device, vec4_size, usage, mem_props);
	_dev.materials = BufferData(phdev, _device, vec4_size, usage, mem_props);

	_dev.lights = BufferData(phdev, _device, vec4_size, usage, mem_props);
	_dev.light_indices = BufferData(phdev, _device, uint_size, usage, mem_props);

	_dev.transforms = BufferData(phdev, _device, mat4_size, usage, mem_props);

	_dev.bind(_device, _dset_raytracing);
	// Initial (blank) binding

	// Rebind to descriptor sets
	_bvh = BVH(_physical_device, _device, _get_bboxes());

	bind_ds(_device,
		_dset_raytracing, _bvh.buffer,
		vk::DescriptorType::eStorageBuffer,
		MESH_BINDING_BVH
	);

	// Bind to descriptor sets
	bind_ds(_device,
		_dset_raytracing, _dev.pixels,
		vk::DescriptorType::eStorageBuffer,
		MESH_BINDING_PIXELS
	);

	// Sampler source images
	_samplers.empty_image = ImageData::blank(_physical_device, _device);
	_samplers.environment_image = ImageData::blank(_physical_device, _device);

	// Initialize samplers
	_samplers.empty = make_sampler(_device, _samplers.empty_image);
	_samplers.environment = make_sampler(_device, _samplers.environment_image);

	// Output image and sampler
	_samplers.result_image = ImageData(
		_physical_device, _device,
		vk::Format::eR8G8B8A8Unorm, _extent,
		vk::ImageTiling::eOptimal,
		vk::ImageUsageFlagBits::eSampled
			| vk::ImageUsageFlagBits::eTransferDst,
		vk::ImageLayout::ePreinitialized,
		vk::MemoryPropertyFlagBits::eDeviceLocal,
		vk::ImageAspectFlagBits::eColor
	);

	_samplers.result = make_sampler(_device, _samplers.result_image);

	// Transition image layout
	auto tmp_cmd = make_command_buffer(_device, _command_pool);
	auto queue = vk::raii::Queue {_device, 0, 0};

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
	bind_ds(_device,
		_dset_raytracing,
		_samplers.environment,
		_samplers.environment_image,
		MESH_BINDING_ENVIRONMENT
	);

	bind_ds(_device,
		_dset_raytracing,
		_dev.pixels,
		vk::DescriptorType::eStorageBuffer,
		MESH_BINDING_PIXELS
	);

	bind_ds(_device,
		_dset_postprocess,
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

	// Initialized
	_initialized = true;
}

// Adding elements
void Layer::add_do(const ptr &e)
{
	// Make sure the resources were intialized
	if (!_initialized) {
		KOBRA_LOG_FUNC(warn) << "rt::Layer is not yet initialized\n";
		return;
	}

	LatchingPacket lp {
		.vertices = _host.vertices,
		.triangles = _host.triangles,
		.materials = _host.materials,

		.lights = _host.lights,
		.light_indices = _host.light_indices,

		.transforms = _host.transforms,

		.albedo_samplers = _albedo_image_descriptors,
		.normal_samplers = _normal_image_descriptors,
	};

	e->latch(lp, _elements.size());

	// Upload all data to device memory
	_dev.vertices.upload(_host.vertices);
	_dev.triangles.upload(_host.triangles);
	_dev.materials.upload(_host.materials);

	_dev.lights.upload(_host.lights);
	_dev.light_indices.upload(_host.light_indices);

	_dev.transforms.upload(_host.transforms);

	// Rebind to descriptor sets
	_dev.bind(_device, _dset_raytracing);

	// Update sampler descriptors
	_update_samplers(_albedo_image_descriptors, MESH_BINDING_ALBEDOS);
	_update_samplers(_normal_image_descriptors, MESH_BINDING_NORMAL_MAPS);

	// Update the BVH
	_bvh = BVH(_physical_device, _device, _get_bboxes());

	// Rebind to descriptor sets
	bind_ds(_device,
		_dset_raytracing, _bvh.buffer,
		vk::DescriptorType::eStorageBuffer,
		MESH_BINDING_BVH
	);
}

// Adding scenes
void Layer::add_scene(const Scene &scene)
{
	// Iterate through each object
	// and check if it is compatible
	// with this layer
	for (const auto &obj : scene) {
		std::string type = obj->type();

		if (type == kobra::Sphere::object_type) {
			kobra::Sphere *sphere = dynamic_cast
				<kobra::Sphere *> (obj.get());
			Sphere *nsphere = new Sphere(*sphere);
			kobra::Layer <_element> ::add(ptr(nsphere));
		}

		if (type == Sphere::object_type) {
			Sphere *sphere = dynamic_cast
				<Sphere *> (obj.get());
			kobra::Layer <_element> ::add(ptr(sphere));
		}

		if (type == kobra::Mesh::object_type) {
			kobra::Mesh *mesh = dynamic_cast
				<kobra::Mesh *> (obj.get());
			Mesh *nmesh = new Mesh(*mesh);
			kobra::Layer <_element> ::add(ptr(nmesh));
		}

		if (type == Mesh::object_type) {
			Mesh *mesh = dynamic_cast
				<Mesh *> (obj.get());
			kobra::Layer <_element> ::add(ptr(mesh));
		}
	}
}

// Set environment map
void Layer::set_environment_map(ImageData &&image)
{
	// Copy, create a new sampler, and rebind
	_samplers.environment_image = std::move(image);

	// Create sampler and bind
	_samplers.environment = make_sampler(_device, _samplers.environment_image);

	bind_ds(_device,
		_dset_raytracing,
		_samplers.environment,
		_samplers.environment_image,
		MESH_BINDING_ENVIRONMENT
	);
}

// Clearning all data
void Layer::clear()
{
	// Call parents clear
	kobra::Layer <_element> ::clear();

	// Clear all the buffers
	_host.vertices.clear();
	_host.triangles.clear();
	_host.materials.clear();
	_host.lights.clear();
	_host.light_indices.clear();
	_host.transforms.clear();
}

// Number of triangles
size_t Layer::triangle_count() const
{
	return _host.triangles.size();
}

// Number of cameras
size_t Layer::camera_count() const
{
	return _cameras.size();
}

// Add a camera to the layer
void Layer::add_camera(const Camera &camera)
{
	_cameras.push_back(camera);
}

// Active camera
Camera *Layer::active_camera()
{
	return _active_camera;
}

// Activate a camera
Camera *Layer::activate_camera(size_t index)
{
	if (index < _cameras.size()) {
		_active_camera = &_cameras[index];
	} else {
		KOBRA_LOG_FUNC(warn) << "Camera index out of range ["
			<< index << "/" << _cameras.size() << "]";
	}

	return _active_camera;
}

// Set active camera
void Layer::set_active_camera(const Camera &camera)
{
	// If active camera has not been set
	if (_active_camera == nullptr) {
		if (_cameras.empty())
			_cameras.push_back(camera);

		_active_camera = &_cameras[0];
	}

	*_active_camera = camera;
}

// Get pixel buffer
const BufferData &Layer::pixels()
{
	return _dev.pixels;
}

// Display memory footprint
void Layer::display_memory_footprint() const
{
	auto bvh = partition(_get_bboxes());
	Logger::notify() << "BVH: " << bvh->node_count() << " nodes, "
		<< bvh->primitive_count() << " primitives\n";

	/* Display memory footprint
	Logger::notify() << "Memory footprint of pixels = "
		<< _pixels.bytes()/float(1024 * 1024) << " MiB\n";
	Logger::notify() << "Memory footprint of bvh = "
		<< bvh->bytes()/float(1024 * 1024) << " MiB\n";
	Logger::notify() << "Memory footprint of vertices = "
		<< _vertices.bytes()/float(1024 * 1024) << " MiB\n";
	Logger::notify() << "Memory footprint of triangles = "
		<< _triangles.bytes()/float(1024 * 1024) << " MiB\n";
	Logger::notify() << "Memory footprint of materials = "
		<< _materials.bytes()/float(1024 * 1024) << " MiB\n";
	Logger::notify() << "Memory footprint of transforms = "
		<< _transforms.bytes()/float(1024 * 1024) << " MiB\n";
	Logger::notify() << "Memory footprint of lights = "
		<< _lights.bytes()/float(1024 * 1024) << " MiB\n";
	Logger::notify() << "Memory footprint of light indices = "
		<< _light_indices.bytes()/float(1024 * 1024) << " MiB\n";

	// For all the textures
	Logger::notify() << "Memory footprint for albedo textures:\n";

	size_t albedo_bytes = 0;
	for (auto &as : _albedo_image_descriptors) {
		auto sampler = as.sampler;
		Logger::plain() << "\tsampler=" << sampler;
		size_t bytes = Sampler::sampler_cache.at(sampler)->bytes();
		Logger::plain() << ", data="
			<< bytes/float(1024 * 1024) << " MiB\n";
		albedo_bytes += bytes;
	}
	Logger::notify() << "Memory footprint of ALLalbedo textures = "
		<< albedo_bytes/float(1024 * 1024) << " MiB\n";

	Logger::notify() << "Memory footprint for normal textures:\n";

	size_t normal_bytes = 0;
	for (auto &ns : _normal_image_descriptors) {
		auto sampler = ns.sampler;
		Logger::plain() << "\tsampler=" << sampler;
		size_t bytes = Sampler::sampler_cache.at(sampler)->bytes();
		Logger::plain() << ", data="
			<< bytes/float(1024 * 1024) << " MiB\n";
		normal_bytes += bytes;
	}

	Logger::notify() << "Memory footprint of ALLnormal textures = "
		<< normal_bytes/float(1024 * 1024) << " MiB\n";

	Logger::plain() << "\n"; */
}

// Laucnh RT batch kernel
void Layer::_launch_kernel(uint32_t index, const Batch &batch, const BatchIndex &bi)
{
	auto &computer = _command_buffers[index];

	computer.begin({});

	// Run the raytracing pipeline
	computer.bindPipeline(
		vk::PipelineBindPoint::eCompute,
		*_active_pipeline()
	);

	// Time as float
	unsigned int time = static_cast <unsigned int>
		(std::chrono::duration_cast
			<std::chrono::milliseconds>
			(std::chrono::system_clock::now().time_since_epoch()).count());

	// Prepare push constants
	PushConstants pc {
		.width = _extent.width,
		.height = _extent.height,

		.xoffset = bi.offset_x,
		.yoffset = bi.offset_y,

		.triangles = (uint) _host.triangles.size(),
		.lights = (uint) _host.light_indices.size(),

		// TODO: still unable to do large number of samples
		.samples_per_pixel = bi.pixel_samples,
		.samples_per_surface = bi.surface_samples,
		.samples_per_light = bi.light_samples,

		.accumulate = (bi.accumulate) ? 1u : 0u,
		.present = (uint32_t) batch.samples(bi),
		.total = (uint32_t) batch.total_samples(),

		// Pass current time (as float)
		.time = (float) time,

		.camera_position = _active_camera->transform.position,
		.camera_forward = _active_camera->transform.forward(),
		.camera_up = _active_camera->transform.up(),
		.camera_right = _active_camera->transform.right(),

		.camera_tunings = glm::vec4 {
			active_camera()->tunings.scale,
			active_camera()->tunings.aspect,
			0, 0
		}
	};

	// Bind descriptor set
	computer.bindDescriptorSets(
		vk::PipelineBindPoint::eCompute,
		*_pipelines.raytracing_layout,
		0, {*_dset_raytracing}, {}
	);

	// Push constants
	computer.pushConstants <PushConstants> (
		*_pipelines.raytracing_layout,
		vk::ShaderStageFlagBits::eCompute,
		0, pc
	);

	// Dispatch the compute shader
	computer.dispatch(bi.width, bi.height, 1);
	computer.end();

	// Submit the command buffer
	vk::SubmitInfo submit_info {
		0, nullptr, nullptr,
		1, &*computer
	};

	_queues[index].submit(submit_info, *_fences[index]);

	// Callback for batch index
	bi.callback();
}

// RT kernel function
void Layer::_kernel(uint32_t index)
{
	// Local (unique) command buffer
	auto &computer = _command_buffers[index];

	// Prepare push constants
	PushConstants pc {
		.width = _extent.width,
		.height = _extent.height,

		.triangles = (uint) _host.triangles.size(),
		.lights = (uint) _host.light_indices.size(),

		// TODO: still unable to do large number of samples
		// These are not expected to change
		.samples_per_pixel = _batch_index.pixel_samples,
		.samples_per_surface = _batch_index.surface_samples,
		.samples_per_light = _batch_index.light_samples,
		.accumulate = (_batch_index.accumulate) ? 1u : 0u,

		.camera_position = _active_camera->transform.position,
		.camera_forward = _active_camera->transform.forward(),
		.camera_up = _active_camera->transform.up(),
		.camera_right = _active_camera->transform.right(),

		.camera_tunings = glm::vec4 {
			active_camera()->tunings.scale,
			active_camera()->tunings.aspect,
			0, 0
		}
	};

	// Loop until termination
	while (_running) {
		// Acquire next batch, if any
		if (_batch->completed())
			continue;

		// Wait for the fence to be signaled
		while (vk::Result(_device.waitForFences(
			*_fences[index], true,
			std::numeric_limits <uint64_t>::max()
		)) != vk::Result::eSuccess);

		// Reset the fence
		_device.resetFences(*_fences[index]);

		// Time
		unsigned int time = static_cast <unsigned int>
			(std::chrono::duration_cast
				<std::chrono::milliseconds>
				(std::chrono::system_clock::now().time_since_epoch()).count());

		// Get the next batch index
		BatchIndex local_index;

		_mutex.lock();

		{
			_batch->increment(_batch_index);

			pc.xoffset = _batch_index.offset_x;
			pc.yoffset = _batch_index.offset_y;

			pc.present = (uint32_t) _batch->samples(_batch_index);

			pc.time = (float) time;

			local_index = _batch_index;
		}

		_mutex.unlock();

		// Kernel time!
		computer.begin({});

		// Run the raytracing pipeline
		computer.bindPipeline(
			vk::PipelineBindPoint::eCompute,
			*_active_pipeline()
		);

		// Bind descriptor set
		computer.bindDescriptorSets(
			vk::PipelineBindPoint::eCompute,
			*_pipelines.raytracing_layout,
			0, {*_dset_raytracing}, {}
		);

		// Push constants
		computer.pushConstants <PushConstants> (
			*_pipelines.raytracing_layout,
			vk::ShaderStageFlagBits::eCompute,
			0, pc
		);

		// Dispatch the compute shader
		computer.dispatch(local_index.width, local_index.height, 1);

		// End the command buffer
		computer.end();

		// Submit the command buffer
		vk::SubmitInfo submit_info {
			0, nullptr, nullptr,
			1, &*computer
		};

		_queues[index].submit(submit_info, *_fences[index]);

		// Callback for batch index
		_mutex.lock();
		local_index.callback();
		_mutex.unlock();
	}

	// Wait for the fence to be signaled
	while (vk::Result(_device.waitForFences(
		*_fences[index], true,
		std::numeric_limits <uint64_t>::max()
	)) != vk::Result::eSuccess);
}

// Launch RT kernels
void Layer::launch()
{
	if (_threads)
		return;

	// Set running flag
	_running = true;

	// Create threads
	_threads = new std::thread[PARALLELIZATION];
	for (uint32_t i = 0; i < PARALLELIZATION; i++)
		_threads[i] = std::thread(&Layer::_kernel, this, i);
}

// Stop RT kernels
void Layer::stop()
{
	_running = false;
	if (_threads) {
		for (int i = 0; i < PARALLELIZATION; i++)
			_threads[i].join();

		delete[] _threads;
	}
}

// Render a batch
void Layer::render(const vk::raii::CommandBuffer &cmd,
		const vk::raii::Framebuffer &framebuffer,
		const vk::Extent2D &extent,
		Batch &batch,
		BatchIndex &bi)
{
	// Handle null pipeline
	if (!_initialized) {
		KOBRA_LOG_FUNC(warn) << "rt::Layer is not yet initialized\n";
		return;
	}

	// Handle null active camera
	if (_active_camera == nullptr) {
		KOBRA_LOG_FUNC(warn) << "rt::Layer has no active camera\n";
		return;
	}

	// Set viewport
	auto viewport = vk::Viewport {
		0.0f, 0.0f,
		static_cast <float> (_extent.width),
		static_cast <float> (_extent.height),
		0.0f, 1.0f
	};

	cmd.setViewport(0, viewport);

	// Set scissor
	auto scissor = vk::Rect2D {
		vk::Offset2D {0, 0},
		_extent
	};

	cmd.setScissor(0, scissor);
	
	// Post process pipeline
	cmd.bindPipeline(
		vk::PipelineBindPoint::eGraphics,
		*_pipelines.postprocess
	);

	// Transition the result image to transfer destination
	_samplers.result_image.transition_layout(cmd, vk::ImageLayout::eTransferDstOptimal);

	copy_data_to_image(cmd,
		_dev.pixels.buffer,
		_samplers.result_image.image,
		_samplers.result_image.format,
		_extent.width,
		_extent.height
	);

	// Transition image back to shader read
	_samplers.result_image.transition_layout(cmd, vk::ImageLayout::eShaderReadOnlyOptimal);

	// Bind descriptor set
	cmd.bindDescriptorSets(
		vk::PipelineBindPoint::eGraphics,
		*_pipelines.postprocess_layout,
		0, {*_dset_postprocess}, {}
	);

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

	// Draw and end
	cmd.draw(6, 1, 0, 0);
	cmd.endRenderPass();
}

}

}

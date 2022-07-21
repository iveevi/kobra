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
	// Push constant structures
	// TODO: goes into source file
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

	// Vulkan context
	Context _ctx;

	// Other Vulkan structures
	vk::raii::RenderPass		_render_pass = nullptr;

	vk::raii::PipelineLayout	_ppl_raytracing = nullptr;
	vk::raii::PipelineLayout	_ppl_postprocess = nullptr;

	vk::raii::Pipeline		_p_raytracing = nullptr;
	vk::raii::Pipeline		_p_postprocess = nullptr;

	// Device buffer data
	struct {
		BufferData		pixels = nullptr;

		BufferData		bvh = nullptr;

		BufferData		vertices = nullptr;
		BufferData		triangles = nullptr;
		BufferData		materials = nullptr;

		BufferData		lights = nullptr;
		BufferData		light_indices = nullptr;

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

	// Descriptor set, layout and bindnigs
	vk::raii::DescriptorSet		_ds_raytracing = nullptr;
	vk::raii::DescriptorSet		_ds_postprocess = nullptr;

	vk::raii::DescriptorSetLayout	_dsl_raytracing = nullptr;
	vk::raii::DescriptorSetLayout	_dsl_postprocess = nullptr;

	static const std::vector <DSLB> _dslb_raytracing;
	static const std::vector <DSLB> _dslb_postprocess;

	// TODO: image2D instead of buffer

	// Helper functions
	void _initialize_vuklan_structures(const vk::AttachmentLoadOp &load) {
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

		GraphicsPipelineInfo grp_info {
			.device = *_ctx.device,
			.render_pass = _render_pass,

			.vertex_shader = std::move(shaders[1]),
			.fragment_shader = std::move(shaders[2]),

			.no_bindings = true,
			.vertex_binding = {},
			.vertex_attributes = {},

			.pipeline_layout = _ppl_postprocess,
			.pipeline_cache = vk::raii::PipelineCache {
				*_ctx.device,
				vk::PipelineCacheCreateInfo {}
			},

			.depth_test = true,
			.depth_write = true
		};

		_p_postprocess = make_graphics_pipeline(grp_info);
	}

	std::vector <BoundingBox> _get_bboxes(const kobra::Raytracer::HostBuffers &hb) const
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

	// Update descriptor sets for samplers
	void _update_samplers(const ImageDescriptors &descriptors, uint32_t binding)
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

	// Accumulation status
	int		_accumulated = 0;
	Transform	_ptransform;
	int		_skip = 2;	// Skip batch size (per dim)
	int		_offsetx = 0;
	int		_offsety = 0;
public:
	// Default constructor
	Raytracer() = default;

	// Constructors
	Raytracer(const Context &ctx, const vk::AttachmentLoadOp &load)
			: _ctx(ctx) {
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

		_dev.lights = BufferData(phdev, device, vec4_size, usage, mem_props);
		_dev.light_indices = BufferData(phdev, device, uint_size, usage, mem_props);

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

	// Render
	void render(const vk::raii::CommandBuffer &cmd,
			const vk::raii::Framebuffer &framebuffer,
			const ECS &ecs) {
		// Initialization phase
		Camera camera;
		bool found_camera = false;

		kobra::Raytracer::HostBuffers host_buffers {.id = 1};

		for (int i = 0; i < ecs.size(); i++) {
			// TODO: how to avoid constructing BVH every single
			// frame?
			// GPU construction?
			// NOTE: CudaRaytracer will rely on OptiX for this

			// Deal with camera component
			if (ecs.exists <Camera> (i)) {
				camera = ecs.get <Camera> (i);
				found_camera = true;
			}

			// Deal with raytracer component
			if (ecs.exists <kobra::Raytracer> (i)) {
				// TODO: raytracer component with these methods
				// (private)
				const kobra::Raytracer *raytracer = &ecs.get <kobra::Raytracer> (i);
				const Transform &transform = ecs.get <Transform> (i);

				raytracer->serialize(transform, host_buffers);
			}

			// Deal with light
			if (ecs.exists <Light> (i)) {
				const Light &light = ecs.get <Light> (i);
				const Transform &transform = ecs.get <Transform> (i);

				// TODO: account for light type
				// TODO: helper method to push this data
				Material mat {.Kd = {1, 1, 1}, .type = eEmissive};
				mat.serialize(host_buffers.materials);

				// New vertices (square 1x1 in center)
				glm::vec3 v1 {-0.5f, 0, -0.5f};
				glm::vec3 v2 {0.5f, 0, -0.5f};
				glm::vec3 v3 {0.5f, 0, 0.5f};
				glm::vec3 v4 {-0.5f, 0, 0.5f};

				v1 = transform.apply(v1);
				v2 = transform.apply(v2);
				v3 = transform.apply(v3);
				v4 = transform.apply(v4);

				// New indices
				uint offset = host_buffers.vertices.size()/VERTEX_STRIDE;

				// TODO: clean singleton raytracer component to
				// TODO: use more optimal layout for lights
				// (emissive mesh != area light)
				host_buffers.vertices.push_back(v1);
				host_buffers.vertices.push_back(v1);
				host_buffers.vertices.push_back(v1);
				host_buffers.vertices.push_back(v1);
				host_buffers.vertices.push_back(v1);

				host_buffers.vertices.push_back(v2);
				host_buffers.vertices.push_back(v2);
				host_buffers.vertices.push_back(v2);
				host_buffers.vertices.push_back(v2);
				host_buffers.vertices.push_back(v2);

				host_buffers.vertices.push_back(v3);
				host_buffers.vertices.push_back(v3);
				host_buffers.vertices.push_back(v3);
				host_buffers.vertices.push_back(v3);
				host_buffers.vertices.push_back(v3);

				host_buffers.vertices.push_back(v4);
				host_buffers.vertices.push_back(v4);
				host_buffers.vertices.push_back(v4);
				host_buffers.vertices.push_back(v4);
				host_buffers.vertices.push_back(v4);

				// TODO: store area light struct with indices
				// and other propreties
				uint ia = offset, ib = offset + 1, ic = offset + 2, id = host_buffers.id - 1;

				glm::vec4 header {LIGHT_TYPE_AREA, 0, 0, 0};

				// TODO: uvec4
				glm::vec4 h1 {
					*reinterpret_cast <float *> (&ia),
					*reinterpret_cast <float *> (&ib),
					*reinterpret_cast <float *> (&ic),
					*reinterpret_cast <float *> (&id)
				};

				ia = offset; ib = offset + 2; ic = offset + 3;

				glm::vec4 h2 {
					*reinterpret_cast <float *> (&ia),
					*reinterpret_cast <float *> (&ib),
					*reinterpret_cast <float *> (&ic),
					*reinterpret_cast <float *> (&id)
				};

				host_buffers.id++;

				// TODO: fixed offset for lights, no need for
				host_buffers.light_indices.push_back(host_buffers.lights.size());
				host_buffers.lights.push_back(header);
				host_buffers.lights.push_back(h1);
				host_buffers.triangles.push_back(h1);

				host_buffers.light_indices.push_back(host_buffers.lights.size());
				host_buffers.lights.push_back(header);
				host_buffers.lights.push_back(h2);
				host_buffers.triangles.push_back(h2);
			}
		}

		/* std::cout << "Triangles: " << host_buffers.triangles.size() << std::endl;
		std::cout << "Vertices: " << host_buffers.vertices.size()/VERTEX_STRIDE << std::endl;
		for (auto &triangle : host_buffers.triangles) {
			glm::uvec4 t = *reinterpret_cast <glm::uvec4 *> (&triangle);
			std::cout << "\t" << t.x << " " << t.y << " " << t.z << " " << t.w << std::endl;

			glm::vec3 v1 = host_buffers.vertices[VERTEX_STRIDE * t.x].data;
			glm::vec3 v2 = host_buffers.vertices[VERTEX_STRIDE * t.y].data;
			glm::vec3 v3 = host_buffers.vertices[VERTEX_STRIDE * t.z].data;

			std::cout << "\t\t" << glm::to_string(v1) << std::endl;
			std::cout << "\t\t" << glm::to_string(v2) << std::endl;
			std::cout << "\t\t" << glm::to_string(v3) << std::endl;
		}

		std::cout << "Lights: " << host_buffers.lights.size() << std::endl;
		std::cout << "Light indices: " << host_buffers.light_indices.size() << std::endl;
		for (int i = 0; i < host_buffers.light_indices.size(); i++) {
			std::cout << "\t" << host_buffers.light_indices[i] << std::endl;

			glm::uvec4 l = *reinterpret_cast <glm::uvec4 *> (&host_buffers.lights[host_buffers.light_indices[i]]);
			std::cout << "\t\t" << l.x << " " << l.y << " " << l.z << " " << l.w << std::endl;

			glm::uvec4 v = *reinterpret_cast <glm::uvec4 *> (&host_buffers.lights[host_buffers.light_indices[i] + 1].data);
			std::cout << "\t\t" << v.x << " " << v.y << " " << v.z << " " << v.w << std::endl;

			glm::vec3 v1 = host_buffers.vertices[VERTEX_STRIDE * v.x].data;
			glm::vec3 v2 = host_buffers.vertices[VERTEX_STRIDE * v.y].data;
			glm::vec3 v3 = host_buffers.vertices[VERTEX_STRIDE * v.z].data;

			std::cout << "\t\t\t" << glm::to_string(v1) << std::endl;
			std::cout << "\t\t\t" << glm::to_string(v2) << std::endl;
			std::cout << "\t\t\t" << glm::to_string(v3) << std::endl;
		} */

		if (!found_camera) {
			// Actually just skip
			throw std::runtime_error("No camera found");
		}

		// TODO: some way to combine bvhs of multiple objects (id
		// clashing...)
		auto bboxes = _get_bboxes(host_buffers);
		auto bvh = rt::partition(bboxes);

		rt::serialize(host_buffers.bvh, bvh);

		// Upload to device buffers
		_dev.bvh.upload(host_buffers.bvh, 0, false);
		_dev.vertices.upload(host_buffers.vertices, 0, false);
		_dev.triangles.upload(host_buffers.triangles, 0, false);
		_dev.materials.upload(host_buffers.materials, 0, false);
		_dev.lights.upload(host_buffers.lights, 0, false);
		_dev.light_indices.upload(host_buffers.light_indices, 0, false);

		_dev.transforms.upload(host_buffers.transforms, 0, false);

		// Compute shader
		cmd.bindPipeline(vk::PipelineBindPoint::eCompute, *_p_raytracing);

		// Time as float
		unsigned int time = static_cast <unsigned int>
			(std::chrono::duration_cast
				<std::chrono::milliseconds>
				(std::chrono::system_clock::now().time_since_epoch()).count());

		// Dirty means reset samples
		bool dirty = (_ptransform != camera.transform);
		if (dirty)
			_accumulated = 0;

		_ptransform = camera.transform;

		// TODO: using progressive rendering, we can skip pixels (every
		// other) and then increment samples after each complete pass.

		// Prepare push constants
		// TODO: method for generating this structure
		PushConstants pc {
			.width = _ctx.extent.width,
			.height = _ctx.extent.height,

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

			.camera_position = camera.transform.position,
			.camera_forward = camera.transform.forward(),
			.camera_up = camera.transform.up(),
			.camera_right = camera.transform.right(),

			.camera_tunings = glm::vec4 {
				camera.tunings.scale,
				camera.tunings.aspect,
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
			_ctx.extent.width,
			_ctx.extent.height
		);

		// Transition image back to shader read
		_samplers.result_image.transition_layout(cmd, vk::ImageLayout::eShaderReadOnlyOptimal);


		// Set viewport
		cmd.setViewport(0,
			vk::Viewport {
				0.0f, 0.0f,
				static_cast <float> (_ctx.extent.width),
				static_cast <float> (_ctx.extent.height),
				0.0f, 1.0f
			}
		);

		// Set scissor
		cmd.setScissor(0,
			vk::Rect2D {
				vk::Offset2D {0, 0},
				_ctx.extent
			}
		);

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
};

}

}

#endif

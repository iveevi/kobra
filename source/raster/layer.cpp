#include "../../include/raster/layer.hpp"
#include "../../include/sphere.hpp"
#include "../../shaders/raster/bindings.h"

namespace kobra {

namespace raster {

//////////////////////
// Static variables //
//////////////////////

const std::vector <DSLB> Layer::_full_dsl_bindings {
	DSLB {
		RASTER_BINDING_ALBEDO_MAP,
		vk::DescriptorType::eCombinedImageSampler,
		1, vk::ShaderStageFlagBits::eFragment
	},

	DSLB {
		RASTER_BINDING_NORMAL_MAP,
		vk::DescriptorType::eCombinedImageSampler,
		1, vk::ShaderStageFlagBits::eFragment
	},

	DSLB {
		RASTER_BINDING_POINT_LIGHTS,
		vk::DescriptorType::eUniformBuffer,
		1, vk::ShaderStageFlagBits::eFragment
	},
};

/////////////////////
// Private helpers //
/////////////////////

void Layer::_initialize_vulkan_structures
		(const vk::AttachmentLoadOp &load,
		const vk::Format &swapchain_format,
		const vk::Format &depth_format)
{
	// Create render pass
	_render_pass = make_render_pass(_device, swapchain_format, depth_format);

	// Create descriptor set layout
	_dsl_full = make_descriptor_set_layout(
		_device, _full_dsl_bindings,
		vk::DescriptorSetLayoutCreateFlagBits::eUpdateAfterBindPool
	);

	// Load necessary shader modules
	// TODO: create a map of names to shaders (for fragment, since
	// vertex is the same)
	auto shaders = make_shader_modules(_device, {
		"shaders/bin/raster/vertex.spv",
		"shaders/bin/raster/color_frag.spv",
		"shaders/bin/raster/normal_frag.spv",
		"shaders/bin/raster/blinn_phong_frag.spv"
	});

	// Push constants
	auto pcr = vk::PushConstantRange {
		vk::ShaderStageFlagBits::eVertex,
		0, sizeof(Mesh::MVP)
	};

	// Pipeline layout
	_pipelines.layout = vk::raii::PipelineLayout {
		_device,
		{{}, *_dsl_full, pcr}
	};

	// Pipeline cache
	auto pc = vk::raii::PipelineCache {
		_device,
		vk::PipelineCacheCreateInfo {}
	};

	// Vertex descriptions
	auto vertex_binding = Vertex::vertex_binding();
	auto vertex_attributes = Vertex::vertex_attributes();

	// Create graphics pipelines
	auto grp_info = GraphicsPipelineInfo {
		.device = _device,
		.render_pass = _render_pass,

		.vertex_shader = nullptr,
		.fragment_shader = nullptr,

		.vertex_binding = vertex_binding,
		.vertex_attributes = vertex_attributes,

		.pipeline_layout = _pipelines.layout,
		.pipeline_cache = pc,

		.depth_test = true,
		.depth_write = true
	};

	// Common vertex shader
	grp_info.vertex_shader = std::move(shaders[0]);

	// Albedo
	grp_info.fragment_shader = std::move(shaders[1]);
	_pipelines.albedo = make_graphics_pipeline(grp_info);

	// Normals
	grp_info.fragment_shader = std::move(shaders[2]);
	_pipelines.normals = make_graphics_pipeline(grp_info);

	// Blinn-Phong
	grp_info.fragment_shader = std::move(shaders[3]);
	_pipelines.blinn_phong = make_graphics_pipeline(grp_info);

	// Initialize buffers
	vk::DeviceSize buffer_size = 1024;

	_ubo_point_lights_buffer = BufferData(
		_physical_device, _device,
		buffer_size,
		vk::BufferUsageFlagBits::eUniformBuffer,
		vk::MemoryPropertyFlagBits::eHostVisible
			| vk::MemoryPropertyFlagBits::eHostCoherent
	);
}

////////////////////
// Public methods //
////////////////////

void Layer::add_scene(const Scene &scene)
{
	for (const ObjectPtr &obj : scene) {
		// TODO: later also add cameras
		if (obj->type() == raster::Mesh::object_type) {
			raster::Mesh *mesh = dynamic_cast
				<raster::Mesh *> (obj.get());
			add(mesh);
		}

		if (obj->type() == kobra::KMesh::object_type) {
			kobra::KMesh *mesh = dynamic_cast
				<kobra::KMesh *> (obj.get());

			raster::Mesh *raster_mesh = new raster::Mesh(
				_physical_device, _device, *mesh
			);

			add(raster_mesh);
		}

		if (obj->type() == kobra::Sphere::object_type) {
			kobra::Sphere *sphere = dynamic_cast
				<kobra::Sphere *> (obj.get());

			glm::vec3 pos = sphere->transform().position;
			std::cout << "Sphere: " << pos.x << ", " << pos.y << ", " << pos.z << std::endl;
			std::cout << "\tradius = " << sphere->radius() << std::endl;

			// Create a sphere mesh
			kobra::KMesh mesh = kobra::KMesh::make_sphere(
				// TODO: no center -- should always be 0,0,0
				{0, 0, 0}, sphere->radius()
			);

			raster::Mesh *raster_mesh = new raster::Mesh(
				_physical_device, _device, mesh
			);

			raster_mesh->transform() = sphere->transform();
			raster_mesh->set_material(sphere->material().copy());
			raster_mesh->set_name(sphere->name());
			add(raster_mesh);
		}
	}
}

}

}

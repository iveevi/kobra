#include "../../include/raster/layer.hpp"
#include "../../include/sphere.hpp"
#include "../../shaders/raster/bindings.h"

namespace kobra {

namespace raster {

//////////////////////
// Static variables //
//////////////////////

const Layer::DSLBindings Layer::_full_dsl_bindings {
	DSLBinding {
		.binding = RASTER_BINDING_ALBEDO_MAP,
		.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
		.descriptorCount = 1,
		.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT
	},

	DSLBinding {
		.binding = RASTER_BINDING_NORMAL_MAP,
		.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
		.descriptorCount = 1,
		.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT
	},

	DSLBinding {
		.binding = RASTER_BINDING_POINT_LIGHTS,
		.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
		.descriptorCount = 1,
		.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT
	},
};

/////////////////////
// Private helpers //
/////////////////////

void Layer::_initialize_vulkan_structures(const VkAttachmentLoadOp &load)
{
	// Create render pass
	_render_pass = _wctx.context.vk->make_render_pass(
		_wctx.context.phdev,
		_wctx.context.device,
		_wctx.swapchain,
		load,
		VK_ATTACHMENT_STORE_OP_STORE
	);

	// Create descriptor set and layout
	_full_dsl = _wctx.context.make_dsl(_full_dsl_bindings);
	/* _full_dsl = _wctx.context.vk->make_descriptor_set_layout(
		_wctx.context.device,
		_full_dsl_bindings,
		VK_DESCRIPTOR_SET_LAYOUT_CREATE_UPDATE_AFTER_BIND_POOL_BIT
	); */

	// Load necessary shader modules
	// TODO: create a map of names to shaders (for fragment, since
	// vertex is the same)
	std::vector <VkShaderModule> shaders = _wctx.context.make_shaders({
		"shaders/bin/raster/vertex.spv",
		"shaders/bin/raster/color_frag.spv",
		"shaders/bin/raster/normal_frag.spv",
		"shaders/bin/raster/blinn_phong_frag.spv"
	});

	// Push constants
	VkPushConstantRange pcr {
		.stageFlags = VK_SHADER_STAGE_VERTEX_BIT,
		.offset = 0,
		.size = sizeof(typename Mesh::MVP)
	};

	// Creation info
	Vulkan::PipelineInfo info {
		.swapchain = _wctx.swapchain,
		.render_pass = _render_pass,

		.dsls = {_full_dsl},

		.vertex_binding = Vertex::vertex_binding(),
		.vertex_attributes = Vertex::vertex_attributes(),

		.push_consts = 1,
		.push_consts_range = &pcr,

		.depth_test = true,

		.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,

		.viewport {
			.width = (int) _wctx.width,
			.height = (int) _wctx.height,
			.x = 0,
			.y = 0
		}
	};

	//////////////////////
	// Create pipelines //
	//////////////////////

	// Common vertex shader
	info.vert = shaders[0];

	// Albedo
	info.frag = shaders[1];
	_pipelines.albedo = _wctx.context.make_pipeline(info);

	// Normals
	info.frag = shaders[2];
	_pipelines.normals = _wctx.context.make_pipeline(info);

	// Blinn-Phong
	info.frag = shaders[3];
	_pipelines.blinn_phong = _wctx.context.make_pipeline(info);

	// Initialize buffers
	BFM_Settings write_settings {
		.size = 1024,
		.usage_type = BFM_WRITE_ONLY,
		.usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
		.descriptor_type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER
	};

	_ubo_point_lights_buffer = BufferManager
		<uint8_t> (_wctx.context, write_settings);
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

		if (obj->type() == kobra::Mesh::object_type) {
			kobra::Mesh *mesh = dynamic_cast
				<kobra::Mesh *> (obj.get());
			raster::Mesh *raster_mesh = new raster::Mesh(_wctx.context, *mesh);
			add(raster_mesh);
		}

		if (obj->type() == kobra::Sphere::object_type) {
			kobra::Sphere *sphere = dynamic_cast
				<kobra::Sphere *> (obj.get());

			glm::vec3 pos = sphere->transform().position;
			std::cout << "Sphere: " << pos.x << ", " << pos.y << ", " << pos.z << std::endl;

			// Create a sphere mesh
			kobra::Mesh mesh = kobra::Mesh::make_sphere(
				// TODO: no center -- should always be 0,0,0
				{0, 0, 0}, sphere->radius()
			);

			raster::Mesh *raster_mesh = new raster::Mesh(_wctx.context, mesh);
			raster_mesh->transform() = sphere->transform();
			raster_mesh->set_material(sphere->material());
			raster_mesh->set_name(sphere->name());
			add(raster_mesh);
		}
	}
}

}

}

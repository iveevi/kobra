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

const Layer::DSLBindings Layer::_mesh_compute_bindings {
	DSLBinding {
		.binding = MESH_BINDING_PIXELS,
		.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
		.descriptorCount = 1,
		.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
		.pImmutableSamplers = nullptr
	},

	DSLBinding {
		.binding = MESH_BINDING_VERTICES,
		.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
		.descriptorCount = 1,
		.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
		.pImmutableSamplers = nullptr
	},

	DSLBinding {
		.binding = MESH_BINDING_TRIANGLES,
		.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
		.descriptorCount = 1,
		.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
		.pImmutableSamplers = nullptr
	},

	DSLBinding {
		.binding = MESH_BINDING_TRANSFORMS,
		.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
		.descriptorCount = 1,
		.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
		.pImmutableSamplers = nullptr
	},

	DSLBinding {
		.binding = MESH_BINDING_BVH,
		.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
		.descriptorCount = 1,
		.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
		.pImmutableSamplers = nullptr
	},

	// Materials buffer
	DSLBinding {
		.binding = MESH_BINDING_MATERIALS,
		.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
		.descriptorCount = 1,
		.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
		.pImmutableSamplers = nullptr
	},

	// Lights buffer
	DSLBinding {
		.binding = MESH_BINDING_LIGHTS,
		.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
		.descriptorCount = 1,
		.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
		.pImmutableSamplers = nullptr
	},

	// Light indices
	DSLBinding {
		.binding = MESH_BINDING_LIGHT_INDICES,
		.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
		.descriptorCount = 1,
		.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
		.pImmutableSamplers = nullptr
	},

	// Texture samplers
	DSLBinding {
		.binding = MESH_BINDING_ALBEDOS,
		.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
		.descriptorCount = MAX_TEXTURES,
		.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
		.pImmutableSamplers = nullptr
	},

	DSLBinding {
		.binding = MESH_BINDING_NORMAL_MAPS,
		.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
		.descriptorCount = MAX_TEXTURES,
		.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
		.pImmutableSamplers = nullptr
	},

	DSLBinding {
		.binding = MESH_BINDING_ENVIRONMENT,
		.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
		.descriptorCount = 1,
		.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
		.pImmutableSamplers = nullptr
	},
	
	DSLBinding {
		.binding = MESH_BINDING_OUTPUT,
		.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
		.descriptorCount = 1,
		.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
		.pImmutableSamplers = nullptr
	},
};

const Layer::DSLBindings Layer::_postproc_bindings = {
	VkDescriptorSetLayoutBinding {
		.binding = 0,
		.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
		.descriptorCount = 1,
		.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT,
		.pImmutableSamplers = nullptr
	},

	/* VkDescriptorSetLayoutBinding {
		.binding = 1,
		.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
		.descriptorCount = 1,
		.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT,
		.pImmutableSamplers = nullptr
	}, */
};

//////////////////////////////
// Private helper functions //
//////////////////////////////

void Layer::_init_compute_pipelines()
{
	// Push constants
	VkPushConstantRange push_constants = {
		.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
		.offset = 0,
		.size = sizeof(PushConstants)
	};

	// Common pipeline layout
	VkPipelineLayoutCreateInfo compute_ppl_info = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
		.setLayoutCount = 1,
		.pSetLayouts = &_mesh_dsl,
		.pushConstantRangeCount = 1,
		.pPushConstantRanges = &push_constants
	};

	VkPipelineLayout compute_ppl;

	VkResult result;
	result = vkCreatePipelineLayout(
		_context.vk_device(),
		&compute_ppl_info,
		nullptr,
		&compute_ppl
	);

	KOBRA_ASSERT(
		result == VK_SUCCESS,
		"Failed to create pipeline layout for common compute pipeline"
	);

	VkPipeline mesh_pp;

	// Get all the shaders
	auto shaders = _context.make_shaders({
		"shaders/bin/generic/normal.spv",
		"shaders/bin/generic/heatmap.spv",
		"shaders/bin/generic/fast_path_tracer.spv",
		"shaders/bin/generic/pbr_path_tracer.spv",
		"shaders/bin/generic/mis_path_tracer.spv",
		"shaders/bin/generic/bidirectional_path_tracer.spv"
	});

	// Shader stage
	VkPipelineShaderStageCreateInfo shader_stages[] = {
		{	// Normals
			.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
			.stage = VK_SHADER_STAGE_COMPUTE_BIT,
			.module = shaders[0],
			.pName = "main"
		},
		{
			// Heatmap
			.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
			.stage = VK_SHADER_STAGE_COMPUTE_BIT,
			.module = shaders[1],
			.pName = "main"
		},
		{	// Fast path tracer
			.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
			.stage = VK_SHADER_STAGE_COMPUTE_BIT,
			.module = shaders[2],
			.pName = "main"
		},
		{	// PBR path tracer
			.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
			.stage = VK_SHADER_STAGE_COMPUTE_BIT,
			.module = shaders[3],
			.pName = "main"
		},
		{
			// MIS path tracer
			.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
			.stage = VK_SHADER_STAGE_COMPUTE_BIT,
			.module = shaders[4],
			.pName = "main"
		},
		{
			// Bidirectional path tracer
			.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
			.stage = VK_SHADER_STAGE_COMPUTE_BIT,
			.module = shaders[5],
			.pName = "main"
		}
	};

	VkComputePipelineCreateInfo ppl_info = {
		.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
		.layout = compute_ppl,
		.basePipelineHandle = VK_NULL_HANDLE,
		.basePipelineIndex = -1
	};

	// Lambda to create the pipeline
	auto ppl_maker = [&](const VkPipelineShaderStageCreateInfo &shader_stage) {
		// Set the shader stage
		ppl_info.stage = shader_stage;

		// Pipeline to return
		Vulkan::Pipeline pipeline;

		VkResult result = vkCreateComputePipelines(
			_context.vk_device(),
			VK_NULL_HANDLE,
			1,
			&ppl_info,
			nullptr,
			&pipeline.pipeline
		);

		KOBRA_ASSERT(
			result == VK_SUCCESS,
			"Failed to create compute pipeline"
		);

		// Transfer the layout and return
		pipeline.layout = compute_ppl;
		return pipeline;
	};

	// Create the pipelines
	_pipelines.normals = ppl_maker(shader_stages[0]);
	_pipelines.heatmap = ppl_maker(shader_stages[1]);
	_pipelines.fast_path_tracer = ppl_maker(shader_stages[2]);
	_pipelines.path_tracer = ppl_maker(shader_stages[3]);
	_pipelines.mis_path_tracer = ppl_maker(shader_stages[4]);
	_pipelines.bidirectional_path_tracer = ppl_maker(shader_stages[5]);
}

void Layer::_init_postproc_pipeline(const Vulkan::Swapchain &swapchain)
{
	// Load the shaders
	auto shaders = _context.make_shaders({
		"shaders/bin/generic/postproc_vert.spv",
		"shaders/bin/generic/postproc_frag.spv"
	});

	// Pipeline stages
	VkPipelineShaderStageCreateInfo vertex {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
		.stage = VK_SHADER_STAGE_VERTEX_BIT,
		.module = shaders[0],
		.pName = "main"
	};

	VkPipelineShaderStageCreateInfo fragment {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
		.stage = VK_SHADER_STAGE_FRAGMENT_BIT,
		.module = shaders[1],
		.pName = "main"
	};

	VkPipelineShaderStageCreateInfo shader_stages[] = {vertex, fragment};

	// Vertex input
	VkPipelineVertexInputStateCreateInfo vertex_input_info {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
		.vertexBindingDescriptionCount = 0,
		.pVertexBindingDescriptions = nullptr,
		.vertexAttributeDescriptionCount = 0,
		.pVertexAttributeDescriptions = nullptr
	};

	// Input assembly
	VkPipelineInputAssemblyStateCreateInfo input_assembly {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
		.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
		.primitiveRestartEnable = VK_FALSE
	};

	// TODO: swapchain function (to generate viewport and scissor, and
	// state)
	// Viewport
	VkViewport viewport {
		.x = 0.0f,
		.y = 0.0f,
		.width = (float) swapchain.extent.width,
		.height = (float) swapchain.extent.height,
		.minDepth = 0.0f,
		.maxDepth = 1.0f
	};

	// Scissor
	VkRect2D scissor {
		.offset = {0, 0},
		.extent = swapchain.extent
	};

	// Viewport state
	VkPipelineViewportStateCreateInfo viewport_state {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
		.viewportCount = 1,
		.pViewports = &viewport,
		.scissorCount = 1,
		.pScissors = &scissor
	};

	// Rasterizer
	VkPipelineRasterizationStateCreateInfo rasterizer {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
		.depthClampEnable = VK_FALSE,
		.rasterizerDiscardEnable = VK_FALSE,
		.polygonMode = VK_POLYGON_MODE_FILL,
		.cullMode = VK_CULL_MODE_NONE,
		.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE,
		.depthBiasEnable = VK_FALSE,
		.depthBiasConstantFactor = 0.0f,
		.depthBiasClamp = 0.0f,
		.depthBiasSlopeFactor = 0.0f,
		.lineWidth = 1.0f,
	};

	// Multisampling
	VkPipelineMultisampleStateCreateInfo multisampling {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
		.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT,
		.sampleShadingEnable = VK_FALSE,
		.minSampleShading = 1.0f,
		.pSampleMask = nullptr,
		.alphaToCoverageEnable = VK_FALSE,
		.alphaToOneEnable = VK_FALSE,
	};

	// Depth stencil
	VkPipelineDepthStencilStateCreateInfo depth_stencil {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
		.depthTestEnable = VK_TRUE,
		.depthWriteEnable = VK_TRUE,
		.depthCompareOp = VK_COMPARE_OP_LESS,
		.depthBoundsTestEnable = VK_FALSE,
		.stencilTestEnable = VK_FALSE,
		.front = {},
		.back = {}
	};

	// Color blending
	VkPipelineColorBlendAttachmentState color_blend_attachment {
		.blendEnable = VK_FALSE,
		.srcColorBlendFactor = VK_BLEND_FACTOR_ONE,
		.dstColorBlendFactor = VK_BLEND_FACTOR_ZERO,
		.colorBlendOp = VK_BLEND_OP_ADD,
		.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE,
		.dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO,
		.alphaBlendOp = VK_BLEND_OP_ADD,
		.colorWriteMask = VK_COLOR_COMPONENT_R_BIT
			| VK_COLOR_COMPONENT_G_BIT
			| VK_COLOR_COMPONENT_B_BIT
			| VK_COLOR_COMPONENT_A_BIT
	};

	// Color blending
	VkPipelineColorBlendStateCreateInfo color_blending {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
		.logicOpEnable = VK_FALSE,
		.logicOp = VK_LOGIC_OP_COPY,
		.attachmentCount = 1,
		.pAttachments = &color_blend_attachment,
		.blendConstants = {0.0f, 0.0f, 0.0f, 0.0f}
	};

	// Pipeline layout
	VkPushConstantRange push_constant_range {
		.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT,
		.offset = 0,
		.size = sizeof(PC_Viewport)
	};

	VkPipelineLayoutCreateInfo pipeline_layout_info {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
		.setLayoutCount = 1,
		.pSetLayouts = &_postproc_dsl,
		.pushConstantRangeCount = 1,
		.pPushConstantRanges = &push_constant_range
	};

	VkResult result;

	VkPipelineLayout pipeline_layout;
	result = vkCreatePipelineLayout(
		_context.vk_device(),
		&pipeline_layout_info,
		nullptr,
		&pipeline_layout
	);

	if (result != VK_SUCCESS) {
		KOBRA_LOG_FUNC(error) << "Failed to create POSTPROC pipeline layout\n";
		return;
	}

	// Pipeline
	VkGraphicsPipelineCreateInfo pipeline_info {
		.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
		.stageCount = 2,
		.pStages = shader_stages,
		.pVertexInputState = &vertex_input_info,
		.pInputAssemblyState = &input_assembly,
		.pViewportState = &viewport_state,
		.pRasterizationState = &rasterizer,
		.pMultisampleState = &multisampling,
		.pDepthStencilState = &depth_stencil,
		.pColorBlendState = &color_blending,
		.pDynamicState = nullptr,
		.layout = pipeline_layout,
		.renderPass = _render_pass,
		.subpass = 0,
		.basePipelineHandle = VK_NULL_HANDLE,
		.basePipelineIndex = -1
	};

	VkPipeline pipeline;
	result = vkCreateGraphicsPipelines(
		_context.vk_device(),
		VK_NULL_HANDLE,
		1,
		&pipeline_info,
		nullptr,
		&pipeline
	);

	if (result != VK_SUCCESS) {
		KOBRA_LOG_FUNC(error) << "Failed to create POSTPROC pipeline\n";
		return;
	}

	// Set pipeline
	_pipelines.postproc = {
		.pipeline = pipeline,
		.layout = pipeline_layout
	};
}

// Initialize all pipelines
void Layer::_init_pipelines(const Vulkan::Swapchain &swapchain)
{
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
	_init_compute_pipelines();
	_init_postproc_pipeline(swapchain);
}

// Update descriptor sets for samplers
void Layer::_update_samplers(const ImageDescriptors &ids, uint binding)
{
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

// Get list of bboxes for each triangle
std::vector <BoundingBox> Layer::_get_bboxes() const
{
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
			glm::vec4 center = vertices[VERTEX_STRIDE * a].data;
			float radius = center.w;

			glm::vec4 min = center - glm::vec4(radius);
			glm::vec4 max = center + glm::vec4(radius);

			bboxes.push_back(BoundingBox {min, max});
		} else {
			glm::vec4 va = vertices[VERTEX_STRIDE * a].data;
			glm::vec4 vb = vertices[VERTEX_STRIDE * b].data;
			glm::vec4 vc = vertices[VERTEX_STRIDE * c].data;

			glm::vec4 min = glm::min(va, glm::min(vb, vc));
			glm::vec4 max = glm::max(va, glm::max(vb, vc));

			bboxes.push_back(BoundingBox {min, max});
		}
	}

	return bboxes;
}

////////////////////
// Public methods //
////////////////////

// Constructor
Layer::Layer(const App::Window &wctx)
		: _context(wctx.context),
		_extent(wctx.swapchain.extent),
		_descriptor_pool(wctx.descriptor_pool),
		_command_pool(wctx.command_pool)
{
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
		.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
			| VK_BUFFER_USAGE_TRANSFER_SRC_BIT
	};
	
	// Debug output settings
	BFM_Settings debug_settings {
		.size = pixels,
		.usage_type = BFM_READ_WRITE,
		.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
			| VK_BUFFER_USAGE_TRANSFER_SRC_BIT
	};

	// TODO: remove this
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

	/////////////////////////////////////////////
	// Fill sampler arrays with blank samplers //
	/////////////////////////////////////////////

	// Initialize samplers
	_empty_sampler = Sampler::blank_sampler(_context, _command_pool);
	_env_sampler = Sampler::blank_sampler(_context, _command_pool);

	// Output image and sampler
	auto texture = Texture {
		.width = _extent.width,
		.height = _extent.height,
		.channels = 4,
	};

	_final_texture = make_texture(_context, _command_pool,
		texture,
		VK_FORMAT_R8G8B8A8_UNORM,
		VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
		VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
	);

	_final_texture.transition_manual(_context, _command_pool,
		VK_IMAGE_LAYOUT_UNDEFINED,
		VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
		VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
		VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT
	);

	_final_sampler = Sampler(_context, _final_texture);
	_final_sampler.bind(_postproc_ds, MESH_BINDING_PIXELS);

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
void Layer::add_do(const ptr &e)
{
	if (_pipelines.path_tracer.pipeline == VK_NULL_HANDLE) {
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

	auto bvh = partition(_get_bboxes());
	Logger::notify() << "BVH: " << bvh->node_count() << " nodes, "
		<< bvh->primitive_count() << " primitives, "
		<< bvh->bytes()/float(1024 * 1024) << " MB\n";

	// Rebind to descriptor sets
	_bvh.bind(_mesh_ds, MESH_BINDING_BVH);
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
void Layer::set_environment_map(const Texture &env)
{
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
void Layer::clear()
{
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
size_t Layer::triangle_count() const
{
	return _triangles.push_size();
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
const BufferManager <uint> &Layer::pixels()
{
	return _pixels;
}

// Render a batch
void Layer::render(const VkCommandBuffer &cmd,
		const VkFramebuffer &framebuffer,
		const Batch &batch,
		const BatchIndex &bi)
{
	// Handle null pipeline
	if (_active_pipeline()->pipeline == VK_NULL_HANDLE) {
		KOBRA_LOG_FUNC(warn) << "rt::Layer is not yet initialized\n";
		return;
	}

	// Handle null active camera
	if (_active_camera == nullptr) {
		KOBRA_LOG_FUNC(warn) << "rt::Layer has no active camera\n";
		return;
	}

	///////////////////////////
	// Mesh compute pipeline //
	///////////////////////////

	// TODO: context method
	vkCmdBindPipeline(cmd,
		VK_PIPELINE_BIND_POINT_COMPUTE,
		_active_pipeline()->pipeline
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

		.triangles = (uint) _triangles.push_size(),
		.lights = (uint) _light_indices.push_size(),

		// TODO: still unable to do large number of samples
		.samples_per_pixel = bi.pixel_samples,
		.samples_per_surface = bi.surface_samples,
		.samples_per_light = bi.light_samples,

		.accumulate = (bi.accumulate) ? 1u : 0u,
		.present = (uint) batch.samples(bi),
		.total = (uint) batch.total_samples(),

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
	vkCmdBindDescriptorSets(cmd,
		VK_PIPELINE_BIND_POINT_COMPUTE,
		_active_pipeline()->layout,
		0, 1, &_mesh_ds,
		0, nullptr
	);

	// Push constants
	vkCmdPushConstants(cmd,
		_active_pipeline()->layout,
		VK_SHADER_STAGE_COMPUTE_BIT,
		0, sizeof(PushConstants), &pc
	);

	// Dispatch the compute shader
	vkCmdDispatch(cmd,
		bi.width,
		bi.height,
		1
	);

	// Buffer memory barrier
	VkBufferMemoryBarrier buffer_barrier {
		.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
		.pNext = nullptr,
		.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
		.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT,
		.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
		.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
		.buffer = _pixels.vk_buffer(),
		.offset = 0,
		.size = VK_WHOLE_SIZE
	};

	// Wait for the compute shader to finish
	vkCmdPipelineBarrier(cmd,
		VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
		VK_PIPELINE_STAGE_TRANSFER_BIT,
		0,
		0, nullptr,
		1, &buffer_barrier,
		0, nullptr
	);

	// KOBRA_LOG_FILE(notify) << "rt::Layer::render()\n";

	//////////////////////////////
	// Post-processing pipeline //
	//////////////////////////////

	vkCmdBindPipeline(cmd,
		VK_PIPELINE_BIND_POINT_GRAPHICS,
		_pipelines.postproc.pipeline
	);

	/////////////////////////////////
	// Copy buffer to output image //
	/////////////////////////////////

	VkBufferImageCopy region {
		.bufferOffset = 0,
		.bufferRowLength = 0,
		.bufferImageHeight = 0,

		.imageSubresource = {
			.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
			.mipLevel = 0,
			.baseArrayLayer = 0,
			.layerCount = 1
		},

		.imageOffset = { 0, 0, 0 },
		.imageExtent = {
			.width = _final_texture.width,
			.height = _final_texture.height,
			.depth = 1
		}
	};

	_final_texture.transition_manual(cmd,
		VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
		VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
		VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
		VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT
	);

	vkCmdCopyBufferToImage(cmd,
		_pixels.vk_buffer(),
		_final_texture.image,
		VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
		1, &region
	);

	_final_texture.transition_manual(cmd,
		VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
		VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
		VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
		VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT
	);

	// Image memory barrier
	VkImageMemoryBarrier image_barrier {
		.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
		.pNext = nullptr,
		.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
		.dstAccessMask = VK_ACCESS_SHADER_READ_BIT,
		.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
		.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
		.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
		.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
		.image = _final_texture.image,
		.subresourceRange = {
			.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
			.baseMipLevel = 0,
			.levelCount = 1,
			.baseArrayLayer = 0,
			.layerCount = 1
		}
	};

	// Wait for the copy to finish

	// Bind descriptor set
	vkCmdBindDescriptorSets(cmd,
		VK_PIPELINE_BIND_POINT_GRAPHICS,
		_pipelines.postproc.layout,
		0, 1, &_postproc_ds,
		0, nullptr
	);

	// Clear colors
	VkClearValue clear_values[2] = {
		{ .color = { 0.0f, 0.0f, 0.0f, 1.0f } },
		{ .depthStencil = { 1.0f, 0 } }
	};

	// Begin render pass
	// TODO: context method
	VkRenderPassBeginInfo rp_info {
		.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
		.renderPass = _render_pass,
		.framebuffer = framebuffer,
		.renderArea = {
			.offset = {0, 0},
			.extent = _extent
		},
		.clearValueCount = 2,
		.pClearValues = clear_values
	};

	vkCmdBeginRenderPass(cmd, &rp_info, VK_SUBPASS_CONTENTS_INLINE);
	vkCmdDraw(cmd, 6, 1, 0, 0);
	vkCmdEndRenderPass(cmd);

	// Callback for batch index
	bi.callback();
}

}

}

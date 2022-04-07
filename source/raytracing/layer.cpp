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
};

const Layer::DSLBindings Layer::_postproc_bindings = {
	VkDescriptorSetLayoutBinding {
		.binding = 0,
		.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
		.descriptorCount = 1,
		.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT,
		.pImmutableSamplers = nullptr
	},

	VkDescriptorSetLayoutBinding {
		.binding = 1,
		.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
		.descriptorCount = 1,
		.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT,
		.pImmutableSamplers = nullptr
	},
};

//////////////////////////////
// Private helper functions //
//////////////////////////////

void Layer::_init_mesh_compute_pipeline()
{
	// Push constants
	VkPushConstantRange push_constants = {
		.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
		.offset = 0,
		.size = sizeof(PushConstants)
	};

	// TODO: context method to create layout
	VkPipelineLayoutCreateInfo mesh_ppl_ci = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
		.setLayoutCount = 1,
		.pSetLayouts = &_mesh_dsl,
		.pushConstantRangeCount = 1,
		.pPushConstantRanges = &push_constants
	};

	VkPipelineLayout mesh_ppl;

	VkResult result;
	result = vkCreatePipelineLayout(
		_context.vk_device(),
		&mesh_ppl_ci,
		nullptr,
		&mesh_ppl
	);

	if (result != VK_SUCCESS) {
		KOBRA_LOG_FUNC(warn) << "Failed to create MESH pipeline layout\n";
		return;
	}

	VkPipeline mesh_pp;

	VkShaderModule mesh_shader_module = _context.make_shader(
		"shaders/bin/generic/mesh.spv"
	);

	VkComputePipelineCreateInfo mesh_pp_ci = {
		.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
		.stage = {
			.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
			.stage = VK_SHADER_STAGE_COMPUTE_BIT,
			.module = mesh_shader_module,
			.pName = "main"
		},
		.layout = mesh_ppl,
		.basePipelineHandle = VK_NULL_HANDLE,
		.basePipelineIndex = -1
	};

	result = vkCreateComputePipelines(
		_context.vk_device(),
		VK_NULL_HANDLE,
		1,
		&mesh_pp_ci,
		nullptr,
		&mesh_pp
	);

	if (result != VK_SUCCESS) {
		KOBRA_LOG_FUNC(warn) << "Failed to create MESH pipeline\n";
		return;
	}

	// Set the pipeline
	_pipelines.mesh = {
		.pipeline = mesh_pp,
		.layout = mesh_ppl
	};
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

////////////////////
// Public methods //
////////////////////

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

}

}

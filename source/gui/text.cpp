#include "../../include/gui/text.hpp"

namespace mercury {

namespace gui {

////////////////
// Text class //
////////////////

void Text::refresh()
{
	_origin->text(this);
	_origin->add(this);
}

//////////////////////
// TextRender class //
//////////////////////

// Create the pipeline
void TextRender::_make_pipeline(const Bootstrap &bs)
{
	auto context = bs.ctx;
	auto swapchain = bs.swapchain;

	// Create pipeline stages
	VkPipelineShaderStageCreateInfo vertex {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
		.stage = VK_SHADER_STAGE_VERTEX_BIT,
		.module = _vertex,
		.pName = "main"
	};

	VkPipelineShaderStageCreateInfo fragment {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
		.stage = VK_SHADER_STAGE_FRAGMENT_BIT,
		.module = _fragment,
		.pName = "main"
	};

	VkPipelineShaderStageCreateInfo shader_stages[] = { vertex, fragment };

	// Vertex input
	auto binding_description = gui::Glyph::Vertex::vertex_binding();
	auto attribute_descriptions = gui::Glyph::Vertex::vertex_attributes();

	VkPipelineVertexInputStateCreateInfo vertex_input {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
		.vertexBindingDescriptionCount = 1,
		.pVertexBindingDescriptions = &binding_description,
		.vertexAttributeDescriptionCount = static_cast <uint32_t> (attribute_descriptions.size()),
		.pVertexAttributeDescriptions = attribute_descriptions.data()
	};

	// Input assembly
	VkPipelineInputAssemblyStateCreateInfo input_assembly {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
		.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
		.primitiveRestartEnable = VK_FALSE
	};

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

	VkPipelineViewportStateCreateInfo viewport_state {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
		.viewportCount = 1,
		.pViewports = &viewport,
		.scissorCount = 1,
		.pScissors = &scissor
	};

	// Rasterizer
	// TODO: method
	VkPipelineRasterizationStateCreateInfo rasterizer {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
		.depthClampEnable = VK_FALSE,
		.rasterizerDiscardEnable = VK_FALSE,
		.polygonMode = VK_POLYGON_MODE_FILL,
		.cullMode = VK_CULL_MODE_BACK_BIT,
		.frontFace = VK_FRONT_FACE_CLOCKWISE,
		.depthBiasEnable = VK_FALSE,
		.lineWidth = 1.0f
	};

	// Multisampling
	// TODO: method
	VkPipelineMultisampleStateCreateInfo multisampling {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
		.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT,
		.sampleShadingEnable = VK_FALSE,
		.minSampleShading = 1.0f,
		.pSampleMask = nullptr,
		.alphaToCoverageEnable = VK_FALSE,
		.alphaToOneEnable = VK_FALSE
	};

	// Color blending
	VkPipelineColorBlendAttachmentState color_blend_attachment {
		.blendEnable = VK_TRUE,
		.srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA,
		.dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA,
		.colorBlendOp = VK_BLEND_OP_ADD,
		.srcAlphaBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA,
		.dstAlphaBlendFactor = VK_BLEND_FACTOR_DST_ALPHA,
		.alphaBlendOp = VK_BLEND_OP_MAX,
		.colorWriteMask = VK_COLOR_COMPONENT_R_BIT
			| VK_COLOR_COMPONENT_G_BIT
			| VK_COLOR_COMPONENT_B_BIT
			| VK_COLOR_COMPONENT_A_BIT
	};

	VkPipelineColorBlendStateCreateInfo color_blending {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
		.logicOpEnable = VK_FALSE,
		.logicOp = VK_LOGIC_OP_COPY,
		.attachmentCount = 1,
		.pAttachments = &color_blend_attachment,
		.blendConstants = {0.0f, 0.0f, 0.0f, 0.0f}
	};

	// Pipeline layout
	VkPipelineLayoutCreateInfo pipeline_layout_info {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
		.setLayoutCount = 1,
		.pSetLayouts = &_layout,
		.pushConstantRangeCount = 0
	};

	VkPipelineLayout pipeline_layout;
	VkResult result = vkCreatePipelineLayout(
		context.vk_device(),
		&pipeline_layout_info,
		nullptr,
		&pipeline_layout
	);

	if (result != VK_SUCCESS) {
		throw std::runtime_error("failed to create pipeline layout!");
	}

	// Graphics pipeline
	VkGraphicsPipelineCreateInfo pipeline_info {
		.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
		.stageCount = 2,
		.pStages = shader_stages,
		.pVertexInputState = &vertex_input,
		.pInputAssemblyState = &input_assembly,
		.pViewportState = &viewport_state,
		.pRasterizationState = &rasterizer,
		.pMultisampleState = &multisampling,
		.pDepthStencilState = nullptr,
		.pColorBlendState = &color_blending,
		.pDynamicState = nullptr,
		.layout = pipeline_layout,
		.renderPass = bs.renderpass,
		.subpass = 0,
		.basePipelineHandle = VK_NULL_HANDLE,
		.basePipelineIndex = -1
	};

	VkPipeline pipeline;
	result = vkCreateGraphicsPipelines(
		context.vk_device(),
		VK_NULL_HANDLE,
		1,
		&pipeline_info,
		nullptr,
		&pipeline
	);

	if (result != VK_SUCCESS) {
		throw std::runtime_error("failed to create graphics pipeline!");
	}

	Logger::ok("[TextRender] Pipeline created");

	// Assign pipeline info
	_pipeline = pipeline;
	_pipeline_layout = pipeline_layout;
}

}

}

#ifndef LAYER_H_
#define LAYER_H_

// Standard headers
#include <map>
#include <vector>

// Engine headers
#include "../app.hpp"
#include "gui.hpp"
#include "text.hpp"

namespace kobra {

namespace gui {

// Contains a set of
//	GUI elements to render
// TODO: derive from element?
class Layer {
	// All elements to render
	std::vector <Element>		_elements;

	// Set of Text Render objects
	// for each font
	std::vector <TextRender>	_text_renders;

	// Map of font names/aliases to
	//	their respective TextRender indices
	std::map <std::string, int>	_font_map;

	// Application context
	App::Window			_wctx;

	// Vulkan structures
	VkRenderPass			_render_pass;

	// TODO: put in backend class
	struct Pipeline {
		VkPipeline pipeline;
		VkPipelineLayout layout;
	};

	Pipeline			_grp_shapes;

	// Allocation methods
	void _init_vulkan_structures(VkAttachmentLoadOp load) {
		// Create render pass
		// 	load previous contents
		_render_pass = _wctx.context.vk->make_render_pass(
			_wctx.context.device,
			_wctx.swapchain,
			load,
			VK_ATTACHMENT_STORE_OP_STORE
		);
	}

	// Hardware resources
	struct {
		VertexBuffer vb;
		IndexBuffer ib;
	} rects;

	// Allocation methods
	void _alloc_rects() {
		BFM_Settings vb_settings {
			.size = 1024,
			.usage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
			.usage_type = BFM_WRITE_ONLY
		};

		BFM_Settings ib_settings {
			.size = 1024,
			.usage = VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
			.usage_type = BFM_WRITE_ONLY
		};

		rects.vb = VertexBuffer(_wctx.context, vb_settings);
		rects.ib = IndexBuffer(_wctx.context, ib_settings);
	}

	// Pipeline creation structure
	template <size_t N>
	struct PipelineInfo {
		VkShaderModule				vert;
		VkShaderModule				frag;

		std::vector <VkDescriptorSetLayout>	dsls;

		VertexBinding				vertex_binding;
		std::array <VertexAttribute, N>		vertex_attributes;

		VkPrimitiveTopology			topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
	};

	// Create a graphics pipeline
	// TODO: vulkan method
	template <size_t N>
	Pipeline _make_pipeline(const PipelineInfo <N> &info) {
		// Create pipeline stages
		VkPipelineShaderStageCreateInfo vertex {
			.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
			.stage = VK_SHADER_STAGE_VERTEX_BIT,
			.module = info.vert,
			.pName = "main"
		};

		VkPipelineShaderStageCreateInfo fragment {
			.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
			.stage = VK_SHADER_STAGE_FRAGMENT_BIT,
			.module = info.frag,
			.pName = "main"
		};

		VkPipelineShaderStageCreateInfo shader_stages[] = { vertex, fragment };

		// Vertex input
		// auto binding_description = gui::Vertex::vertex_binding();
		// auto attribute_descriptions = gui::Vertex::vertex_attributes();

		VkPipelineVertexInputStateCreateInfo vertex_input {
			.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
			.vertexBindingDescriptionCount = 1,
			.pVertexBindingDescriptions = &info.vertex_binding,
			.vertexAttributeDescriptionCount
				= static_cast <uint32_t> (info.vertex_attributes.size()),
			.pVertexAttributeDescriptions = info.vertex_attributes.data()
		};

		// Input assembly
		VkPipelineInputAssemblyStateCreateInfo input_assembly {
			.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
			.topology = info.topology,
			.primitiveRestartEnable = VK_FALSE
		};

		// Viewport
		VkViewport viewport {
			.x = 0.0f,
			.y = 0.0f,
			.width = (float) _wctx.width,
			.height = (float) _wctx.height,
			.minDepth = 0.0f,
			.maxDepth = 1.0f
		};

		// Scissor
		VkRect2D scissor {
			.offset = {0, 0},
			.extent = _wctx.swapchain.extent
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
			.setLayoutCount = static_cast <uint32_t> (info.dsls.size()),
			.pSetLayouts = info.dsls.data(),
			.pushConstantRangeCount = 0
		};

		VkPipelineLayout pipeline_layout;
		VkResult result = vkCreatePipelineLayout(
			_wctx.context.vk_device(),
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
			.renderPass = _render_pass,
			.subpass = 0,
			.basePipelineHandle = VK_NULL_HANDLE,
			.basePipelineIndex = -1
		};

		VkPipeline pipeline;
		result = vkCreateGraphicsPipelines(
			_wctx.context.vk_device(),
			VK_NULL_HANDLE,
			1,
			&pipeline_info,
			nullptr,
			&pipeline
		);

		if (result != VK_SUCCESS) {
			throw std::runtime_error("failed to create graphics pipeline!");
		}

		Logger::ok("[profiler] Pipeline created");
		return {pipeline, pipeline_layout};
	}
public:
	// Default
	Layer() = default;

	// Constructor
	Layer(const App::Window &wctx, const VkAttachmentLoadOp &load = VK_ATTACHMENT_LOAD_OP_LOAD) : _wctx(wctx) {
		// Initialize all Vulkan objects
		_init_vulkan_structures(load);

		// Allocate RenderPacket data
		_alloc_rects();

		// Load all shaders
		// TODO: backend function to load a list of shaders
		VkShaderModule shapes_vertex = _wctx.context.vk->make_shader(
			_wctx.context.device,
			"shaders/bin/gui/basic_vert.spv"
		);

		VkShaderModule shapes_fragment = _wctx.context.vk->make_shader(
			_wctx.context.device,
			"shaders/bin/gui/basic_frag.spv"
		);

		// Create graphics pipelines
		PipelineInfo <2> grp_shapes_info {
			.vert = shapes_vertex,
			.frag = shapes_fragment,
			.dsls = {},
			.vertex_binding = Vertex::vertex_binding(),
			.vertex_attributes = Vertex::vertex_attributes()
		};

		_grp_shapes = _make_pipeline(grp_shapes_info);
	}

	// Add elements
	void add(const Element &element) {
		_elements.push_back(element);
	}

	void add(_element *ptr) {
		_elements.push_back(Element(ptr));
	}

	// Add multiple elements
	void add(const std::vector <Element> &elements) {
		_elements.insert(
			_elements.end(),
			elements.begin(),
			elements.end()
		);
	}

	void add(const std::vector <_element *> &elements) {
		for (auto &e : elements)
			_elements.push_back(Element(e));
	}

	// Load fonts
	void load_font(const std::string &alias, const std::string &path) {
		size_t index = _text_renders.size();
		_text_renders.push_back(TextRender(_wctx, _render_pass, path));
		_font_map[alias] = index;
	}

	// Get TextRender
	TextRender *text_render(int index) {
		return &_text_renders[index];
	}

	TextRender *text_render(const std::string &alias) {
		return text_render(_font_map.at(alias));
	}

	// Render using command buffer and framebuffer
	void render(const VkCommandBuffer &cmd_buffer, const VkFramebuffer &framebuffer) {
		// Start render pass
		// TODO: vulkan method
		VkClearValue clear_color = {0.0f, 0.0f, 0.0f, 1.0f};

		VkRenderPassBeginInfo render_pass_info {
			.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
			.renderPass = _render_pass,
			.framebuffer = framebuffer,
			.renderArea {
				.offset = {0, 0},
				.extent = _wctx.swapchain.extent
			},
			.clearValueCount = 1,
			.pClearValues = &clear_color
		};

		vkCmdBeginRenderPass(cmd_buffer, &render_pass_info, VK_SUBPASS_CONTENTS_INLINE);

		// Bind graphics pipeline
		vkCmdBindPipeline(cmd_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, _grp_shapes.pipeline);

		// Initialize render packet
		RenderPacket rp {
			.rects {
				.vb = &rects.vb,
				.ib = &rects.ib
			}
		};

		// Reset RenderPacket
		rp.reset();

		// Render all elements onto the RenderPacket
		for (auto &elem : _elements)
			elem->render_element(rp);

		// Sync RenderPacket
		rp.sync();

		// Render all parts of the RenderPacket
		// TODO: separate method

		// Draw rectangles
		VkBuffer	vbuffers[] = {rects.vb.vk_buffer()};
		VkDeviceSize	offsets[] = {0};

		vkCmdBindVertexBuffers(cmd_buffer, 0, 1, vbuffers, offsets);
		vkCmdBindIndexBuffer(cmd_buffer, rects.ib.vk_buffer(), 0, VK_INDEX_TYPE_UINT32);

		vkCmdDrawIndexed(cmd_buffer, rects.ib.push_size(), 1, 0, 0, 0);

		// Render all the text renders
		for (auto &tr : _text_renders)
			tr.render(_wctx.context, _wctx.command_pool, cmd_buffer);

		// End render pass
		vkCmdEndRenderPass(cmd_buffer);
	}
};

}

}

#endif

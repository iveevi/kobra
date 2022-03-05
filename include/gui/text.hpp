#ifndef TEXT_H_
#define TEXT_H_

// Standard headers
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

// Engine headers
#include "../backend.hpp"
#include "font.hpp"

namespace mercury {

namespace gui {

// Text class
// 	contains glyphs
// 	and is served by
// 	the text render class
struct Text {
	std::string		str;
	std::vector <Glyph>	glyphs;
};

// TextRender class
// 	holds Vulkan structures
// 	and renders for a single font
class TextRender {
public:
	// Required elements for construction
	// TODO: subcontext structure?
	struct Bootstrap {
		Vulkan::Context		ctx;
		VkDescriptorPool	pool;
		Vulkan::Swapchain	swapchain;
		VkRenderPass		renderpass;
		VkCommandPool		cpool;
	};
private:
	// Reference to glyph (in a text class)
	//	so that updating text is not a pain
	struct Ref {
		int		index;
		const Text	*text;

		// For sets
		bool operator==(const Ref &r) const {
			return index == r.index && text == r.text;
		}

		bool operator<(const Ref &r) const {
			return index < r.index;
		}
	};

	using RefSet = std::set <Ref>;

	// Map of each character to the set of glyphs
	// 	prevents the need to keep rebinding
	// 	the bitmaps
	std::unordered_map <char, RefSet>	_chars;

	// Font to use
	Font					_font;

	// Vulkan structures
	VkPipeline				_pipeline;
	VkPipelineLayout			_pipeline_layout;

	VkShaderModule				_vertex;
	VkShaderModule				_fragment;

	// Descriptors
	VkDescriptorSetLayout			_layout;
	VkDescriptorSet				_set;

	// Vertex buffer for text
	Glyph::VertexBuffer			_vbuf;

	// Texture and sampler for characters
	raster::TexturePacket			_tex;
	raster::Sampler				_sampler;

	// Screen dimensions
	float					_width;
	float					_height;

	// Create the pipeline
	void _make_pipeline(const Bootstrap &bs) {
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
		auto binding_description = gui::Vertex::vertex_binding();
		auto attribute_descriptions = gui::Vertex::vertex_attributes();

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
public:
	// Default constructor
	TextRender() {}

	// Constructor from paht to font file
	TextRender(const Bootstrap &bs, const std::string &path) {
		// Get context
		Vulkan::Context context = bs.ctx;

		// Create the descriptor set
		_layout = Glyph::make_bitmap_dsl(bs.ctx);
		_set = Glyph::make_bitmap_ds(bs.ctx, bs.pool);

		// Load shaders
		_vertex = context.vk->make_shader(context.device, "shaders/bin/gui/glyph_vert.spv");
		_fragment = context.vk->make_shader(context.device, "shaders/bin/gui/bitmap_frag.spv");

		// Allocate vertex buffer
		_vbuf = Glyph::VertexBuffer(bs.ctx, Glyph::vb_settings);

		// Create staging texture and sampler
		_tex = raster::make_texture(bs.ctx, bs.cpool,
			{
				.width = 1024,
				.height = 1024,
				.channels = 1
			},
			VK_FORMAT_R8_UNORM,
			VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
			VK_IMAGE_LAYOUT_UNDEFINED
		);

		// Transition layout appropriately
		_tex.transition_manual(bs.ctx, bs.cpool,
			VK_IMAGE_LAYOUT_UNDEFINED,
			VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
			VK_PIPELINE_STAGE_TRANSFER_BIT,
			VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT
		);

		_sampler = raster::Sampler(bs.ctx, _tex);
		_sampler.bind(bs.ctx, _set, 0);

		// Create pipeline
		_make_pipeline(bs);

		// Load font
		_font = Font(bs.ctx, bs.cpool, path);

		// Dimensions
		_width = bs.swapchain.extent.width;
		_height = bs.swapchain.extent.height;
	}

	// Create text object
	Text text(const std::string &text, const glm::vec2 &pos, const glm::vec4 &color) {
		float x = pos.x/_width;
		float y = pos.y/_height;

		// Initialize text object
		Text txt {.str = text};

		// Create glyphs
		for (char c : text) {
			// Get metrics for current character
			FT_Glyph_Metrics metrics = _font.metrics(c);

			// Get glyph bounds
			float w = metrics.width/_width;
			float h = metrics.height/_height;

			// Create glyph
			Glyph g {
				{x, y, x + w, y + h},
				color
			};

			txt.glyphs.push_back(g);

			// Advance
			x += w;
		}

		// Return text
		return txt;
	}

	// Add text to render
	void add(const Text &txt) {
		// Add each character to the table
		for (int i = 0; i < txt.str.size(); i++) {
			// Add to table
			RefSet &refs = _chars[txt.str[i]];
			refs.insert(refs.begin(), Ref {i, &txt});
		}
	}

	// Update vertex buffer
	void update(char c) {
		// Get context
		_vbuf.reset_push_back();

		// Get glyphs
		RefSet &refs = _chars[c];

		// Iterate over glyphs
		for (auto &ref : refs)
			ref.text->glyphs[ref.index].upload(_vbuf);

		// Update vertex buffer
		_vbuf.sync_size();
		_vbuf.upload();
	}

	// Render text
	void render(const Vulkan::Context &ctx, const VkCommandPool &cpool, const VkCommandBuffer &cmd) {
		// Iterate over characters
		for (auto &c : _chars) {
			// Get texture for character
			const raster::TexturePacket &tp = _font.bitmap(c.first);

			// Transition for copying
			_tex.transition_manual(ctx, cpool,
				VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
				VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
				VK_PIPELINE_STAGE_TRANSFER_BIT,
				VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT
			);

			// Copy to staging texture
			_tex.copy(ctx, cpool, tp,
				VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
				VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL
			);
			
			// Transition back
			_tex.transition_manual(ctx, cpool,
				VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
				VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
				VK_PIPELINE_STAGE_TRANSFER_BIT,
				VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT
			);

			// Update vertex buffer
			update(c.first);

			// Bind pipeline
			vkCmdBindPipeline(
				cmd,
				VK_PIPELINE_BIND_POINT_GRAPHICS,
				_pipeline
			);

			// Bind descriptor set
			vkCmdBindDescriptorSets(
				cmd,
				VK_PIPELINE_BIND_POINT_GRAPHICS,
				_pipeline_layout,
				0,
				1, &_set,
				0, nullptr
			);
		}
	}
};

}

}

#endif

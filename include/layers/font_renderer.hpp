#ifndef KOBRA_LAYERS_FONT_RENDERER_H_
#define KOBRA_LAYERS_FONT_RENDERER_H_

// Standard headers
#include <map>

// TODO: remove
#include <glm/gtx/string_cast.hpp>

// Engine headers
#include "../backend.hpp"
#include "../gui/font.hpp" // TODO: bring to this directory
#include "../ui/text.hpp"

namespace kobra {

namespace layers {

// TODO: move glyph struct here
// Renders chaarcters of a specific font
class FontRenderer {
	// Vulkan context
	Context				_ctx;

	// Pipline and descriptor set layout
	vk::raii::Pipeline		_pipeline = nullptr;
	vk::raii::PipelineLayout	_ppl = nullptr;

	vk::raii::DescriptorSetLayout	_dsl = nullptr;

	// Font to render
	gui::Font			_font;

	/* Glyph type
	struct Glyph {
	// TODO: use a smaller footprint glyph
	}; */

	// Buffer of glyphs
	BufferData			_glyphs = nullptr;

	// Construct text
	using GlyphMap = std::map <char, std::vector <gui::Glyph>>;

	void construct(GlyphMap &gmap, const ui::Text &text) {
		static const float factor = 1/1000.0f;

		glm::vec2 pos = text.anchor;

		pos.x /= _ctx.extent.width;
		pos.y /= _ctx.extent.height;

		pos.x = 2 * pos.x - 1;
		pos.y = 2 * pos.y - 1;

		float scale = text.size;

		/* KOBRA_LOG_FILE(notify) << "Creating text: " << text.text << " at "
			<< pos.x << ", " << pos.y << " with scale "
			<< scale << std::endl; */

		// NOTE: pos is the origin pos, not the top-left corner
		float x = pos.x;
		float y = pos.y;

		float minx = x, maxx = x;
		float miny = y, maxy = y;

		float iwidth = scale * factor/_ctx.extent.width;
		float iheight = scale * factor/_ctx.extent.height;

		// Array of glyphs
		std::vector <gui::Glyph> glyphs;

		// Create glyphs
		for (char c : text.text) {
			// If newline
			if (c == '\n') {
				x = pos.x;
				y += _font.line_height();
				continue;
			}

			// Get metrics for current character
			FT_Glyph_Metrics metrics = _font.metrics(c);

			// Get glyph top-left
			float x0 = x + (metrics.horiBearingX * iwidth);
			float y0 = y - (metrics.horiBearingY * iheight);

			miny = std::min(miny, y0);

			// Get glyph bounds
			float w = metrics.width * iwidth;
			float h = metrics.height * iheight;

			maxy = std::max(maxy, y0 + h);

			// Create glyph
			gui::Glyph g {
				{x0, y0, x0 + w, y0 + h},
				text.color
			};

			glyphs.push_back(g);

			// Advance
			x += metrics.horiAdvance * iwidth;
		}

		// Update text bounds
		maxx = x;

		// Reasses positioning
		float dx = pos.x - minx;
		float dy = pos.y - miny;

		// Move glyphs and add to glyph map
		for (int i = 0; i < glyphs.size(); i++) {
			gui::Glyph g = glyphs[i];
			g.move({dx, dy});

			// Add to glyph map
			char c = text.text[i];
			gmap[c].push_back(g);
		}
	}
public:
	// Default constructor
	FontRenderer() = default;

	// Constructor
	FontRenderer(const Context &ctx, const vk::raii::RenderPass &render_pass, const std::string &font_path)
			: _ctx(ctx),
			_font(*_ctx.phdev, *_ctx.device,
				*_ctx.command_pool,
				*_ctx.descriptor_pool,
				font_path
			) {
		// Create the descriptor set layout
		_dsl = make_descriptor_set_layout(
			*_ctx.device,
			{gui::Glyph::bitmap_binding}
		);

		// Pipline layout
		_ppl = vk::raii::PipelineLayout {
			*_ctx.device,
			{{}, *_dsl, {}}
		};

		// Pipeline
		auto shaders = make_shader_modules(*_ctx.device, {
			"shaders/bin/ui/glyph_vert.spv",
			"shaders/bin/ui/bitmap_frag.spv"
		});

		auto pipeline_cache = vk::raii::PipelineCache {*_ctx.device, {}};

		// Vertex binding and attribute descriptions
		auto binding_description = gui::Glyph::Vertex::vertex_binding();
		auto attribute_descriptions = gui::Glyph::Vertex::vertex_attributes();

		// Create the graphics pipeline
		GraphicsPipelineInfo grp_info {
			.device = *_ctx.device,
			.render_pass = render_pass,

			.vertex_shader = std::move(shaders[0]),
			.fragment_shader = std::move(shaders[1]),

			.vertex_binding = binding_description,
			.vertex_attributes = attribute_descriptions,

			.pipeline_layout = _ppl,
			.pipeline_cache = pipeline_cache,

			.depth_test = false,
			.depth_write = false,
		};

		_pipeline = make_graphics_pipeline(grp_info);

		// Create the buffer of glyphs
		vk::DeviceSize size = sizeof(gui::Glyph::Vertex) * 1024;

		_glyphs = BufferData(*_ctx.phdev, *_ctx.device, size,
			vk::BufferUsageFlagBits::eVertexBuffer,
			vk::MemoryPropertyFlagBits::eHostVisible
				| vk::MemoryPropertyFlagBits::eHostCoherent
		);
	}

	// Dimension of text in pixels
	// TODO: refactor the parameters
	glm::vec2 size(const ui::Text &text) const {
		static const float factor = 1/1000.0f;

		glm::vec2 pos = text.anchor;

		pos.x /= _ctx.extent.width;
		pos.y /= _ctx.extent.height;

		pos.x = 2 * pos.x - 1;
		pos.y = 2 * pos.y - 1;

		float scale = text.size;

		/* KOBRA_LOG_FILE(notify) << "Creating text: " << text.text << " at "
			<< pos.x << ", " << pos.y << " with scale "
			<< scale << std::endl; */

		// NOTE: pos is the origin pos, not the top-left corner
		float x = pos.x;
		float y = pos.y;

		float minx = x, maxx = x;
		float miny = y, maxy = y;

		float iwidth = scale * factor/_ctx.extent.width;
		float iheight = scale * factor/_ctx.extent.height;

		// Array of glyphs
		std::vector <gui::Glyph> glyphs;

		// Create glyphs
		for (char c : text.text) {
			// If newline
			if (c == '\n') {
				x = pos.x;
				y += _font.line_height();
				continue;
			}

			// Get metrics for current character
			FT_Glyph_Metrics metrics = _font.metrics(c);

			// Get glyph top-left
			float x0 = x + (metrics.horiBearingX * iwidth);
			float y0 = y - (metrics.horiBearingY * iheight);

			miny = std::min(miny, y0);

			// Get glyph bounds
			float w = metrics.width * iwidth;
			float h = metrics.height * iheight;

			maxy = std::max(maxy, y0 + h);

			// Advance
			x += metrics.horiAdvance * iwidth;
		}

		// Update text bounds
		maxx = x;

		// Return dimension
		return {
			_ctx.extent.width * (maxx - minx),
			_ctx.extent.height * (maxy - miny)
		};
	}

	// Render texts, assuming the render pass is active
	// TODO: should also pass in extent
	void render(const vk::raii::CommandBuffer &cmd, const std::vector <ui::Text> &texts) {
		// Map of character to glyphs
		GlyphMap glyph_map;
		for (const ui::Text &text : texts)
			construct(glyph_map, text);

		// Process each character set
		// TODO: might be a bad idea to repeatedly upload to the buffer
		std::vector <gui::Glyph::Vertex> vertices;

		std::vector <vk::DeviceSize> offsets;
		std::vector <size_t> counts;
		std::vector <char> chars;

		for (auto &pair : glyph_map) {
			// Get character
			char c = pair.first;
			auto &glyphs = pair.second;

			// Compute offset
			offsets.push_back(vertices.size());
			for (const gui::Glyph &glyph : glyphs) {
				// Get vertices
				glyph.upload(vertices);
			}

			// Add to arrays
			counts.push_back(glyphs.size());
			chars.push_back(c);
		}

		// Upload data
		_glyphs.upload(vertices);

		// Bind pipeline
		cmd.bindPipeline(vk::PipelineBindPoint::eGraphics, *_pipeline);

		// Begin rendering process
		for (int i = 0; i < chars.size(); i++) {
			// Get descriptor set for glyph
			const auto &dset = *_font.glyph_ds(chars[i]);

			// Bind descriptor set
			cmd.bindDescriptorSets(
				vk::PipelineBindPoint::eGraphics,
				*_ppl, 0, {dset}, {}
			);

			// Bind buffer
			cmd.bindVertexBuffers(0, {*_glyphs.buffer}, {0});

			// Draw
			cmd.draw(6 * counts[i], 1, offsets[i], 0);
		}
	}
};

}

}

#endif

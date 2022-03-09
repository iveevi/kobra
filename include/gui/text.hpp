#ifndef TEXT_H_
#define TEXT_H_

// Standard headers
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

// Engine headers
#include "../app.hpp"
#include "../backend.hpp"
#include "font.hpp"
#include "gui.hpp"

namespace mercury {

namespace gui {

// Forward declarations
class TextRender;

// Text class
// 	contains glyphs
// 	and is served by
// 	the text render class
class Text : public _element {
	std::vector <Glyph>	_glyphs;
	TextRender *		_origin;

	// Cache previous text properties
	std::string		_str;

	glm::vec2		_pos;
	glm::vec3		_color;
	glm::vec4		_bounds;
	float			_scale;
public:
	std::string		str;

	glm::vec2		position;
	glm::vec3		color;
	glm::vec4		bounds;
	float			scale;

	// Update by using the TextRender class
	void refresh();

	// Render
	void render(RenderPacket &rp) override {
		// Check if the text has changed
		if (scale != _scale || bounds != _bounds || _str != str) {
			// Update the text
			refresh();
		} else {
			// These changes do not need a full refresh
			if (color != _color) {
				// Update the color
				for (Glyph &g : _glyphs)
					g.color() = color;
			}

			if (position != _pos) {
				glm::vec2 delta = position - _pos;
				for (Glyph &g : _glyphs)
					g.move(delta);
			}
		}

		// Update values
		_str = str;
		_pos = position;
		_color = color;
		_bounds = bounds;
		_scale = scale;
	}

	// Friend classes
	friend class TextRender;
};

// TextRender class
// 	holds Vulkan structures
// 	and renders for a single font
class TextRender {
private:
	// Reference to glyph (in a text class)
	//	so that updating text is not a pain
	struct Ref {
		int		index;
		const Text	*text;

		bool operator<(const Ref &r) const {
			return text < r.text
				|| (text == r.text && index < r.index);
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

	// Vertex buffer for text
	Glyph::VertexBuffer			_vbuf;

	// Screen dimensions
	float					_width;
	float					_height;

	// Create the pipeline
	void _make_pipeline(const App::Window &, const VkRenderPass &);

	// Remove previous references to a Text object
	void remove(Text *text) {
		// Iterate through each character
		for (auto &pair : _chars) {
			// Remove the references
			for (auto it = pair.second.begin(); it != pair.second.end();) {
				if (it->text == text)
					it = pair.second.erase(it);
				else
					++it;
			}
		}
	}
public:
	// Default constructor
	TextRender() {}

	// Constructor from paht to font file
	TextRender(const App::Window &wctx, const VkRenderPass &render_pass, const std::string &path) {
		// Get context
		Vulkan::Context context = wctx.context;

		// Create the descriptor set
		// TODO: remove later, use the ones from font
		_layout = Glyph::make_bitmap_dsl(context);

		// Load shaders
		_vertex = context.vk->make_shader(context.device, "shaders/bin/gui/glyph_vert.spv");
		_fragment = context.vk->make_shader(context.device, "shaders/bin/gui/bitmap_frag.spv");

		// Allocate vertex buffer
		_vbuf = Glyph::VertexBuffer(context, Glyph::vb_settings);

		// Create pipeline
		_make_pipeline(wctx, render_pass);

		// Load font
		_font = Font(context,
			wctx.command_pool,
			wctx.descriptor_pool,
			path
		);

		// Dimensions
		_width = wctx.width;
		_height = wctx.height;
	}

	// Create text object
	// TODO: remove this, do the text method in the next one, then add
	// TODO: make text will also add the text in the heap vector
	Text *text(const std::string &text, const glm::vec2 &pos, const glm::vec4 &color, float scale = 1.0f) {
		static const float factor = 1/1000.0f;

		// NOTE: pos is the origin pos, not the top-left corner
		float x = 2 * pos.x/_width - 1.0f;
		float y = 2 * pos.y/_height - 1.0f;

		float minx = x, maxx = x;
		float miny = y, maxy = y;

		float iwidth = scale * factor/_width;
		float iheight = scale * factor/_height;

		// Initialize text object
		Text *txt = new Text();

		txt->str = text;
		txt->color = color;
		txt->scale = scale;
		txt->position = pos;
		txt->_origin = this;

		// Create glyphs
		for (char c : text) {
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
			Glyph g {
				{x0, y0, x0 + w, y0 + h},
				color
			};

			txt->_glyphs.push_back(g);

			// Advance
			x += metrics.horiAdvance * iwidth;
		}

		// Update text bounds
		maxx = x;
		txt->bounds = {
			minx, miny,
			maxx, maxy
		};

		// TODO: use set position on text to correct its position

		// Return text
		return txt;
	}

	// TODO: use a more immediate approach
	// 	since every text object holds the text render origin,
	// 	we can add the text and then render
	void text(Text *txt) {
		static const float factor = 1/1000.0f;

		// NOTE: pos is the origin pos, not the top-left corner
		float x = 2 * txt->position.x/_width - 1.0f;
		float y = 2 * txt->position.y/_height - 1.0f;

		float minx = x, maxx = x;
		float miny = y, maxy = y;

		float iwidth = txt->scale * factor/_width;
		float iheight = txt->scale * factor/_height;

		// Clear old glyphs
		txt->_glyphs.clear();

		// Create glyphs
		for (char c : txt->str) {
			// std::cout << "\tc = " << c << " (" << (int) c << ")" << std::endl;
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

			// std::cout << "\t\tbounds = " << x0 << ", " << y0 << ", " << w << ", " << h << std::endl;

			// Create glyph
			Glyph g {
				{x0, y0, x0 + w, y0 + h},
				txt->color
			};

			txt->_glyphs.push_back(g);

			// Advance
			x += metrics.horiAdvance * iwidth;
		}

		// Update text bounds
		maxx = x;
		txt->bounds = {
			minx, miny,
			maxx, maxy
		};
	}

	// Add text to render
	// TODO: store and free text
	void add(Text *text) {
		remove(text);

		// Add each character to the table
		for (int i = 0; i < text->str.size(); i++) {
			// Add to table
			RefSet &refs = _chars[text->str[i]];
			refs.insert(refs.begin(), Ref {i, text});
		}
	}

	// Update vertex buffer
	size_t update(char c) {
		// Get glyphs
		RefSet &refs = _chars[c];

		// Iterate over glyphs
		for (auto &ref : refs)
			ref.text->_glyphs[ref.index].upload(_vbuf);

		return refs.size();
	}

	// Draw call structure
	struct Draw {
		size_t offset;
		size_t number;
	};

	std::vector <Draw> update() {
		_vbuf.reset_push_back();

		// Get glyphs
		std::vector <Draw> draws;

		// Iterate over glyphs
		for (auto &refs : _chars) {
			size_t offset = _vbuf.push_size();
			size_t number = update(refs.first);

			draws.push_back({offset, number});
		}

		_vbuf.sync_size();
		_vbuf.upload();

		return draws;
	}

	// Render text
	void render(const Vulkan::Context &ctx, const VkCommandPool &cpool, const VkCommandBuffer &cmd) {
		// Bind pipeline
		vkCmdBindPipeline(
			cmd,
			VK_PIPELINE_BIND_POINT_GRAPHICS,
			_pipeline
		);

		// Update vertex buffer
		std::vector <Draw> draws = update();
		size_t i = 0;

		// Iterate over characters
		for (auto &c : _chars) {
			// Get descriptor set for glyph
			VkDescriptorSet set = _font.glyph_ds(c.first);

			// Bind descriptor set
			vkCmdBindDescriptorSets(
				cmd,
				VK_PIPELINE_BIND_POINT_GRAPHICS,
				_pipeline_layout,
				0,
				1, &set,
				0, nullptr
			);

			// Bind vertex buffer
			VkBuffer buffers[] {_vbuf.vk_buffer()};
			VkDeviceSize offsets[] {0};

			vkCmdBindVertexBuffers(
				cmd,
				0, 1, buffers, offsets
			);

			// Draw call info
			Draw draw = draws[i++];

			// Draw
			vkCmdDraw(
				cmd,
				6 * draw.number, 1,
				draw.offset, 0
			);
		}
	}
};

}

}

#endif

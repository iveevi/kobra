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

namespace kobra {

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

	glm::vec2		pos;
	glm::vec3		color;
	// glm::vec4		bounds;
	float			scale;

	// Update by using the TextRender class
	void refresh();

	// Virtual methods
	glm::vec2 position() const override {
		return pos;
	}

	glm::vec4 bounding_box() const override {
		return _bounds;
	}

	// NDC width and height
	float width() const {
		return _bounds.z - _bounds.x;
	}

	float height() const {
		return _bounds.w - _bounds.y;
	}

	// Latching onto a layer
	void latch(LatchingPacket &lp) override {}

	// Render
	void render(RenderPacket &rp) override {
		// Check if the text has changed
		if (scale != _scale || _str != str) {
			// Update the text
			refresh();
		} else {
			// These changes do not need a full refresh
			if (color != _color) {
				// Update the color
				for (Glyph &g : _glyphs)
					g.color() = color;
			}

			if (pos != _pos) {
				glm::vec2 delta = pos - _pos;
				_bounds += glm::vec4(delta, delta);
				for (Glyph &g : _glyphs)
					g.move(delta);
			}
		}

		// Update values
		_str = str;
		_pos = pos;
		_color = color;
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
	vk::raii::Pipeline			_pipeline = nullptr;
	vk::raii::PipelineLayout		_pipeline_layout = nullptr;

	// Descriptors
	vk::raii::DescriptorSetLayout		_descriptor_set_layout = nullptr;

	// Vertex buffer for text
	BufferData				_device_buffer = nullptr;
	std::vector <Glyph::Vertex>		_host_buffer;

	// Screen dimensions
	// TODO: do we really need width and hieght (can also be done during
	// rendering)
	float					_width;
	float					_height;

	// Create the pipeline
	void _make_pipeline(const vk::raii::Device &, const vk::raii::RenderPass &);

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
	TextRender() = default;

	// Constructor from paht to font file
	TextRender(const vk::raii::PhysicalDevice &phdev,
			const vk::raii::Device &device,
			const vk::raii::CommandPool &command_pool,
			const vk::raii::DescriptorPool &descriptor_pool,
			const vk::raii::RenderPass &render_pass,
			const std::string &path,
			float width, float height)
			: _width(width), _height(height) {
		// Create the descriptor set
		// TODO: remove later, use the ones from font
		// _layout = Glyph::make_bitmap_dsl(context);
		_descriptor_set_layout = make_descriptor_set_layout(
			device, {Glyph::bitmap_binding},
			vk::DescriptorSetLayoutCreateFlagBits::eUpdateAfterBindPool
		);

		// Allocate vertex buffer
		// TODO: auto resizing for vertex buffer
		_device_buffer = BufferData(phdev, device,
			sizeof(Glyph::Vertex) * 1000,	// TODO: constant
			vk::BufferUsageFlagBits::eVertexBuffer,
			vk::MemoryPropertyFlagBits::eHostVisible
				| vk::MemoryPropertyFlagBits::eHostCoherent
		);

		// Create pipeline
		_make_pipeline(device, render_pass);

		// Load font
		_font = Font(phdev, device,
			command_pool,
			descriptor_pool,
			path
		);
	}

	// Create text object
	// TODO: remove this, do the text method in the next one, then add
	// TODO: make text will also add the text in the heap vector
	Text *text(const std::string &text, const glm::vec2 &pos, const glm::vec4 &color, float scale = 1.0f) {
		static const float factor = 1/1000.0f;

		KOBRA_LOG_FILE(notify) << "Creating text: " << text << " at "
			<< pos.x << ", " << pos.y << " with scale "
			<< scale << std::endl;

		// NOTE: pos is the origin pos, not the top-left corner
		float x = pos.x;
		float y = pos.y;

		float minx = x, maxx = x;
		float miny = y, maxy = y;

		float iwidth = scale * factor/_width;
		float iheight = scale * factor/_height;

		// Initialize text object
		Text *txt = new Text();

		txt->str = text;
		txt->color = color;
		txt->scale = scale;
		txt->pos = pos;
		txt->_origin = this;

		// Create glyphs
		for (char c : text) {
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
		txt->_bounds = {
			minx, miny,
			maxx, maxy
		};

		// Reasses positioning
		float dx = pos.x - minx;
		float dy = pos.y - miny;

		// Move glyphs
		for (Glyph &g : txt->_glyphs)
			g.move({dx, dy});

		// Change bounds
		txt->_bounds += glm::vec4(dx, dy, dx, dy);

		// TODO: use set position on text to correct its position

		// Return text
		return txt;
	}

	// With screen coordinates
	Text *text(const std::string &str, const coordinates::Screen &sc, const glm::vec4 &color, float scale = 1.0f) {
		return text(str, sc.to_ndc(), color, scale);
	}

	// TODO: use a more immediate approach
	// 	since every text object holds the text render origin,
	// 	we can add the text and then render
	void text(Text *txt) {
		static const float factor = 1/1000.0f;

		/* NOTE: pos is the origin pos, not the top-left corner
		float x = 2 * txt->pos.x/_width - 1.0f;
		float y = 2 * txt->pos.y/_height - 1.0f; */

		float x = txt->pos.x;
		float y = txt->pos.y;

		float minx = x, maxx = x;
		float miny = y, maxy = y;

		float iwidth = txt->scale * factor/_width;
		float iheight = txt->scale * factor/_height;

		// Clear old glyphs
		txt->_glyphs.clear();

		// Create glyphs
		for (char c : txt->str) {
			// If newline
			if (c == '\n') {
				x = txt->pos.x;
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
		txt->_bounds = {
			minx, miny,
			maxx, maxy
		};

		// Reasses positioning
		// TODO: method
		float dx = txt->pos.x - minx;
		float dy = txt->pos.y - miny;

		// Move glyphs
		for (Glyph &g : txt->_glyphs)
			g.move({dx, dy});

		// Change bounds
		txt->_bounds += glm::vec4(dx, dy, dx, dy);
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
			ref.text->_glyphs[ref.index].upload(_host_buffer);

		return refs.size();
	}

	// Draw call structure
	struct Draw {
		size_t offset;
		size_t number;
	};

	std::vector <Draw> update() {
		_host_buffer.clear();

		// Get glyphs
		std::vector <Draw> draws;

		// Iterate over glyphs
		for (auto &refs : _chars) {
			size_t offset = _host_buffer.size();
			size_t number = update(refs.first);

			draws.push_back({offset, number});
		}

		// Upload
		// TODO: resize
		_device_buffer.upload(_host_buffer);

		return draws;
	}

	// Render text
	void render(const vk::raii::CommandBuffer &cmd) {
		// Bind pipeline
		cmd.bindPipeline(vk::PipelineBindPoint::eGraphics, *_pipeline);

		// Update vertex buffer
		std::vector <Draw> draws = update();
		size_t i = 0;

		// Iterate over characters
		for (auto &c : _chars) {
			// Get descriptor set for glyph
			const auto &dset = *_font.glyph_ds(c.first);

			// Bind descriptor set
			cmd.bindDescriptorSets(
				vk::PipelineBindPoint::eGraphics,
				*_pipeline_layout, 0, {dset}, {}
			);

			// Bind buffer
			cmd.bindVertexBuffers(0, {*_device_buffer.buffer}, {0});

			// Draw call info
			Draw draw = draws[i++];

			// Draw
			cmd.draw(6 * draw.number, 1, draw.offset, 0);
		}
	}
};

}

}

#endif

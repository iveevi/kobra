#ifndef FONT_H_
#define FONT_H_

// Standard headers
#include <array>
#include <string>
#include <unordered_map>

// FreeType headers
#include <ft2build.h>
#include <vulkan/vulkan_core.h>
#include FT_FREETYPE_H
#include FT_BBOX_H
#include FT_OUTLINE_H

// Engine headers
#include "../common.hpp"
#include "../logger.hpp"
#include "../texture.hpp"
#include "gui.hpp"

namespace mercury {

namespace gui {

// Glyph outline structure
struct GlyphOutline {
	// Character code
	uint32_t code;

	// Glyph metrics
	struct Metrics {
		float xbear;
		float ybear;

		float width;
		float height;
	} metrics;

	// Start of current contour
	glm::vec2 start;

	// Store outline data as a list
	// of quadratic bezier curves
	BufferManager <glm::vec2> outline;

	// Default constructor
	GlyphOutline() {}

	// Constructor takes bounding box
	// TODO: first vec2 contains number of curves
	GlyphOutline(const Vulkan::Context &ctx, uint32_t c, const Metrics &m)
			: code(c), metrics(m) {
		// Allocate space for outline data
		BFM_Settings outline_settings {
			.size = 256,
			.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
			.usage_type = BFM_WRITE_ONLY
		};

		outline = BufferManager <glm::vec2> (ctx, outline_settings);
	}

	// Push a new point to the list
	void push(const glm::vec2 &p) {
		outline.push_back(p);
	}

	glm::vec2 convert(const FT_Vector *point) {
		// Normalize coordinates
		float x = (point->x - metrics.xbear) / metrics.width;
		float y = (point->y + metrics.height - metrics.ybear) / metrics.height;
		return glm::vec2 {x/2 + 0.1, y/2 + 0.1};
	}

	// Bind buffer to descriptor set
	void bind(const VkDescriptorSet &ds, uint32_t binding) {
		outline.bind(ds, binding);
	}

	// Sync and upload data to GPU
	void upload() {
		// Insert size vector
		outline.push_back(start);
		size_t size = outline.push_size();
		outline.push_front(glm::vec2 {
			static_cast <float> (size),
			0.0f
		});

		// TODO: brute force normalize?

		outline.sync_size();
		outline.upload();
	}

	// TODO: debugging only
	// Dump outline data to console
	void dump() const {
		Logger::ok() << "Glyph outline: ";
		for (const auto &p : outline.vector())
			Logger::plain() << "(" << p.x << ", " << p.y << "), ";
		Logger::plain() << std::endl;
	}
};

// Glyph structure
// TODO: text class will hold shaders and stuff
class Glyph {
public:
	// Vertex and buffer type
	struct Vertex {
		glm::vec4 bounds;

		// Get vertex binding description
		static VertexBinding vertex_binding() {
			return VertexBinding {
				.binding = 0,
				.stride = sizeof(Vertex),
				.inputRate = VK_VERTEX_INPUT_RATE_VERTEX
			};
		}

		// Get vertex attribute descriptions
		static std::array <VertexAttribute, 1> vertex_attributes() {
			return std::array <VertexAttribute, 1> {
				VertexAttribute {
					.location = 0,
					.binding = 0,
					.format = VK_FORMAT_R32G32B32A32_SFLOAT,
					.offset = offsetof(Vertex, bounds)
				},
			};
		}
	};

	using VertexBuffer = BufferManager <Vertex>;
private:
	// Vertex and index data
	glm::vec4	_bounds;
	glm::vec3	_color	= glm::vec3 {1.0};
public:
	// Constructor
	Glyph() {}
	Glyph(glm::vec4 bounds) : _bounds(bounds) {}

	// Render the glyph
	// TODO: render method or upload method (instacing)?
	void upload(VertexBuffer &vb) {
		std::array <Vertex, 6> vertices {
			Vertex {_bounds},
			Vertex {_bounds},
			Vertex {_bounds},
			Vertex {_bounds},
			Vertex {_bounds},
			Vertex {_bounds}
		};

		vb.push_back(vertices);
	}

	// Static buffer properties
	static constexpr BFM_Settings vb_settings {
		.size = 1024,
		.usage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
		.usage_type = BFM_WRITE_ONLY
	};

	// TODO: remove
	static constexpr BFM_Settings ib_settings {
		.size = 1024,
		.usage = VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
		.usage_type = BFM_WRITE_ONLY
	};

	// Descriptor sets for shader
	static constexpr VkDescriptorSetLayoutBinding bitmap_dsl {
		.binding = 0,
		.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
		.descriptorCount = 1,
		.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT
	};

	static constexpr VkDescriptorSetLayoutBinding outline_dsl {
		.binding = 0,
		.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
		.descriptorCount = 1,
		.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT
	};

	// Make descriptor set layout
	static VkDescriptorSetLayout make_bitmap_dsl(const Vulkan::Context &ctx) {
		static VkDescriptorSetLayout dsl = VK_NULL_HANDLE;

		if (dsl != VK_NULL_HANDLE)
			return dsl;

		// Create layout if not created
		dsl = ctx.vk->make_descriptor_set_layout(
			ctx.device,
			{ bitmap_dsl }
		);

		return dsl;
	}

	static VkDescriptorSetLayout make_outline_dsl(const Vulkan::Context &ctx) {
		static VkDescriptorSetLayout dsl = VK_NULL_HANDLE;

		if (dsl != VK_NULL_HANDLE)
			return dsl;

		// Create layout if not created
		dsl = ctx.vk->make_descriptor_set_layout(
			ctx.device,
			{ outline_dsl }
		);

		return dsl;
	}

	// Make descriptor set
	static VkDescriptorSet make_bitmap_ds(const Vulkan::Context &ctx, const VkDescriptorPool &pool) {
		static VkDescriptorSet ds = VK_NULL_HANDLE;

		if (ds != VK_NULL_HANDLE)
			return ds;

		// Create descriptor set
		ds = ctx.vk->make_descriptor_set(
			ctx.device,
			pool,
			make_bitmap_dsl(ctx)
		);

		return ds;
	}

	static VkDescriptorSet make_outline_ds(const Vulkan::Context &ctx, const VkDescriptorPool &pool) {
		static VkDescriptorSet ds = VK_NULL_HANDLE;

		if (ds != VK_NULL_HANDLE)
			return ds;

		// Create descriptor set
		ds = ctx.vk->make_descriptor_set(
			ctx.device,
			pool,
			make_outline_dsl(ctx)
		);

		return ds;
	}
};

// Font class holds information
//	about a single font
class Font {
	// Font bitmaps
	std::unordered_map <char, raster::TexturePacket>	_bitmaps;

	// Font outlines
	std::unordered_map <char, GlyphOutline>			_glyphs;

	// Check FreeType error
	void check_error(FT_Error error) const {
		if (error) {
			Logger::error() << "FreeType error: "
				<< error << std::endl;
			throw -1;
		}
	}

	// Outline processors
	static FT_Error _outline_move_to(const FT_Vector *to, void *user) {
		GlyphOutline *outline = (GlyphOutline *) user;
		glm::vec2 pt = outline->convert(to);
		if (outline->outline.push_size() > 0) {
			Logger::warn() << "Move pre --> " << outline->start.x << ", "
				<< outline->start.y << " --> size = " << outline->outline.push_size()
				<< std::endl;
			// outline->push(pt);
			// outline->push({-1, -1});
			outline->push(pt);
		}

		assert(outline->outline.push_size() % 2 == 0);

		outline->push(pt);
		outline->start = pt;
		if (outline->code == 'g') {
			Logger::ok() << "Move to: " << pt.x << ", " << pt.y << std::endl;
		}
		return 0;
	}

	static FT_Error _outline_line_to(const FT_Vector *to, void *user) {
		GlyphOutline *outline = (GlyphOutline *) user;
		glm::vec2 curr = outline->convert(to);
		outline->push(curr);
		outline->push(curr);
		if (outline->code == 'g')
			Logger::ok() << "Line to: " << to->x << ", " << to->y << std::endl;
		return 0;
	}

	static FT_Error _outline_conic_to(const FT_Vector *control, const FT_Vector *to, void *user) {
		GlyphOutline *outline = (GlyphOutline *) user;
		outline->push(outline->convert(control));
		outline->push(outline->convert(to));
		if (outline->code == 'g')
			Logger::ok() << "Conic to: " << control->x << ", " << control->y << ", " << to->x << ", " << to->y << std::endl;
		return 0;
	}

	static FT_Error _outline_cubic_to(const FT_Vector *control1, const FT_Vector *control2, const FT_Vector *to, void *user) {
		GlyphOutline *outline = (GlyphOutline *) user;
		/* outline->push(control1);
		outline->push(control2);
		outline->push(to); */
		Logger::error() << "IGNORED CUBIC TO" << std::endl;
		return 0;
	}

	// Load FreeType library
	void load_font(const Vulkan::Context &ctx, const VkCommandPool &cpool, const std::string &file) {
		// Load library
		FT_Library library;
		check_error(FT_Init_FreeType(&library));

		// Load font
		FT_Face face;
		check_error(FT_New_Face(library, file.c_str(), 0, &face));

		// Set font size
		check_error(FT_Set_Char_Size(face, 0, 1000 * 64, 96, 96));

		// Process
		size_t total_points = 0;
		size_t total_cells = 0;

		for (size_t i = 0; i < 96; i++) {
			char c = i + ' ';
			std::cout << "Processing character '" << c << "'" << std::endl;

			/* Get outline
			FT_UInt glyph_index = FT_Get_Char_Index(face, c);
			check_error(FT_Load_Glyph(face, glyph_index, FT_LOAD_DEFAULT));

			// Process outline
			FT_Outline_Funcs funcs = {
				.move_to = _outline_move_to,
				.line_to = _outline_line_to,
				.conic_to = _outline_conic_to,
				.cubic_to = _outline_cubic_to,
			};

			// Get bounding box
			GlyphOutline outline = {ctx, (uint32_t) c, {
				.xbear = (float) face->glyph->metrics.horiBearingX,
				.ybear = (float) face->glyph->metrics.horiBearingY,
				.width = (float) face->glyph->metrics.width,
				.height = (float) face->glyph->metrics.height
			}};

			// Get number of curves
			FT_Outline_Decompose(&face->glyph->outline, &funcs, &outline);
			outline.upload();

			if (c == 'g') {
				Logger::warn() << "GLYPH: " << c << std::endl;
				Logger::warn() << "\txbear: " << outline.metrics.xbear << std::endl;
				Logger::warn() << "\tybear: " << outline.metrics.ybear << std::endl;
				outline.dump();
			}

			// Add to glyphs
			_glyphs[c] = outline; */

			// Load bitmap into texture
			FT_Load_Char(face, c, FT_LOAD_RENDER);
			uint width = face->glyph->bitmap.width;
			uint height = face->glyph->bitmap.rows;
			Texture tex {
				.width = width,
				.height = height,
				.channels = 1
			};

			Logger::warn() << "Glyph: " << width << " x " << height << std::endl;
			if (width * height == 0)
				continue;

			// Load texture
			tex.data = bytes(width * height);
			memcpy(tex.data.data(), face->glyph->bitmap.buffer, width * height);
			Logger::warn() << "data size = " << tex.data.size() << std::endl;

			int nulls = 0;
			for (auto &b : tex.data) {
				if (b == 0)
					nulls++;
			}

			Logger::warn() << "tex.data Nulls: " << nulls << std::endl;

			nulls = 0;
			for (int i = 0; i < tex.data.size(); i++) {
				if (face->glyph->bitmap.buffer[i] == 0)
					nulls++;
			}

			Logger::warn() << "face->glyph->bitmap.buffer Nulls: " << nulls << std::endl;

			// Create texture
			raster::TexturePacket tp = raster::make_texture(
				ctx, cpool, tex,
				VK_FORMAT_R8_UNORM
			);

			// Add to dictionary
			_bitmaps[c] = tp;
		}
	}
public:
	Font() {}
	Font(const Vulkan::Context &ctx, const VkCommandPool &cpool, const std::string &file) {
		// Check that the file exists
		if (!file_exists(file)) {
			Logger::error("Font file not found: " + file);
			throw -1;
		}

		// Load font
		load_font(ctx, cpool, file);
	}

	// Retrieve glyph outline
	const GlyphOutline &operator[](char c) const {
		auto it = _glyphs.find(c);
		if (it == _glyphs.end()) {
			Logger::error() << "Glyph not found: " << c << std::endl;
			throw -1;
		}

		return it->second;
	}

	// Retrieve glyph bitmap
	const raster::TexturePacket &bitmap(char c) const {
		auto it = _bitmaps.find(c);
		if (it == _bitmaps.end()) {
			Logger::error() << "Glyph not found: " << c << std::endl;
			throw -1;
		}

		return it->second;
	}
};

}

}

#endif

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
#include "gui.hpp"

namespace mercury {

namespace gui {

// Glyph outline structure
struct GlyphOutline {
	// Glyph bounding box
	glm::vec4 bbox;

	// Store outline data as a list
	// of quadratic bezier curves
	BufferManager <glm::vec2> outline;

	// Default constructor
	GlyphOutline() : bbox {0.0f} {}

	// Constructor takes bounding box
	// TODO: first vec2 contains number of curves
	GlyphOutline(const Vulkan::Context &ctx, const glm::vec4 &b) : bbox {b} {
		// Allocate space for outline data
		BFM_Settings outline_settings {
			.size = 100,
			.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
			.usage_type = BFM_WRITE_ONLY
		};

		outline = BufferManager <glm::vec2> (ctx, outline_settings);
	}

	// Push a new point to the list
	void push(const glm::vec2 &p) {
		outline.push_back(p);
	}

	void push(const FT_Vector *point) {
		// Normalize coordinates
		outline.push_back(glm::vec2 {
			static_cast <float> (point->x) / (bbox.z - bbox.x),
			static_cast <float> (point->y) / (bbox.w - bbox.y)
		});
	}

	// Bind buffer to descriptor set
	void bind(const VkDescriptorSet &ds, uint32_t binding) {
		outline.bind(ds, binding);
	}

	// Sync and upload data to GPU
	void upload() {
		// Insert size vector
		size_t size = outline.push_size();
		outline.push_front(glm::vec2 {
			static_cast <float> (size),
			0.0f
		});

		outline.sync_size();
		outline.upload();
	}

	// TODO: debugging only
	// Dump outline data to console
	void dump() const {
		Logger::ok() << "Glyph outline: ";
		for (const auto &p : outline.vector())
			Logger::plain() << "(" << p.x << ", " << p.y << "); ";
		Logger::plain() << std::endl;
	}
};

// Glyph structure
// TODO: text class will hold shaders and stuff
class Glyph {
	// Vertex and index data
	glm::vec4	_bounds;
	glm::vec3	_color	= glm::vec3 {1.0};
public:
	// Constructor
	Glyph() {}
	Glyph(glm::vec4 bounds) : _bounds(bounds) {}

	// Render the glyph
	void upload(VertexBuffer &vb, IndexBuffer &ib) {
		// Create vertex data
		std::array <Vertex, 4> vertices {
			Vertex { glm::vec2 { _bounds.x, _bounds.y }, _color },
			Vertex { glm::vec2 { _bounds.x, _bounds.w }, _color },
			Vertex { glm::vec2 { _bounds.z, _bounds.w }, _color },
			Vertex { glm::vec2 { _bounds.z, _bounds.y }, _color }
		};

		uint32_t vsize = vb.push_size();
		std::array <uint32_t, 6> indices {
			vsize, vsize + 2, vsize + 1,
			vsize, vsize + 3, vsize + 2
		};

		// Upload vertex data
		vb.push_back(vertices);
		ib.push_back(indices);
	}

	// Static buffer properties
	static constexpr BFM_Settings vb_settings {
		.size = 1024,
		.usage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
		.usage_type = BFM_WRITE_ONLY
	};

	static constexpr BFM_Settings ib_settings {
		.size = 1024,
		.usage = VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
		.usage_type = BFM_WRITE_ONLY
	};

	// Descriptor set for shader
	static constexpr VkDescriptorSetLayoutBinding glyph_dsl {
		.binding = 0,
		.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
		.descriptorCount = 1,
		.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT
	};

	// Make descriptor set layout
	static VkDescriptorSetLayout make_glyph_dsl(const Vulkan::Context &ctx) {
		static VkDescriptorSetLayout dsl = VK_NULL_HANDLE;

		if (dsl != VK_NULL_HANDLE)
			return dsl;
		
		// Create layout if not created
		dsl = ctx.vk->make_descriptor_set_layout(
			ctx.device,
			{ glyph_dsl }
		);

		return dsl;
	}

	// Make descriptor set
	static VkDescriptorSet make_glyph_ds(const Vulkan::Context &ctx, const VkDescriptorPool &pool) {
		static VkDescriptorSet ds = VK_NULL_HANDLE;

		if (ds != VK_NULL_HANDLE)
			return ds;
		
		// Create descriptor set
		ds = ctx.vk->make_descriptor_set(
			ctx.device,
			pool,
			make_glyph_dsl(ctx)
		);

		return ds;
	}
};

// Font class holds information
//	about a single font
class Font {
	// Font data
	std::unordered_map <char, GlyphOutline> _glyphs;

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
		outline->push({0, 0});
		outline->push(to);
		return 0;
	}

	static FT_Error _outline_line_to(const FT_Vector *to, void *user) {
		GlyphOutline *outline = (GlyphOutline *) user;
		outline->push(to);
		return 0;
	}

	static FT_Error _outline_conic_to(const FT_Vector *control, const FT_Vector *to, void *user) {
		GlyphOutline *outline = (GlyphOutline *) user;
		outline->push(control);
		outline->push(to);
		return 0;
	}

	// Ignore cubic bezier curves

	// Load FreeType library
	void load_font(const Vulkan::Context &ctx, const std::string &file) {
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

			// Get outline
			FT_UInt glyph_index = FT_Get_Char_Index(face, c);
			check_error(FT_Load_Glyph(face, glyph_index, FT_LOAD_DEFAULT));

			// Process outline
			FT_Outline_Funcs funcs = {
				.move_to = _outline_move_to,
				.line_to = _outline_line_to,
				.conic_to = _outline_conic_to,
				.cubic_to = nullptr
			};

			// Get bounding box
			FT_BBox bbox;
			FT_Outline_Get_BBox(&face->glyph->outline, &bbox);

			GlyphOutline outline = {ctx, glm::vec4 {
				bbox.xMin, bbox.yMin, bbox.xMax, bbox.yMax
			}};

			float width = bbox.xMax - bbox.xMin;
			float height = bbox.yMax - bbox.yMin;

			Logger::ok() << "Bounding box: "
				<< bbox.xMin << ", " << bbox.yMin << ", "
				<< bbox.xMax << ", " << bbox.yMax << std::endl;
			Logger::warn() << "\tWidth: " << width << std::endl;
			Logger::warn() << "\tHeight: " << height << std::endl;

			// Get number of curves
			FT_Outline_Decompose(&face->glyph->outline, &funcs, &outline);
			outline.upload();
			outline.dump();

			// Add to glyphs
			_glyphs[c] = outline;
		}
	}
public:
	Font() {}
	Font(const Vulkan::Context &ctx, const std::string &file) {
		// Check that the file exists
		if (!file_exists(file)) {
			Logger::error("Font file not found: " + file);
			throw -1;
		}

		// Load font
		load_font(ctx, file);
	}

	// Retrieve glyph
	const GlyphOutline &operator[](char c) const {
		auto it = _glyphs.find(c);
		if (it == _glyphs.end()) {
			Logger::error() << "Glyph not found: " << c << std::endl;
			throw -1;
		}

		return it->second;
	}
};

}

}

#endif

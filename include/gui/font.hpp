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
};

// Font class holds information
//	about a single font
class Font {
	// Font bitmaps
	std::unordered_map <char, raster::TexturePacket>	_bitmaps;

	// Font metrics
	std::unordered_map <char, FT_Glyph_Metrics>		_metrics;

	// Check FreeType error
	void check_error(FT_Error error) const {
		if (error) {
			Logger::error() << "FreeType error: "
				<< error << std::endl;
			throw -1;
		}
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
			_metrics[c] = face->glyph->metrics;
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

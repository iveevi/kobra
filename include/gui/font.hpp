#ifndef FONT_H_
#define FONT_H_

// Standard headers
#include <array>
#include <string>
#include <unordered_map>

// Vulkan headers
#include <vulkan/vulkan_core.h>

// FreeType headers
#include <ft2build.h>
#include FT_FREETYPE_H
#include FT_BBOX_H
#include FT_OUTLINE_H

// Engine headers
#include "../common.hpp"
#include "../logger.hpp"
#include "../texture.hpp"
#include "gui.hpp"

namespace kobra {

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
	Glyph(glm::vec4 bounds, glm::vec3 color)
			: _bounds(bounds), _color(color) {}

	// Getters
	inline glm::vec4 bounds() const { return _bounds; }

	// Move
	void move(const glm::vec2 &offset) {
		_bounds.x += offset.x;
		_bounds.y += offset.y;

		_bounds.z += offset.x;
		_bounds.w += offset.y;
	}

	// Color
	glm::vec3 &color() { return _color; }

	// Render the glyph
	// TODO: render method or upload method (instacing)?
	void upload(VertexBuffer &vb) const {
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
		VkDescriptorSet ds = VK_NULL_HANDLE;

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
	// TODO: remove, replaced by descritpor sets
	std::unordered_map <char, raster::TexturePacket>	_bitmaps;

	// Descriptor set for each glyph texture
	std::unordered_map <char, VkDescriptorSet>		_glyph_ds;

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
	void load_font(const Vulkan::Context &ctx,
			const VkCommandPool &cpool,
			const VkDescriptorPool &dpool,
			const std::string &file) {
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

		// Add space character
		{
			raster::TexturePacket tp = raster::make_texture(
				ctx,
				cpool,
				Texture {
					.width = 1024,
					.height = 1024
				},
				VK_FORMAT_R8_UNORM,
				VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
				VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
			);

			tp.transition_manual(ctx, cpool,
				VK_IMAGE_LAYOUT_UNDEFINED,
				VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
				VK_PIPELINE_STAGE_TRANSFER_BIT,
				VK_PIPELINE_STAGE_TRANSFER_BIT
			);

			raster::Sampler sampler(ctx, tp);

			VkDescriptorSet ds = Glyph::make_bitmap_ds(ctx, dpool);
			sampler.bind(ds, 0);

			_glyph_ds[' '] = ds;
			_metrics[' '] = FT_Glyph_Metrics {
				.horiBearingX = 0,
				.horiBearingY = 0,
				.horiAdvance = 32 * 1000,
				.vertBearingX = 0,
				.vertBearingY = 0,
				.vertAdvance = 0,
			};
		}

		for (char c = 0; c < 127; c++) {
			// Load bitmap into texture
			FT_Load_Char(face, c, FT_LOAD_RENDER);
			uint width = face->glyph->bitmap.width;
			uint height = face->glyph->bitmap.rows;
			Texture tex {
				.width = width,
				.height = height,
				.channels = 1
			};

			// Logger::warn() << "Glyph: " << width << " x " << height << std::endl;
			if (width * height == 0)
				continue;

			// Load texture
			tex.data = bytes(width * height);
			memcpy(tex.data.data(), face->glyph->bitmap.buffer, width * height);
			// Logger::warn() << "data size = " << tex.data.size() << std::endl;

			int nulls = 0;
			for (auto &b : tex.data) {
				if (b == 0)
					nulls++;
			}

			// Logger::warn() << "tex.data Nulls: " << nulls << std::endl;

			nulls = 0;
			for (int i = 0; i < tex.data.size(); i++) {
				if (face->glyph->bitmap.buffer[i] == 0)
					nulls++;
			}

			// Logger::warn() << "face->glyph->bitmap.buffer Nulls: " << nulls << std::endl;

			// Create texture
			raster::TexturePacket tp = raster::make_texture(
				ctx, cpool, tex,
				VK_FORMAT_R8_UNORM,
				VK_IMAGE_USAGE_TRANSFER_DST_BIT
					| VK_IMAGE_USAGE_SAMPLED_BIT,
				VK_IMAGE_LAYOUT_UNDEFINED
			);

			tp.transition_manual(ctx, cpool,
				VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
				VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
				VK_PIPELINE_STAGE_TRANSFER_BIT,
				VK_PIPELINE_STAGE_TRANSFER_BIT
			);

			// Add to dictionary
			_bitmaps[c] = tp;
			_metrics[c] = face->glyph->metrics;

			VkDescriptorSet ds = Glyph::make_bitmap_ds(ctx, dpool);
			raster::Sampler sampler = raster::Sampler(ctx, tp);
			sampler.bind(ds, 0);

			_glyph_ds[c] = ds;
		}
	}
public:
	Font() {}
	Font(const Vulkan::Context &ctx, const VkCommandPool &cpool, const VkDescriptorPool &dpool, const std::string &file) {
		// Check that the file exists
		if (!common::file_exists(file)) {
			Logger::error("Font file not found: " + file);
			throw -1;
		}

		// Load font
		load_font(ctx, cpool, dpool, file);
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

	const FT_Glyph_Metrics &metrics(char c) const {
		auto it = _metrics.find(c);
		if (it == _metrics.end()) {
			KOBRA_LOG_FUNC(error) << "Glyph metrics not found: \'"
				<< c << "\' (" << (int) c << ")\n" << std::endl;
			throw -1;
		}

		return it->second;
	}

	const VkDescriptorSet &glyph_ds(char c) const {
		auto it = _glyph_ds.find(c);
		if (it == _glyph_ds.end()) {
			Logger::error() << "Glyph descriptor set not found: "
				<< (int) c << "(\'" << c << "\')" << std::endl;
			throw -1;
		}

		return it->second;
	}
};

}

}

#endif

#ifndef FONT_H_
#define FONT_H_

// Standard headers
#include <array>
#include <string>
#include <map>

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
// #include "../sampler.hpp"
#include "gui.hpp"

namespace kobra {

namespace gui {

// Glyph structure
// TODO: separate header
// TODO: text class will hold shaders and stuff
class Glyph {
public:
	// Vertex and buffer type
	struct Vertex {
		glm::vec4 bounds;
		glm::vec3 color;

		// Get vertex binding description
		static vk::VertexInputBindingDescription
				vertex_binding() {
			return vk::VertexInputBindingDescription {
				0, sizeof(Vertex),
				vk::VertexInputRate::eVertex
			};
		}

		// Get vertex attribute descriptions
		static std::vector <vk::VertexInputAttributeDescription>
				vertex_attributes() {
			return {
				vk::VertexInputAttributeDescription {
					0, 0, vk::Format::eR32G32B32A32Sfloat, 0
				},

				vk::VertexInputAttributeDescription {
					1, 0, vk::Format::eR32G32B32A32Sfloat,
					sizeof(glm::vec4)
				}
			};
		}
	};

	// using VertexBuffer = BufferManager <Vertex>;
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
	void upload(std::vector <Vertex> &vb) const {
		std::vector <Vertex> vertices {
			Vertex {_bounds, _color},
			Vertex {_bounds, _color},
			Vertex {_bounds, _color},
			Vertex {_bounds, _color},
			Vertex {_bounds, _color},
			Vertex {_bounds, _color}
		};

		vb.insert(vb.end(), vertices.begin(), vertices.end());
	}

	/* Static buffer properties
	static constexpr BFM_Settings vb_settings {
		.size = 1024,
		.usage_type = BFM_WRITE_ONLY,
		.usage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
	};

	// TODO: remove
	static constexpr BFM_Settings ib_settings {
		.size = 1024,
		.usage_type = BFM_WRITE_ONLY,
		.usage = VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
	}; */

	// Descriptor set layout binding
	static constexpr DSLB bitmap_binding {
		0, vk::DescriptorType::eCombinedImageSampler,
		1, vk::ShaderStageFlagBits::eFragment
	};

	/* Make descriptor set layout
	[[deprecated("Manually create the descriptor set layout using bitmap_binding")]]
	static Vulkan::DSL make_bitmap_dsl(const Vulkan::Context &ctx) {
		return ctx.make_dsl({bitmap_binding});
	}

	// Make descriptor set
	// TODO: this is not a very nice method
	[[deprecated("Manually create the descriptor set using bitmap_binding")]]
	static VkDescriptorSet make_bitmap_ds(const Vulkan::Context &ctx, const VkDescriptorPool &pool) {
		return ctx.make_ds(pool, make_bitmap_dsl(ctx));
	} */
};

// Font class holds information
//	about a single font
class Font {
	// Font bitmap data
	std::vector <ImageData>	_bitmaps;
	std::vector <vk::raii::DescriptorSet> _glyph_ds;
	std::vector <vk::raii::Sampler>	_glyph_samplers;

	// Character to index map
	std::map <char, uint32_t> _char_to_index;

	// Font metrics
	std::unordered_map <char, FT_Glyph_Metrics>	_metrics;

	// Lint height
	float						_line_height;

	// Check FreeType error
	void check_error(FT_Error error) const {
		if (error) {
			Logger::error() << "FreeType error: "
				<< error << std::endl;
			throw -1;
		}
	}

	// Load FreeType library
	void load_font(const vk::raii::PhysicalDevice &phdev,
			const vk::raii::Device &device,
			const vk::raii::CommandPool &command_pool,
			const vk::raii::DescriptorPool &descriptor_pool,
			const std::string &file) {
		// Temporary command buffer
		vk::raii::CommandBuffer tmp_cmd = make_command_buffer(device, command_pool);

		// Queue to submit commands to
		vk::raii::Queue queue {device, 0, 0};

		// Load library
		FT_Library library;
		check_error(FT_Init_FreeType(&library));

		// Load font
		FT_Face face;
		check_error(FT_New_Face(library, file.c_str(), 0, &face));

		// Set font size
		check_error(FT_Set_Char_Size(face, 0, 1000 * 64, 96, 96));

		// Get line height
		_line_height = face->height / (64.0f * 1000.0f);

		// Process
		size_t total_points = 0;
		size_t total_cells = 0;

		// Create DSL for general Glyph
		vk::raii::DescriptorSetLayout general_dsl = make_descriptor_set_layout(
			device, {Glyph::bitmap_binding}
		);

		// Add space character
		{
			// Create the image data
			ImageData img = ImageData(
				phdev, device,
				vk::Format::eR8Unorm,
				vk::Extent2D {1, 1},
				vk::ImageTiling::eOptimal,
				vk::ImageUsageFlagBits::eSampled
					| vk::ImageUsageFlagBits::eTransferDst,
				vk::ImageLayout::ePreinitialized,
				vk::MemoryPropertyFlagBits::eDeviceLocal,
				vk::ImageAspectFlagBits::eColor
			);

			// Transition image layout
			// TODO: method using command_pool
			{
				tmp_cmd.begin({});

				transition_image_layout(tmp_cmd,
					*img.image, img.format,
					vk::ImageLayout::eUndefined,
					vk::ImageLayout::eShaderReadOnlyOptimal
				);

				tmp_cmd.end();

				queue.submit(
					vk::SubmitInfo {
						0, nullptr, nullptr, 1, &*tmp_cmd
					},
					nullptr
				);
			}

			// Create sampler handle
			vk::raii::Sampler sampler = make_sampler(device, img);

			// Create descriptor set and bind it
			// TODO: preallocate all needed descriptor sets, and
			// store those... (instead of indexing each time)
			vk::raii::DescriptorSets dsets = vk::raii::DescriptorSets {
				device, {*descriptor_pool, *general_dsl}
			};

			vk::raii::DescriptorSet dset = std::move(dsets.front());

			bind_ds(device, dset, sampler, img, 0);

			// Store everything
			_glyph_ds.emplace_back(std::move(dset));
			_glyph_samplers.emplace_back(std::move(sampler));
			_bitmaps.emplace_back(std::move(img));

			_metrics[' '] = FT_Glyph_Metrics {
				.horiBearingX = 0,
				.horiBearingY = 0,
				.horiAdvance = 32 * 1000,
				.vertBearingX = 0,
				.vertBearingY = 0,
				.vertAdvance = 0,
			};

			_char_to_index[' '] = 0;
		}

		for (char c = 0; c < 127; c++) {
			// Load bitmap data
			FT_Load_Char(face, c, FT_LOAD_RENDER);

			uint width = face->glyph->bitmap.width;
			uint height = face->glyph->bitmap.rows;

			if (width * height == 0)
				continue;

			// Create image data
			ImageData img = make_image(phdev, device,
				command_pool,
				width, height,
				face->glyph->bitmap.buffer,
				vk::Format::eR8Unorm,
				vk::ImageTiling::eOptimal,
				vk::ImageUsageFlagBits::eSampled
					| vk::ImageUsageFlagBits::eTransferDst,
				vk::MemoryPropertyFlagBits::eDeviceLocal,
				vk::ImageAspectFlagBits::eColor
			);

			// Create sampler handle
			vk::raii::Sampler sampler = make_sampler(device, img);

			// Create descriptor set and bind it
			vk::raii::DescriptorSets dsets = vk::raii::DescriptorSets {
				device, {*descriptor_pool, *general_dsl}
			};

			vk::raii::DescriptorSet dset = std::move(dsets.front());

			bind_ds(device, dset, sampler, img, 0);

			// Store everything
			_glyph_ds.emplace_back(std::move(dset));
			_glyph_samplers.emplace_back(std::move(sampler));
			_bitmaps.emplace_back(std::move(img));
			_metrics[c] = face->glyph->metrics;
			_char_to_index[c] = _glyph_ds.size() - 1;
		}
	}
public:
	Font() {}
	Font(const vk::raii::PhysicalDevice &phdev,
			const vk::raii::Device &device,
			const vk::raii::CommandPool &command_pool,
			const vk::raii::DescriptorPool &descriptor_pool,
			const std::string &file) {
		// Check that the file exists
		if (!common::file_exists(file)) {
			Logger::error("Font file not found: " + file);
			throw -1;
		}

		// Load font
		load_font(phdev, device, command_pool, descriptor_pool, file);
	}

	// Retrieve glyph bitmap
	const ImageData &bitmap(char c) const {
		uint32_t index = _char_to_index.at(c);
		return _bitmaps.at(index);
	}

	// Retrieve glyph metrics
	const FT_Glyph_Metrics &metrics(char c) const {
		auto it = _metrics.find(c);
		if (it == _metrics.end()) {
			KOBRA_LOG_FUNC(Log::ERROR) << "Glyph metrics not found: \'"
				<< c << "\' (" << (int) c << ")\n" << std::endl;
			throw -1;
		}

		return it->second;
	}

	// Retrieve glyph descriptor set
	const vk::raii::DescriptorSet &glyph_ds(char c) const {
		uint32_t index = _char_to_index.at(c);
		return _glyph_ds.at(index);
	}

	// Get line height
	float line_height() const {
		return _line_height;
	}
};

}

}

#endif

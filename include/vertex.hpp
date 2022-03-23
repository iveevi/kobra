#ifndef VERTEX_H_
#define VERTEX_H_

// Standard headers
#include <vector>

// GLM headers
#include <glm/glm.hpp>

// Engine headers
#include "backend.hpp"

namespace kobra {

struct Vertex {
	glm::vec3 position;
	glm::vec3 normal;
	glm::vec2 tex_coords;

	// Vertex binding
	static VertexBinding vertex_binding() {
		return VertexBinding {
			.binding = 0,
			.stride = sizeof(Vertex),
			.inputRate = VK_VERTEX_INPUT_RATE_VERTEX
		};
	}
	
	// Get vertex attribute descriptions
	static std::vector <VertexAttribute> vertex_attributes() {
		return {
			VertexAttribute {
				.location = 0,
				.binding = 0,
				.format = VK_FORMAT_R32G32B32_SFLOAT,
				.offset = offsetof(Vertex, position)
			},
			
			VertexAttribute {
				.location = 1,
				.binding = 0,
				.format = VK_FORMAT_R32G32B32_SFLOAT,
				.offset = offsetof(Vertex, normal)
			},

			VertexAttribute {
				.location = 2,
				.binding = 0,
				.format = VK_FORMAT_R32G32_SFLOAT,
				.offset = offsetof(Vertex, tex_coords)
			}
		};
	}

	// Create descriptor set layouts
	static constexpr VkDescriptorSetLayoutBinding vertex_dsl {
		.binding = 0,
		.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
		.descriptorCount = 1,
		.stageFlags = VK_SHADER_STAGE_VERTEX_BIT
	};

	static VkDescriptorSetLayout descriptor_set_layout(const Vulkan::Context &ctx) {
		static VkDescriptorSetLayout dsl = VK_NULL_HANDLE;

		if (dsl != VK_NULL_HANDLE)
			return dsl;

		// Create layout if not already created
		// TODO: context method
		dsl = ctx.vk->make_descriptor_set_layout(
			ctx.device,
			{vertex_dsl}
		);

		return dsl;
	}
};

// Aliases
using VertexList = std::vector <Vertex>;
using IndexList = std::vector <uint32_t>;

}

#endif

#ifndef VERTEX_H_
#define VERTEX_H_

// Standard headers
#include <vector>

// GLM headers
#include <glm/glm.hpp>

// Engine headers
#include "backend.hpp"

namespace kobra {

// Vertex types
using VertexType = uint32_t;

constexpr VertexType VERTEX_TYPE_POSITION	= 0;
constexpr VertexType VERTEX_TYPE_NORMAL		= 0x1;
constexpr VertexType VERTEX_TYPE_TEXCOORD	= 0x2;
constexpr VertexType VERTEX_TYPE_COLOR		= 0x4;
constexpr VertexType VERTEX_TYPE_TANGENT	= 0x8;

// Vertex information (templated by vertex type)
// TODO: header for each specialization (vertex directory)
template <VertexType T = VERTEX_TYPE_POSITION>
struct Vertex {
	// Vertex type
	static constexpr VertexType type = T;

	// Number of attributes
	static constexpr uint32_t attributes = 1;

	// Data
	glm::vec3 pos;

	// Vertex constructor
	Vertex() {}

	// Vertex constructor
	Vertex(const glm::vec3 &pos) : pos {pos} {}

	// Vertex binding
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
				.format = VK_FORMAT_R32G32B32_SFLOAT,
				.offset = offsetof(Vertex, pos)
			}
		};
	}

	// Create descriptor set layouts
	static constexpr VkDescriptorSetLayoutBinding _dsl {
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
			{_dsl}
		);

		return dsl;
	}

	// Create descriptor set
	static VkDescriptorSet descriptor_set(VkDescriptorPool pool) {
		return VK_NULL_HANDLE;
	}
};

// Default vertex type
template <>
struct Vertex <VERTEX_TYPE_POSITION> {
	// Vertex type
	static constexpr VertexType type = VERTEX_TYPE_POSITION;

	// Number of attributes
	static constexpr uint32_t attributes = 1;

	// Data
	glm::vec3 pos;

	// Vertex constructor
	Vertex() {}

	// Vertex constructor
	Vertex(const glm::vec3 &pos) : pos {pos} {}

	// Vertex binding
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
				.format = VK_FORMAT_R32G32B32_SFLOAT,
				.offset = offsetof(Vertex, pos)
			}
		};
	}
	
	// Create descriptor set layouts
	static constexpr VkDescriptorSetLayoutBinding _dsl {
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
			{_dsl}
		);

		return dsl;
	}

	// Create descriptor set
	static VkDescriptorSet descriptor_set(VkDescriptorPool pool) {
		return VK_NULL_HANDLE;
	}
};

// Vertex with normal
template <>
struct Vertex <VERTEX_TYPE_NORMAL> {
	// Vertex type
	static constexpr VertexType type = VERTEX_TYPE_NORMAL;

	// Number of attributes
	static constexpr uint32_t attributes = 2;

	// Data
	glm::vec3 pos;
	glm::vec3 normal;

	// Vertex constructor
	Vertex() {}

	// Vertex constructor
	Vertex(const glm::vec3 &pos, const glm::vec3 &normal) : pos {pos}, normal {normal} {}

	// Vertex binding
	static VertexBinding vertex_binding() {
		return VertexBinding {
			.binding = 0,
			.stride = sizeof(Vertex),
			.inputRate = VK_VERTEX_INPUT_RATE_VERTEX
		};
	}

	// Get vertex attribute descriptions
	static std::array <VertexAttribute, 2> vertex_attributes() {
		return std::array <VertexAttribute, 2> {
			VertexAttribute {
				.location = 0,
				.binding = 0,
				.format = VK_FORMAT_R32G32B32_SFLOAT,
				.offset = offsetof(Vertex, pos)
			},
			VertexAttribute {
				.location = 1,
				.binding = 0,
				.format = VK_FORMAT_R32G32B32_SFLOAT,
				.offset = offsetof(Vertex, normal)
			}
		};
	}

	// Create descriptor set layout
	static VkDescriptorSetLayout descriptor_set_layout() {
		return VK_NULL_HANDLE;
	}

	// Create descriptor set
	static VkDescriptorSet descriptor_set(VkDescriptorPool pool) {
		return VK_NULL_HANDLE;
	}
};

// Aliases
template <VertexType T>
using VertexList = std::vector <Vertex <T>>;
using IndexList = std::vector <uint32_t>;

}

#endif

// Standard headers
#include <iostream>

#define KOBRA_VALIDATION_LAYERS
// #define KOBRA_THROW_ERROR

// Engine headers
#include "../include/backend.hpp"
#include "../include/mesh.hpp"
#include "../include/camera.hpp"
#include "../shaders/raster/bindings.h"

// More Vulkan stuff
#include <vulkan/vulkan_core.h>
#include <vulkan/vulkan_raii.hpp>
#include <vulkan/vulkan_structs.hpp>

using namespace kobra;

// Camera
Camera camera = Camera {
	Transform {
		{0, 0, 5},
		{0, 0, 0}
	},

	Tunings { 45.0f, 800, 800 }
};

// MVP structure
struct PC_Material {
	glm::vec3	albedo;
	int		type;
	float		hightlight;
	float		has_albedo;
	float		has_normal;
};

struct PC_Vertex {
	glm::mat4 model;
	glm::mat4 view;
	glm::mat4 projection;

	PC_Material material;
};

// Mouse movement
void mouse_movement(GLFWwindow *, double xpos, double ypos)
{
	static const float sensitivity = 0.0005f;

	static bool first_movement = true;

	static float px = 0.0f;
	static float py = 0.0f;

	static float yaw = 0.0f;
	static float pitch = 0.0f;

	// Deltas and directions
	float dx = xpos - px;
	float dy = ypos - py;

	yaw -= dx * sensitivity;
	pitch -= dy * sensitivity;

	if (pitch > 89.0f)
		pitch = 89.0f;
	if (pitch < -89.0f)
		pitch = -89.0f;

	camera.transform.rotation.x = pitch;
	camera.transform.rotation.y = yaw;

	px = xpos;
	py = ypos;
}

// Key callback
void keyboard_input(GLFWwindow *window, int key, int, int action, int)
{
	if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS)
		glfwSetWindowShouldClose(window, GLFW_TRUE);
}

// Input handling
void input_handling(GLFWwindow *window)
{
	float speed = 0.01f;

	glm::vec3 forward = camera.transform.forward();
	glm::vec3 right = camera.transform.right();
	glm::vec3 up = camera.transform.up();

	if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS)
		camera.transform.move(forward * speed);
	else if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS)
		camera.transform.move(-forward * speed);

	if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS)
		camera.transform.move(-right * speed);
	else if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS)
		camera.transform.move(right * speed);

	if (glfwGetKey(window, GLFW_KEY_E) == GLFW_PRESS)
		camera.transform.move(up * speed);
	else if (glfwGetKey(window, GLFW_KEY_Q) == GLFW_PRESS)
		camera.transform.move(-up * speed);
}

// Present an image
void present_swapchain_image(const Swapchain &swapchain, uint32_t &image_index,
		const vk::raii::Queue queue,
		const vk::raii::Semaphore &signal_semaphore)
{
	vk::PresentInfoKHR present_info {
		*signal_semaphore,
		*swapchain.swapchain,
		image_index
	};

	vk::Result result = queue.presentKHR(present_info);

	KOBRA_ASSERT(
		result == vk::Result::eSuccess,
		"Failed to present image (result = " + vk::to_string(result) + ")"
	);
}

// Per frame data
struct FrameData {
	vk::raii::Fence		fence = nullptr;
	vk::raii::Semaphore	present_completed = nullptr;
	vk::raii::Semaphore	render_completed = nullptr;

	// Default constructor
	FrameData() = default;

	// Construct from device
	FrameData(const vk::raii::Device &device) :
		fence(device, vk::FenceCreateInfo {vk::FenceCreateFlagBits::eSignaled}),
		present_completed(device, vk::SemaphoreCreateInfo {}),
		render_completed(device, vk::SemaphoreCreateInfo {}) {}
};

// Vulkan representation of a model
struct VkModel {
	BufferData	vertex_buffer;
	BufferData	index_buffer;

	uint32_t	vertex_count;
	uint32_t	index_count;
};

// Create a VkModel from a Mesh
VkModel make_vk_model(const vk::raii::PhysicalDevice &phdev, const vk::raii::Device &device, const Mesh &mesh)
{
	BufferData vertex_buffer = BufferData(phdev, device,
		mesh.vertices().size() * sizeof(Vertex),
		vk::BufferUsageFlagBits::eVertexBuffer
			| vk::BufferUsageFlagBits::eShaderDeviceAddress,
		vk::MemoryPropertyFlagBits::eHostVisible
			| vk::MemoryPropertyFlagBits::eHostCoherent
	);

	BufferData index_buffer = BufferData(phdev, device,
		mesh.indices().size() * sizeof(uint32_t),
		vk::BufferUsageFlagBits::eIndexBuffer
			| vk::BufferUsageFlagBits::eShaderDeviceAddress,
		vk::MemoryPropertyFlagBits::eHostVisible
			| vk::MemoryPropertyFlagBits::eHostCoherent
	);

	vertex_buffer.upload(mesh.vertices());
	index_buffer.upload(mesh.indices());

	return {
		std::move(vertex_buffer),
		std::move(index_buffer),
		static_cast <uint32_t> (mesh.vertices().size()),
		static_cast <uint32_t> (mesh.indices().size())
	};
}

namespace as {	// Acceleration structure functions

// KHR acceleration structure aliases
using Geometry = vk::AccelerationStructureGeometryKHR;
using Flags = vk::BuildAccelerationStructureFlagsKHR;
using FlagBits = vk::BuildAccelerationStructureFlagBitsKHR;

using BuildInfo = vk::AccelerationStructureBuildGeometryInfoKHR;
using BuildSize = vk::AccelerationStructureBuildSizesInfoKHR;
using BuildRange = vk::AccelerationStructureBuildRangeInfoKHR;

// Bottom level acceleration structure (BLAS) info
struct BLAS {
	std::vector <Geometry>		geometries;
	std::vector <BuildRange>	build_offsets;
	Flags				flags {0};
};

// Create BLAS info
BLAS make_blas(const vk::raii::Device &device, const VkModel &model)
{
	// Buffer addresses
	vk::DeviceAddress vertex_buffer_address = buffer_addr(device, model.vertex_buffer);
	vk::DeviceAddress index_buffer_address = buffer_addr(device, model.index_buffer);

	// Maximum number of primitives
	uint32_t max_primitives = model.index_count/3;

	// Vertex buffer description
	vk::AccelerationStructureGeometryTrianglesDataKHR triangles_data;
	triangles_data.vertexFormat = vk::Format::eR32G32B32Sfloat;
	triangles_data.vertexData.deviceAddress = vertex_buffer_address;
	triangles_data.vertexStride = sizeof(Vertex);
	triangles_data.maxVertex = model.vertex_count;
	triangles_data.indexType = vk::IndexType::eUint32;
	triangles_data.indexData.deviceAddress = index_buffer_address;

	// Geometry description
	vk::AccelerationStructureGeometryKHR geometry;
	geometry.geometryType = vk::GeometryTypeKHR::eTriangles;
	geometry.geometry.triangles = triangles_data;
	geometry.flags = vk::GeometryFlagBitsKHR::eOpaque;

	// Build range
	vk::AccelerationStructureBuildRangeInfoKHR build_range;
	build_range.primitiveCount = max_primitives;
	build_range.primitiveOffset = 0;
	build_range.firstVertex = 0;
	build_range.transformOffset = 0;

	return {
		{geometry},
		{build_range}
	};
}

// Acceleration structure wrapper
struct Accelerator {
	vk::raii::AccelerationStructureKHR	as = nullptr;
	BufferData				scratch = nullptr;

	// Default constructor
	Accelerator() = default;

	// Construct from device
	Accelerator(const vk::raii::PhysicalDevice &phdev,
			const vk::raii::Device &device,
			const vk::AccelerationStructureCreateInfoKHR &info)
		: as(device, info),
		scratch(phdev, device,
			info.size,
			vk::BufferUsageFlagBits::eStorageBuffer
				| vk::BufferUsageFlagBits::eShaderDeviceAddress,
			vk::MemoryPropertyFlagBits::eDeviceLocal
		) {}
};

// Final acceleration structure (AS) info
struct AccelerationStructure {
	Accelerator			tlas;
	std::vector <Accelerator>	blas;
};

// Record a BLAS build command buffer
void cmd_build_blas(const vk::raii::PhysicalDevice &phdev,
		const vk::raii::Device &device,
		const vk::raii::CommandBuffer &cmd,
		std::vector <Accelerator> &blas_out,
		const std::vector <uint32_t> &blas_indices,
		std::vector <BuildInfo> &build_infos,
		const std::vector <BuildSize> &build_sizes,
		const std::vector <const BuildRange *> &build_ranges,
		const vk::DeviceAddress &scratch_addr,
		const vk::raii::QueryPool &query_pool)
{
	/* if (query_pool != nullptr) {
		cmd.resetQueryPool(*query_pool,
			0, static_cast <uint32_t> (blas_indices.size())
		);
	} */

	// Query count
	uint32_t query_count;

	// Iterate through BLAS indices
	for (const auto &i : blas_indices) {
		// Creation info
		vk::AccelerationStructureCreateInfoKHR create_info;
		create_info.type = vk::AccelerationStructureTypeKHR::eBottomLevel;
		create_info.size = build_sizes[i].accelerationStructureSize;

		// Create acceleration structure
		blas_out[i] = std::move(Accelerator(phdev, device, create_info));

		// Build command buffer
		build_infos[i].dstAccelerationStructure = *blas_out[i].as;
		build_infos[i].scratchData.deviceAddress = scratch_addr;

		cmd.buildAccelerationStructuresKHR(build_infos[i], build_ranges[i]);

		// Memory barrier since we are batching
		vk::BufferMemoryBarrier memory_barrier;
		memory_barrier.srcAccessMask = vk::AccessFlagBits::eAccelerationStructureWriteKHR;
		memory_barrier.dstAccessMask = vk::AccessFlagBits::eAccelerationStructureReadKHR;

		cmd.pipelineBarrier(vk::PipelineStageFlagBits::eAccelerationStructureBuildKHR,
			vk::PipelineStageFlagBits::eAccelerationStructureBuildKHR,
			vk::DependencyFlagBits::eByRegion,
			{},
			{memory_barrier},
			{}
		);

		/* Query
		if (query_pool != nullptr) {
			cmd.writeAccelerationStructurePropertiesKHR(blas_out[i].as,
				vk::QueryTypeKHR::eAccelerationStructureKHR,
				0, 1,
				query_pool,
				query_count
			);
		} */
	}
}

// Record a TLAS build command buffer
void cmd_build_tlas(const vk::raii::PhysicalDevice &phdev,
		const vk::raii::Device &device,
		const vk::raii::CommandBuffer &cmd,
		Accelerator &tlas_out,
		uint32_t instance_count,
		const vk::DeviceAddress &instance_buffer_addr,
		BufferData &scratchBuffer,
		VkBuildAccelerationStructureFlagsKHR flags)
		// Assuming no update or motion
{
	bool update = false;

	// Wraps a device pointer to the above uploaded instances.
	VkAccelerationStructureGeometryInstancesDataKHR instance_data {
		VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_INSTANCES_DATA_KHR
	};

	instance_data.data.deviceAddress = instance_buffer_addr;

	// Put the above into a VkAccelerationStructureGeometryKHR. We need to put the instances struct in a union and label it as instance data.
	VkAccelerationStructureGeometryKHR topASGeometry{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR};
	topASGeometry.geometryType       = VK_GEOMETRY_TYPE_INSTANCES_KHR;
	topASGeometry.geometry.instances = instance_data;

	// Find sizes
	VkAccelerationStructureBuildGeometryInfoKHR buildInfo{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR};
	buildInfo.flags         = flags;
	buildInfo.geometryCount = 1;
	buildInfo.pGeometries   = &topASGeometry;
	buildInfo.mode = update ? VK_BUILD_ACCELERATION_STRUCTURE_MODE_UPDATE_KHR : VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
	buildInfo.type                     = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;
	buildInfo.srcAccelerationStructure = VK_NULL_HANDLE;

	VkAccelerationStructureBuildSizesInfoKHR sizeInfo{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR};
	vkGetAccelerationStructureBuildSizesKHR(*device,
			VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR, &buildInfo,
			&instance_count, &sizeInfo
	);

	// Create TLAS
	if(update == false) {
		VkAccelerationStructureCreateInfoKHR createInfo{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR};
		createInfo.type = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;
		createInfo.size = sizeInfo.accelerationStructureSize;

		tlas_out = std::move(Accelerator(phdev, device, createInfo));
	}

	// Allocate the scratch memory
	// scratchBuffer = m_alloc->createBuffer(sizeInfo.buildScratchSize,
	//		VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT);
	scratchBuffer = BufferData(phdev, device,
			sizeInfo.buildScratchSize,
			vk::BufferUsageFlagBits::eStorageBuffer
				| vk::BufferUsageFlagBits::eShaderDeviceAddress,
			vk::MemoryPropertyFlagBits::eDeviceLocal
	);

	VkBufferDeviceAddressInfo bufferInfo {
		VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO,
		nullptr,
		*scratchBuffer.buffer
	};

	VkDeviceAddress           scratchAddress = vkGetBufferDeviceAddress(*device, &bufferInfo);

	// Update build information
	buildInfo.srcAccelerationStructure  = update ? *(tlas_out.as) : VK_NULL_HANDLE;
	buildInfo.dstAccelerationStructure  = *(tlas_out.as);
	buildInfo.scratchData.deviceAddress = scratchAddress;

	// Build Offsets info: n instances
	VkAccelerationStructureBuildRangeInfoKHR        buildOffsetInfo {instance_count, 0, 0, 0};
	const VkAccelerationStructureBuildRangeInfoKHR* pBuildOffsetInfo = &buildOffsetInfo;

	// Build the TLAS
	// auto build_range = BuildRange(pBuildOffsetInfo, 1);
	// cmd.buildAccelerationStructuresKHR({buildInfo}, {build_range});
	
	vkCmdBuildAccelerationStructuresKHR(*cmd, 1, &buildInfo, &pBuildOffsetInfo);
}

// Build AS
void build_as(const vk::raii::PhysicalDevice &phdev,
		const vk::raii::Device &device,
		const vk::raii::CommandPool &command_pool,
		const std::vector <BLAS> &input,
		const Flags &flags)
{
	// Size infos
	uint32_t blas_count = static_cast <uint32_t> (input.size());

	vk::DeviceSize blas_size = 0;
	vk::DeviceSize blas_scratch_size = 0;
	uint32_t blas_compactions = 0;

	// Population Vulkun build info structure
	std::vector <BuildInfo> build_infos(blas_count);
	std::vector <BuildSize> build_sizes(blas_count);
	std::vector <const BuildRange *> build_offsets(blas_count);

	for (uint32_t i = 0; i < blas_count; i++) {
		// Fill in build info
		build_infos[i].flags = flags | input[i].flags;
		build_infos[i].type = vk::AccelerationStructureTypeKHR::eBottomLevel;
		build_infos[i].mode = vk::BuildAccelerationStructureModeKHR::eBuild;
		build_infos[i].geometryCount = static_cast <uint32_t> (input[i].geometries.size());
		build_infos[i].pGeometries = input[i].geometries.data();

		// Range info
		build_offsets[i] = input[i].build_offsets.data();

		// Determine sizes
		std::vector <uint32_t> max_prims(input[i].build_offsets.size());
		for (uint32_t j = 0; j < input[i].build_offsets.size(); j++)
			max_prims[j] = input[i].build_offsets[j].primitiveCount;

		// Query sizes
		build_sizes[i] = device.getAccelerationStructureBuildSizesKHR(
			vk::AccelerationStructureBuildTypeKHR::eDevice,
			build_infos[i],
			max_prims
		);

		// Accumulate sizes
		blas_size += build_sizes[i].accelerationStructureSize;
		blas_scratch_size = std::max(blas_scratch_size, build_sizes[i].buildScratchSize);
		blas_compactions += (build_infos[i].flags
			& vk::BuildAccelerationStructureFlagBitsKHR::eAllowCompaction) ? 1 : 0;
	}

	// Create scratch buffer
	BufferData blas_scratch_buffer(phdev, device, blas_scratch_size,
		vk::BufferUsageFlagBits::eStorageBuffer
			| vk::BufferUsageFlagBits::eShaderDeviceAddress,
		vk::MemoryPropertyFlagBits::eDeviceLocal
	);

	vk::DeviceAddress blas_scratch_address = buffer_addr(device, blas_scratch_buffer);

	// Query real sizes
	vk::raii::QueryPool blas_query_pool = nullptr;
	if (blas_compactions > 0) {
		blas_query_pool = vk::raii::QueryPool(device,
			vk::QueryPoolCreateInfo {
				vk::QueryPoolCreateFlags(),
				vk::QueryType::eAccelerationStructureCompactedSizeKHR,
				blas_compactions
			}
		);
	}

	// Batching build
	std::vector <uint32_t> blas_indices;
	vk::DeviceSize batch_size = 0;
	vk::DeviceSize batch_limit = 256'000'000;	// 256 MB

	// Final BLAS
	std::vector <Accelerator> blas_out;

	for (uint32_t i = 0; i < blas_count; i++) {
		blas_indices.push_back(i);

		// Update size coun
		batch_size += build_sizes[i].accelerationStructureSize;

		// Last batch or batch limit reached
		if (batch_size >= batch_limit || i == blas_count - 1) {
			vk::raii::CommandBuffer cmd = make_command_buffer(device, command_pool);

			// Create acceleration structure
			cmd.begin(vk::CommandBufferBeginInfo {
				vk::CommandBufferUsageFlagBits::eOneTimeSubmit
			});

			cmd_build_blas(phdev, device, cmd,
				blas_out,
				blas_indices,
				build_infos,
				build_sizes,
				build_offsets,
				blas_scratch_address,
				blas_query_pool
			);

			cmd.end();
		}
	}

	// Outline TLAS
	std::vector <vk::AccelerationStructureInstanceKHR> tlas;

	for (uint32_t i = 0; i < blas_count; i++) {
		vk::AccelerationStructureInstanceKHR inst;
		inst.transform = std::array <std::array <float, 4>, 3> {
			std::array <float, 4> {1.0f, 0.0f, 0.0f, 0.0f},
			std::array <float, 4> {0.0f, 1.0f, 0.0f, 0.0f},
			std::array <float, 4> {0.0f, 0.0f, 1.0f, 0.0f}
		};

		inst.instanceCustomIndex = i;
		inst.accelerationStructureReference = acceleration_structure_addr(device, blas_out[i].as);
		inst.flags = (uint32_t) vk::GeometryInstanceFlagBitsKHR::eTriangleCullDisable;
		inst.mask = 0xFF;
		inst.instanceShaderBindingTableRecordOffset = 0;

		tlas.push_back(inst);
	}

	// TODO: another function
	// Create TLAS
	
	uint32_t tlas_count = static_cast <uint32_t> (tlas.size());

	// Command buffer
	vk::raii::CommandBuffer cmd = make_command_buffer(device, command_pool);

	// Buffer for instance data
	BufferData tlas_buffer(phdev, device,
		tlas_count * sizeof(vk::AccelerationStructureInstanceKHR),
		vk::BufferUsageFlagBits::eStorageBuffer
			| vk::BufferUsageFlagBits::eShaderDeviceAddress,
		vk::MemoryPropertyFlagBits::eDeviceLocal
	);

	vk::DeviceAddress tlas_address = buffer_addr(device, tlas_buffer);

	// Copy instance data
	cmd.begin(vk::CommandBufferBeginInfo {
		vk::CommandBufferUsageFlagBits::eOneTimeSubmit
	});

	cmd.end();
}

}		// Acceleration structure functions

namespace rt {

// Rya tracing push constants
struct PC_Ray {
	glm::vec4	clear_color;
	glm::vec3	light_pos;
	float		light_intensity;
};

// Build the raytracing pipeline
void build_pipeline(const vk::raii::Device &device)
{
	// Stage indices
	enum StageIndices {
		eRayGeneration = 0,
		eMiss,
		eClosestHit,
		eShaderGroupCount
	};

	// Load all the shader modules
	auto shaders = make_shader_modules(device, {
		"experimental/bin/raytrace.rgen.spv",
		"experimental/bin/raytrace.rmiss.spv",
		"experimental/bin/raytrace.rchit.spv"
	});

	// Array of shader stages
	std::array <vk::PipelineShaderStageCreateInfo, eShaderGroupCount> stages;
	vk::PipelineShaderStageCreateInfo stage;

	// Ray generation
	stage.stage = vk::ShaderStageFlagBits::eRaygenKHR;
	stage.module = *shaders[eRayGeneration];
	stage.pName = "main";

	stages[eRayGeneration] = stage;

	// Miss
	stage.stage = vk::ShaderStageFlagBits::eMissKHR;
	stage.module = *shaders[eMiss];
	stage.pName = "main";

	stages[eMiss] = stage;

	// Closest hit
	stage.stage = vk::ShaderStageFlagBits::eClosestHitKHR;
	stage.module = *shaders[eClosestHit];
	stage.pName = "main";

	stages[eClosestHit] = stage;

	// Shader group info
	std::vector <vk::RayTracingShaderGroupCreateInfoKHR> shader_groups;

	vk::RayTracingShaderGroupCreateInfoKHR shader_group;

	shader_group.anyHitShader = VK_SHADER_UNUSED_KHR;
	shader_group.closestHitShader = VK_SHADER_UNUSED_KHR;
	shader_group.generalShader = VK_SHADER_UNUSED_KHR;
	shader_group.intersectionShader = VK_SHADER_UNUSED_KHR;

	// Ray generation
	shader_group.type = vk::RayTracingShaderGroupTypeKHR::eGeneral;
	shader_group.generalShader = eRayGeneration;

	shader_groups.push_back(shader_group);

	// Miss
	shader_group.type = vk::RayTracingShaderGroupTypeKHR::eGeneral;
	shader_group.generalShader = eMiss;

	shader_groups.push_back(shader_group);

	// Closest hit
	shader_group.type = vk::RayTracingShaderGroupTypeKHR::eTrianglesHitGroup;
	shader_group.generalShader = VK_SHADER_UNUSED_KHR;
	shader_group.closestHitShader = eClosestHit;

	shader_groups.push_back(shader_group);

	// Push constants
	vk::PushConstantRange push_constants;
	push_constants.offset = 0;
	push_constants.size = sizeof(PC_Ray);
	push_constants.stageFlags = vk::ShaderStageFlagBits::eRaygenKHR
		| vk::ShaderStageFlagBits::eMissKHR
		| vk::ShaderStageFlagBits::eClosestHitKHR;

	// Ray tracing pipeline layout
	vk::PipelineLayoutCreateInfo ppl_info;

	ppl_info.pushConstantRangeCount = 1;
	ppl_info.pPushConstantRanges = &push_constants;
}

}

int main()
{
	// Choosing physical device
	auto window = Window("Vulkan RT", {1000, 1000});

	// GLFW callbacks
	// TODO: window methods
	glfwSetCursorPosCallback(window.handle, mouse_movement);
	glfwSetKeyCallback(window.handle, keyboard_input);

	// Disable cursor
	glfwSetInputMode(window.handle, GLFW_CURSOR, GLFW_CURSOR_DISABLED);

	// Create a surface
	vk::raii::SurfaceKHR surface = make_surface(window);

	auto extensions = {
		VK_KHR_SWAPCHAIN_EXTENSION_NAME,

		// For raytracing
		"VK_KHR_spirv_1_4",
		"VK_KHR_shader_float_controls",
		"VK_KHR_ray_tracing_pipeline",
		"VK_KHR_acceleration_structure",
		"VK_EXT_descriptor_indexing",
		"VK_KHR_maintenance3",
		"VK_KHR_buffer_device_address",
		"VK_KHR_deferred_host_operations",
		"VK_KHR_buffer_device_address",
		// "VK_KHR_ray_query"
	};

	auto predicate = [&extensions](const vk::raii::PhysicalDevice &dev) {
		return physical_device_able(dev, extensions);
	};

	auto phdev = pick_physical_device(predicate);

	std::cout << "Chosen device: " << phdev.getProperties().deviceName << std::endl;
	std::cout << "\tgraphics queue family: " << find_graphics_queue_family(phdev) << std::endl;
	std::cout << "\tpresent queue family: " << find_present_queue_family(phdev, surface) << std::endl;

	// Verification and creating a logical device
	auto queue_family = find_queue_families(phdev, surface);
	std::cout << "\tqueue family (G): " << queue_family.graphics << std::endl;
	std::cout << "\tqueue family (P): " << queue_family.present << std::endl;

	auto queue_families = phdev.getQueueFamilyProperties();

	std::cout <<"Queue families:" << std::endl;
	for (uint32_t i = 0; i < queue_families.size(); i++) {
		auto flags = queue_families[i].queueFlags;
		std::cout << "\tqueue family " << i << ": ";
		if (flags & vk::QueueFlagBits::eGraphics)
			std::cout << "Graphics ";
		if (flags & vk::QueueFlagBits::eCompute)
			std::cout << "Compute ";
		if (flags & vk::QueueFlagBits::eTransfer)
			std::cout << "Transfer ";
		if (flags & vk::QueueFlagBits::eSparseBinding)
			std::cout << "SparseBinding ";

		// Count
		std::cout <<" [" << queue_families[i].queueCount << "]\n";
	}

	auto device = make_device(phdev, queue_family, extensions);

	// Load Vulkan extensions
	load_vulkan_extensions(device);

	KOBRA_LOG_FILE(ok) << "Vulkan extensions loaded\n";

	// Command pool and buffer
	auto command_pool = vk::raii::CommandPool {
		device,
		{
			vk::CommandPoolCreateFlagBits::eResetCommandBuffer,
			queue_family.graphics
		}
	};

	// TODO: overload function for this
	vk::raii::CommandBuffer command_buffers[2] {
		make_command_buffer(device, command_pool),
		make_command_buffer(device, command_pool)
	};

	// Queues
	auto graphics_queue = vk::raii::Queue { device, queue_family.graphics, 0 };
	auto present_queue = vk::raii::Queue { device, queue_family.present, 0 };

	// Swapchain
	auto swapchain = Swapchain {phdev, device, surface, window.extent, queue_family};

	// Transition swapchain images to presentable
	{
		// Temporary command buffer
		auto temp_command_buffer = make_command_buffer(device, command_pool);

		// Record
		temp_command_buffer.begin(vk::CommandBufferBeginInfo {
			vk::CommandBufferUsageFlagBits::eOneTimeSubmit
		});

		// Transition
		for (auto &img : swapchain.images) {
			transition_image_layout(temp_command_buffer,
				img,
				vk::Format::eB8G8R8A8Unorm,
				vk::ImageLayout::eUndefined,
				vk::ImageLayout::ePresentSrcKHR
			);
		}

		// End
		temp_command_buffer.end();

		// Submit
		graphics_queue.submit(
			vk::SubmitInfo {
				0, nullptr, nullptr, 1, &*temp_command_buffer
			},
			nullptr
		);

		// Wait
		graphics_queue.waitIdle();
	}

	// Depth buffer and render pass
	auto depth_buffer = DepthBuffer {phdev, device, vk::Format::eD32Sfloat, window.extent};
	auto render_pass = make_render_pass(device, swapchain.format, depth_buffer.format);

	auto framebuffers = make_framebuffers(
		device, render_pass,
		swapchain.image_views,
		&depth_buffer.view,
		window.extent
	);

	// Box mesh
	auto box_mesh = Mesh::make_box({0, 0, 0}, {1, 1, 1});

	// Vulkan model
	auto box_vk = make_vk_model(phdev, device, box_mesh);

	/* Acceleration structure things
	auto blas = as::make_blas(device, box_vk);
	as::build_as(phdev, device, command_pool, {blas}, as::FlagBits::eAllowUpdate); */

	// Load shaders
	// TODO: compile function for shaders
	auto vertex = make_shader_module(device, "shaders/bin/raster/vertex.spv");
	auto fragment = make_shader_module(device, "shaders/bin/raster/color_frag.spv");

	// Descriptor set layout
	auto dsl = make_descriptor_set_layout(device, {
		{
			0,
			vk::DescriptorType::eCombinedImageSampler,
			1,
			vk::ShaderStageFlagBits::eFragment
		},
	});

	// Push constant (glm::mat4 MVP)
	auto pcl = vk::PushConstantRange {
		vk::ShaderStageFlagBits::eVertex,
		0, sizeof(PC_Vertex)
	};

	auto ppl = vk::raii::PipelineLayout {device, {{}, *dsl, pcl}};
	auto ppl_cache = vk::raii::PipelineCache {device, vk::PipelineCacheCreateInfo()};

	// Vertex input binding and attributes
	auto vertex_input_binding = vk::VertexInputBindingDescription {
		0, sizeof(Vertex),
		vk::VertexInputRate::eVertex
	};

	auto vertex_input_attributes = Vertex::vertex_attributes();

	std::cout << "Vertex input attributes:" << vertex_input_attributes.size() << std::endl;
	for (auto &attr : vertex_input_attributes) {
		std::cout << "\t" << attr.binding << " " << attr.location << " "
			<< vk::to_string(attr.format) << " " << attr.offset << std::endl;
	}

	// Create the graphics pipeline
	auto grp_info = GraphicsPipelineInfo {
		.device = device,
		.render_pass = render_pass,

		.vertex_shader = std::move(vertex),
		.fragment_shader = std::move(fragment),

		.vertex_binding = vertex_input_binding,
		.vertex_attributes = vertex_input_attributes,

		.pipeline_layout = ppl,
		.pipeline_cache = ppl_cache,

		.depth_test = true,
		.depth_write = true,
	};

	auto pipeline = make_graphics_pipeline(grp_info);

	// Descriptor pool and sets
	auto descriptor_pool = make_descriptor_pool(
		device,
		{
			{vk::DescriptorType::eCombinedImageSampler, 1}
		}
	);

	auto descriptor_sets = vk::raii::DescriptorSets {device, {*descriptor_pool, *dsl}};
	auto dset = std::move(descriptor_sets.front());

	// Other variables
	vk::Result result;
	uint32_t image_index;

	// Load image texture
	ImageData img = nullptr;

	KOBRA_LOG_FILE(notify) << "Starting image creation process\n";

	{
		// Temporary command buffer
		// TODO: simpler method for single time use (lambda into
		// function...)
		auto temp_command_buffer = make_command_buffer(device, command_pool);

		// Record
		temp_command_buffer.begin(vk::CommandBufferBeginInfo {
			vk::CommandBufferUsageFlagBits::eOneTimeSubmit
		});

		// Staging buffer
		BufferData staging_buffer = nullptr;

		// Create the texture
		img = std::move(make_image(temp_command_buffer,
			phdev, device, staging_buffer,
			"resources/brickwall.jpg",
			vk::ImageTiling::eOptimal,
			vk::ImageUsageFlagBits::eSampled
				| vk::ImageUsageFlagBits::eTransferDst
				| vk::ImageUsageFlagBits::eTransferSrc,
			vk::MemoryPropertyFlagBits::eDeviceLocal,
			vk::ImageAspectFlagBits::eColor
		));

		// End
		temp_command_buffer.end();

		// Submit
		graphics_queue.submit(
			vk::SubmitInfo {
				0, nullptr, nullptr, 1, &*temp_command_buffer
			},
			nullptr
		);

		// Wait
		graphics_queue.waitIdle();
	}

	KOBRA_LOG_FILE(notify) << "Image loaded\n";

	auto sampler = make_sampler(device, img);

	// Bind sampler to descriptor set
	auto dset_info = std::array <vk::DescriptorImageInfo, 1> {
		vk::DescriptorImageInfo {
			*sampler,
			*img.view,
			vk::ImageLayout::eShaderReadOnlyOptimal
		}
	};

	vk::WriteDescriptorSet dset_write {
		*dset,
		0, 0,
		vk::DescriptorType::eCombinedImageSampler,
		dset_info
	};

	device.updateDescriptorSets(dset_write, nullptr);

	// Record the command buffer
	auto record = [&](const vk::raii::CommandBuffer &command_buffer) {
		command_buffer.begin({});

		// Set the viewport and scissor
		auto viewport = vk::Viewport {
			0.0f, 0.0f,
			static_cast <float> (window.extent.width),
			static_cast <float> (window.extent.height),
			0.0f, 1.0f
		};

		command_buffer.setViewport(0, viewport);
		command_buffer.setScissor(0, vk::Rect2D {
			vk::Offset2D {0, 0},
			window.extent
		});

		// Push constants
		PC_Material mat;
		mat.albedo = glm::vec3 {1.0f, 0.0f, 0.0f};
		mat.type = Shading::eDiffuse;
		mat.hightlight = 0;
		mat.has_albedo = true;
		mat.has_normal = false;

		PC_Vertex pc;
		pc.model = glm::mat4(1.0f);
		pc.view = camera.view();
		pc.projection = camera.perspective();
		pc.material = mat;

		std::array <PC_Vertex, 1> pcs = {pc};

		command_buffer.pushConstants <PC_Vertex>
			(*ppl, vk::ShaderStageFlagBits::eVertex, 0, pcs);

		// Start the render pass
		std::array <vk::ClearValue, 2> clear_values = {
			vk::ClearValue {
				vk::ClearColorValue {
					std::array <float, 4> {0.0f, 0.0f, 0.0f, 1.0f}
				}
			},
			vk::ClearValue {
				vk::ClearDepthStencilValue {
					1.0f, 0
				}
			}
		};

		command_buffer.beginRenderPass(
			vk::RenderPassBeginInfo {
				*render_pass,
				*framebuffers[image_index],
				vk::Rect2D {
					vk::Offset2D {0, 0},
					window.extent
				},
				static_cast <uint32_t> (clear_values.size()),
				clear_values.data()
			},
			vk::SubpassContents::eInline
		);

		// Bind the pipeline
		command_buffer.bindPipeline(vk::PipelineBindPoint::eGraphics, *pipeline);

		// Bind the descriptor set
		command_buffer.bindDescriptorSets(
			vk::PipelineBindPoint::eGraphics,
			*ppl, 0, {*dset}, {}
		);

		// Bind the cube and draw it
		command_buffer.bindVertexBuffers(0, *box_vk.vertex_buffer.buffer, {0});
		command_buffer.bindIndexBuffer(*box_vk.index_buffer.buffer, 0, vk::IndexType::eUint32);
		command_buffer.drawIndexed(
			static_cast <uint32_t> (box_mesh.indices().size()),
			1, 0, 0, 0
		);

		// End the render pass
		command_buffer.endRenderPass();

		// End
		command_buffer.end();
	};

	// Submit the command buffer
	vk::PipelineStageFlags stage_flags = vk::PipelineStageFlagBits::eColorAttachmentOutput;

	// Initialize per frame data
	auto frame_data = std::vector <FrameData> (framebuffers.size());
	for (auto &frame : frame_data) {
		frame = FrameData {device};
		std::cout << "FrameData:\n\tFence " << *frame.fence<< std::endl;
		std::cout << "\tPresent " << *frame.present_completed << std::endl;
		std::cout << "\tRender " << *frame.render_completed << std::endl;
	}

	// Present while valid window
	uint32_t frame_index = 0;
	while (!glfwWindowShouldClose(window.handle)) {
		// Poll events
		glfwPollEvents();

		// Handle input
		input_handling(window.handle);

		// Get command buffer
		const auto &command_buffer = command_buffers[frame_index];

		// TODO: handle resizing

		// Acquire an image from the swap chain
		vk::Result result;
		std::tie(result, image_index) = swapchain.swapchain.acquireNextImage(
			std::numeric_limits <uint64_t> ::max(),
			*frame_data[frame_index].present_completed
		);

		KOBRA_ASSERT(result == vk::Result::eSuccess, "Failed to acquire swapchain image");

		// Wait for the previous frame to finish
		while(device.waitForFences({*frame_data[frame_index].fence}, VK_TRUE, std::numeric_limits <uint64_t> ::max()) == vk::Result::eTimeout);

		// Reset the fence
		device.resetFences(*frame_data[frame_index].fence);

		// Record the command buffer
		record(command_buffer);

		// Submit the command buffer
		vk::SubmitInfo submit_info = {
			1, &*frame_data[frame_index].present_completed,
			&stage_flags,
			1, &*command_buffer,
			1, &*frame_data[frame_index].render_completed
		};

		graphics_queue.submit(submit_info, *frame_data[frame_index].fence);

		// Present the image
		vk::PresentInfoKHR present_info {
			*frame_data[frame_index].render_completed,
			*swapchain.swapchain,
			image_index // TODO: the problem is here?
		};

		result = present_queue.presentKHR(present_info);

		KOBRA_ASSERT(result == vk::Result::eSuccess, "Failed to present swapchain image");

		// Increment frame index
		frame_index = (frame_index + 1) % framebuffers.size();
	}

	// Wait before exiting
	device.waitIdle();
}

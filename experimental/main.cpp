// Standard headers
#include <iostream>

// More Vulkan stuff
#include <vulkan/vulkan_core.h>
#include <vulkan/vulkan_raii.hpp>
#include <vulkan/vulkan_structs.hpp>

#define KOBRA_VALIDATION_LAYERS
#define KOOBRA_THROW_ERROR

// Engine headers
#include "../include/backend.hpp"
#include "../include/mesh.hpp"
#include "../include/camera.hpp"
#include "../shaders/raster/bindings.h"

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

	std::cout << "Pitch: " << pitch << " Yaw: " << yaw << std::endl;
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
	float speed = 0.1f;

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

	std::cout << "Position: " << camera.transform.position.x << " "
		<< camera.transform.position.y << " "
		<< camera.transform.position.z << std::endl;
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
		vk::BufferUsageFlagBits::eVertexBuffer,
		vk::MemoryPropertyFlagBits::eHostVisible
			| vk::MemoryPropertyFlagBits::eHostCoherent
	);

	BufferData index_buffer = BufferData(phdev, device,
		mesh.indices().size() * sizeof(uint32_t),
		vk::BufferUsageFlagBits::eIndexBuffer,
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
using BuildRange = vk::AccelerationStructureBuildRangeInfoKHR;
using Flags = vk::BuildAccelerationStructureFlagsKHR;
using Accelerator = vk::AccelerationStructureKHR;

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

// Final acceleration structure (AS) info
struct AccelerationStructure {
	Accelerator			tlas;
	std::vector <Accelerator>	blas;
};

// Create AS info
void build_as(const std::vector <BLAS> &input, const Flags &flags)
{
}

}		// Acceleration structure functions

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
		"VK_KHR_deferred_host_operations"
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

	/* Allocate vertex and index buffer
	auto vertex_buffer = BufferData(phdev, device,
		box_mesh.vertices().size() * sizeof(Vertex),
		vk::BufferUsageFlagBits::eVertexBuffer,
		vk::MemoryPropertyFlagBits::eHostVisible
			| vk::MemoryPropertyFlagBits::eHostCoherent
	);

	auto index_buffer = BufferData(phdev, device,
		box_mesh.indices().size() * sizeof(uint32_t),
		vk::BufferUsageFlagBits::eIndexBuffer,
		vk::MemoryPropertyFlagBits::eHostVisible
			| vk::MemoryPropertyFlagBits::eHostCoherent
	);

	// Fill with data
	vertex_buffer.upload(box_mesh.vertices());
	index_buffer.upload(box_mesh.indices()); */

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

	auto _vertex_input_attributes = Vertex::vertex_attributes();

	// Convert to vk::VertexInputAttributeDescription
	auto vertex_input_attributes = std::vector <vk::VertexInputAttributeDescription> {};
	for (auto &attr : _vertex_input_attributes) {
		vk::Format format;

		// Convert from VkFormat to vk::Format
		if (attr.format == VK_FORMAT_R32G32B32_SFLOAT)
			format = vk::Format::eR32G32B32Sfloat;
		else if (attr.format == VK_FORMAT_R32G32B32A32_SFLOAT)
			format = vk::Format::eR32G32B32A32Sfloat;
		else if (attr.format == VK_FORMAT_R32G32_SFLOAT)
			format = vk::Format::eR32G32Sfloat;

		vertex_input_attributes.push_back(
			vk::VertexInputAttributeDescription {
				attr.location,
				attr.binding,
				format,
				attr.offset
			}
		);
	}

	// Create the graphics pipeline
	auto grp_info = GraphicsPipelineInfo {
		.device = device,
		.render_pass = render_pass,

		.vertex_shader = vertex,
		.fragment_shader = fragment,

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
		img = std::move(make_texture(temp_command_buffer,
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
	for (auto &fd : frame_data)
		fd = FrameData {device};

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
		device.resetFences({*frame_data[frame_index].fence});

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

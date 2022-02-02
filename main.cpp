#define MERCURY_VALIDATION_LAYERS
#define MERCURY_VALIDATION_ERROR_ONLY
// #define MERCURY_THROW_ERROR

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

// Standard headers
#include <iostream>
#include <cstring>

// Engine headers
#include "include/backend.hpp"
#include "include/light.hpp"
#include "include/object.hpp"
#include "include/camera.hpp"
#include "include/logger.hpp"

// Pixel type
struct uvec4 {
	uint8_t b;
	uint8_t g;
	uint8_t r;
	uint8_t a;

	// Defaut constructor
	uvec4() : b(0), g(0), r(0), a(255) {}

	// Construtor as RGB
	uvec4(uint8_t r, uint8_t g, uint8_t b)
		: r(r), g(g), b(b), a(255) {}
	
	// Constructor as RGBA
	uvec4(uint8_t r, uint8_t g, uint8_t b, uint8_t a)
		: r(r), g(g), b(b), a(a) {}
};

// Basic operations
inline uvec4 operator*(float s, const uvec4 &v)
{
        return {
                (uint8_t) (s * v.r),
                (uint8_t) (s * v.g),
                (uint8_t) (s * v.b),
                (uint8_t) (s * v.a)
        };
}

Camera camera {
        Transform {
		glm::vec3(0.0f, 0.0f, -4.0f)
	},
 
        Tunings {
                90.0f,
                800,
                600
        }
};

// Keyboard callback
// TODO: in class
bool rerender = true;

void key_callback(GLFWwindow *window, int key, int scancode, int action, int mods)
{
        if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS)
                glfwSetWindowShouldClose(window, GL_TRUE);

        // Camera movement
        float speed = 0.5f;
        if (key == GLFW_KEY_W) {
                camera.transform.position += camera.transform.forward * speed;
                rerender = true;
        } else if (key == GLFW_KEY_S) {
                camera.transform.position -= camera.transform.forward * speed;
                rerender = true;
        }

        if (key == GLFW_KEY_A) {
                camera.transform.position -= camera.transform.right * speed;
                rerender = true;
        } else if (key == GLFW_KEY_D) {
                camera.transform.position += camera.transform.right * speed;
                rerender = true;
        }
}

// Pixel buffer 
Vulkan::Buffer pixel_buffer;
Vulkan::Buffer world_buffer;
Vulkan::Buffer object_buffer;

// Compute shader
VkShaderModule compute_shader;

// Command buffer function per frame index
void cmd_buffer_maker(Vulkan *vk, size_t i) {
	// Clear color
	VkClearValue clear_color = {
		{ {0.0f, 0.0f, 0.0f, 1.0f} }
	};
	
	// Render pass creation info
	VkRenderPassBeginInfo render_pass_info {
		.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
		.renderPass = vk->render_pass,
		.framebuffer = vk->swch_framebuffers[i],
		.renderArea = {
			.offset = {0, 0},
			.extent = vk->swch_extent
		},
		.clearValueCount = 1,
		.pClearValues = &clear_color
	};

	// Render pass creation
	vkCmdBeginRenderPass(
		vk->command_buffers[i],
		&render_pass_info,
		VK_SUBPASS_CONTENTS_INLINE
	);

		// Create pipeline
		VkPipelineLayoutCreateInfo pipeline_layout_info {
			.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
			.setLayoutCount = 1,
			.pSetLayouts = &vk->descriptor_set_layouts[i],
			.pushConstantRangeCount = 0,
			.pPushConstantRanges = nullptr
		};

		VkPipelineLayout pipeline_layout;

		VkResult res = vkCreatePipelineLayout(
			vk->device,
			&pipeline_layout_info,
			nullptr,
			&pipeline_layout
		);

		if (res != VK_SUCCESS) {
			std::cerr << "Failed to create pipeline layout" << std::endl;
			return;
		}

		// Execute compute shader on the pixel buffer
		VkPipeline pipeline;

		VkComputePipelineCreateInfo compute_pipeline_info {
			.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
			.stage = {
				.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
				.stage = VK_SHADER_STAGE_COMPUTE_BIT,
				.module = compute_shader,
				.pName = "main"
			},
			.layout = pipeline_layout
		};

		res = vkCreateComputePipelines(
			vk->device,
			VK_NULL_HANDLE,
			1,
			&compute_pipeline_info,
			nullptr,
			&pipeline
		);

		if (res != VK_SUCCESS) {
			std::cerr << "Failed to create compute pipeline" << std::endl;
			return;
		}

		// Bind pipeline
		vkCmdBindPipeline(
			vk->command_buffers[i],
			VK_PIPELINE_BIND_POINT_COMPUTE,
			pipeline
		);

		// Bind buffer
		vkCmdBindDescriptorSets(
			vk->command_buffers[i],
			VK_PIPELINE_BIND_POINT_COMPUTE,
			pipeline_layout,
			0,
			1,
			vk->descriptor_sets.data(),
			0,
			nullptr
		);

	vkCmdEndRenderPass(vk->command_buffers[i]);
		
	// Dispatch
	vkCmdDispatch(
		vk->command_buffers[i],
		50, 50, 1
	);
		
	// Get image at current index
	VkImage image = vk->swch_images[i];

	// Buffer copy regions
	VkBufferImageCopy buffer_copy_region {
		.bufferOffset = 0,
		.bufferRowLength = 0,
		.bufferImageHeight = 0,
		.imageSubresource = {
			.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
			.mipLevel = 0,
			.baseArrayLayer = 0,
			.layerCount = 1
		},
		.imageOffset = {0, 0, 0},
		.imageExtent = {
			vk->swch_extent.width,
			vk->swch_extent.height,
			1
		}
	};

	// Copy buffer to image
	vkCmdCopyBufferToImage(
		vk->command_buffers[i],
		pixel_buffer.buffer,
		image,
		VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
		1,
		&buffer_copy_region
	);
};

// Initialize the pixel buffer
uvec4 *init_pixels(int width, int height, const uvec4 &base)
{
	uvec4 *pixels = new uvec4[width * height];

	for (int i = 0; i < width * height; i++) {
		pixels[i] = base;
	}

	return pixels;
}

// All objects
int nobjs = 5;

Renderable *objects[] = {
        new Sphere(vec3(0.0f, 0.0f, 4.0f), 1.0f),
        new Sphere(vec3(3.0f, 0.0f, 3.0f), 3.0f),
        new Sphere(vec3(6.0f, -2.0f, 5.0f), 6.0f),
        new Sphere(vec3(6.0f, 3.0f, 10.0f), 2.0f),
        new Sphere(vec3(6.0f, 3.0f, -4.0f), 2.0f),
};

Object *lights[] = {
        new PointLight(vec3(0.0f, 0.0f, 0.0f))
};

// Initialize the pixel buffer
void clear(uvec4 *pixels, int width, int height, const uvec4 &base)
{
        for (int i = 0; i < width * height; i++)
                pixels[i] = base;
}

// Convert color to discetized grey scale value
uvec4 dcgrey(const uvec4 &color)
{
        // Discrete steps
        static const int steps = 16;

        // Get grey scale vector
        uint8_t grey = (color.r + color.g + color.b) / 3;

        // Discretize (remove remainder)
        grey = grey - (grey % steps);

        // Return color
        return uvec4 {grey, grey, grey, color.a};
}

void render(uvec4 *pixels, int width, int height)
{
        // Color wheel
        static uvec4 colors[] = {
                {0x00, 0x00, 0xFF, 0xFF},
                {0x00, 0xFF, 0x00, 0xFF},
                {0x00, 0xFF, 0xFF, 0xFF},
                {0xFF, 0x00, 0x00, 0xFF},
                {0xFF, 0x00, 0xFF, 0xFF},
                {0xFF, 0xFF, 0x00, 0xFF},
                {0xFF, 0xFF, 0xFF, 0xFF}
        };

        // Color wheel index
        int cid = 0;

        // Each object to a color
        int *obj_colors = new int[nobjs];
        for (int i = 0; i < nobjs; i++) {
                obj_colors[i] = cid++;
                if (cid == 8)
                        cid = 0;
        }

        // Iterate over the pixels
        glm::vec3 cpos = camera.transform.position;
        for (int y = 0; y < height; y++) {
                for (int x = 0; x < width; x++) {
                        // NDC coordinates
                        float nx = (x + 0.5f) / (float) width;
                        float ny = (y + 0.5f) / (float) height;

                        Ray ray = camera.ray(nx, ny);

                        // Find the closest object
                        int iclose = -1;

                        // TODO: change to tclose
                        float dclose = std::numeric_limits <float> ::max();

                        for (int i = 0; i < nobjs; i++) {
                                float d = objects[i]->intersect(ray);
                                if (d > 0 && dclose > d) {
                                        dclose = d;
                                        iclose = i;
                                }
                        }

                        // If there is an intersection
                        if (iclose != -1) {
                                // Get point of intersection
				glm::vec3 p = ray.at(dclose);

                                // Calculate diffuse
				glm::vec3 n = objects[iclose]->normal(p);
				glm::vec3 l = glm::normalize(lights[0]->position - p);
                                float diffuse = glm::max(glm::dot(n, l), 0.0f);

                                // Color the pixel
                                uvec4 fcolor = diffuse * colors[obj_colors[iclose]];
                                pixels[y * width + x] = fcolor; // dcgrey(fcolor);
                        }
                }
        }

        // Free memory
        delete[] obj_colors;

        // TODO: anti aliasing pass, plus any post processing
}

void descriptor_set_maker(Vulkan *vulkan, int i)
{
	VkDescriptorBufferInfo pb_info {
		.buffer = pixel_buffer.buffer,
		.offset = 0,
		.range = pixel_buffer.size
	};
	
	VkDescriptorBufferInfo wb_info {
		.buffer = world_buffer.buffer,
		.offset = 0,
		.range = world_buffer.size
	};
	
	VkDescriptorBufferInfo ob_info {
		.buffer = object_buffer.buffer,
		.offset = 0,
		.range = object_buffer.size
	};

	VkWriteDescriptorSet pb_write = {
		.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
		.dstSet = vulkan->descriptor_sets[i],
		.dstBinding = 0,
		.dstArrayElement = 0,
		.descriptorCount = 1,
		.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
		.pBufferInfo = &pb_info
	};
	
	VkWriteDescriptorSet wb_write = {
		.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
		.dstSet = vulkan->descriptor_sets[i],
		.dstBinding = 1,
		.dstArrayElement = 0,
		.descriptorCount = 1,
		.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
		.pBufferInfo = &wb_info
	};
	
	VkWriteDescriptorSet ob_write = {
		.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
		.dstSet = vulkan->descriptor_sets[i],
		.dstBinding = 2,
		.dstArrayElement = 0,
		.descriptorCount = 1,
		.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
		.pBufferInfo = &ob_info
	};

	VkWriteDescriptorSet writes[] = {
		pb_write,
		wb_write,
		ob_write
	};

	vkUpdateDescriptorSets(
		vulkan->device, 3,
		&writes[0],
		0, nullptr
	);
}

struct alignas(16) Aligned {
	glm::vec4 data;
};

// World structure
// TODO: refactor to GPU world,
// store cpu world in another structure that is called World
struct World {
	uint32_t objects;
	uint32_t lights;
	uint32_t backgound;

	uint32_t width = 800;
	uint32_t height = 600;

	alignas(16) glm::vec3 camera_position = glm::vec3(0.0f, 0.0f, -15.0f);

	float fov = camera.tunings.fov;
	float scale = camera.tunings.scale;
	float aspect = camera.tunings.aspect;

	// TODO: max objects
	Aligned sphere_position[16];
};

World world = {
	.objects = (uint32_t) 1,
	.lights = 1,
	.backgound = 0x202020,
	.sphere_position {
		{ {0.0f, 0.0f, 0.0f, 6.0f} }
	}
};

int main()
{
	std::cout << "Align of Aligned: " << alignof(Aligned) << std::endl;
	std::cout << "Sizeof world: " << sizeof(World) << std::endl;
	std::cout << "Align of vec4: " << alignof(glm::vec4) << std::endl;

	// Initialize Vulkan
	Vulkan vulkan;

        uvec4 base = {200, 200, 220, 255};
        uvec4 *pixels = init_pixels(800, 600, base);

	Aligned *objects = new Aligned[16] {
		{ {0.0, 0.0, 0.0, 3.0} }
	};

	// Pixel data
	size_t pixel_size = sizeof(uvec4) * 800 * 600;
	size_t world_size = sizeof(World);
	size_t object_size = sizeof(Aligned) * 16;

	VkBufferUsageFlags pixel_usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT
		| VK_BUFFER_USAGE_TRANSFER_SRC_BIT
		| VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;

	VkBufferUsageFlags world_usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT
		| VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;

	VkBufferUsageFlags object_usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT
		| VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;

	// Create buffers
	vulkan.make_buffer(pixel_buffer, pixel_size, pixel_usage);
	vulkan.make_buffer(world_buffer, sizeof(World), world_usage);
	vulkan.make_buffer(object_buffer, object_size, object_usage);

	vulkan.map_buffer(&pixel_buffer, pixels, pixel_size);
	vulkan.map_buffer(&world_buffer, &world, sizeof(World));
	vulkan.map_buffer(&object_buffer, objects, object_size);

	std::cout << "Pixel buffer: " << pixel_buffer.size << std::endl;
	std::cout << "World buffer: " << world_buffer.size << std::endl;
	std::cout << "Object buffer: " << object_buffer.size << std::endl;

	// Compute shader descriptor
	compute_shader = vulkan.make_shader("shaders/pixel.spv");

	// Buffer descriptor
	for (int i = 0; i < vulkan.swch_images.size(); i++)
		descriptor_set_maker(&vulkan, i);

	// Set keyboard callback
	glfwSetKeyCallback(vulkan.window, key_callback);

	vulkan.set_command_buffers(cmd_buffer_maker);
	while (!glfwWindowShouldClose(vulkan.window)) {
		glfwPollEvents();
		vulkan.frame();
	}

	vulkan.idle();
}

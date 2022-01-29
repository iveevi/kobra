#define MERCURY_VALIDATION_LAYERS

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#include <iostream>

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
Vulkan::Buffer *pixel_buffer;

// Command buffer function per frame index
auto cmd_buffer_maker = [](Vulkan *vk, size_t i) {
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

	vkCmdEndRenderPass(vk->command_buffers[i]);
		
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
		pixel_buffer->buffer,
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

int main()
{
	// Initialize Vulkan
	Vulkan vulkan;

        uvec4 base = {200, 200, 200, 255};
        uvec4 *pixels = init_pixels(800, 600, base);
	render(pixels, 800, 600);

	// Pixel data
	size_t size = sizeof(uvec4) * 800 * 600;
	VkBufferUsageFlags usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;

	// Create a buffer
	pixel_buffer = vulkan.make_buffer(size, usage);
	vulkan.map_buffer(pixel_buffer, pixels, size);

	// Set keyboard callback
	glfwSetKeyCallback(vulkan.window, key_callback);

	vulkan.set_command_buffers(cmd_buffer_maker);
	while (!glfwWindowShouldClose(vulkan.window)) {
		glfwPollEvents();

		// Regenerate the image
		if (rerender) {
			clear(pixels, 800, 600, base);
			render(pixels, 800, 600);
			vulkan.map_buffer(pixel_buffer, pixels, size);
			rerender = false;
		}

		vulkan.frame();
	}

	vulkan.idle();
}

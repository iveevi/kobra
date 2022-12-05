#ifndef KOBRA_LAYERS_IMGUI_H_
#define KOBRA_LAYERS_IMGUI_H_

// ImGUI headers
#include <imgui/imgui.h>
#include <imgui/backends/imgui_impl_glfw.h>
#include <imgui/backends/imgui_impl_vulkan.h>

// Engine headers
#include "../backend.hpp"
#include "../ui/attachment.hpp"

namespace kobra {

namespace layers {

// ImGUI rendering backend
class ImGUI {
	// TODO: GraphicalLayer abstract for these common structures
	// 	.initialize(phdev, dev, ...)

	// TODO: m_...
	vk::raii::PhysicalDevice *phdev = nullptr;
	vk::raii::Device *device = nullptr;
	vk::raii::CommandPool *command_pool = nullptr;

	vk::raii::DescriptorPool descriptor_pool = nullptr;
	vk::raii::RenderPass render_pass = nullptr;
	
	std::vector <std::shared_ptr <ui::ImGUIAttachment>> attachments;
public:
	// Default constructor
	ImGUI() = default;

	// Constructor
	ImGUI(const Context &context,
			const Window &window,
			const vk::raii::Queue &queue) {
		// TODO: just get the graphics queue from the device...

		// Assume that ImGUI is initialized, including context and backend...

		// Copy context parameters
		phdev = context.phdev;
		device = context.device;
		command_pool = context.command_pool;

		// Create dedicated Vulkan resources for ImGUI
		std::vector <vk::DescriptorPoolSize> pool_sizes {
			vk::DescriptorPoolSize(vk::DescriptorType::eSampler, 1000),
			vk::DescriptorPoolSize(vk::DescriptorType::eCombinedImageSampler, 1000),
			vk::DescriptorPoolSize(vk::DescriptorType::eSampledImage, 1000),
			vk::DescriptorPoolSize(vk::DescriptorType::eStorageImage, 1000),
			vk::DescriptorPoolSize(vk::DescriptorType::eUniformTexelBuffer, 1000),
			vk::DescriptorPoolSize(vk::DescriptorType::eStorageTexelBuffer, 1000),
			vk::DescriptorPoolSize(vk::DescriptorType::eUniformBuffer, 1000),
			vk::DescriptorPoolSize(vk::DescriptorType::eStorageBuffer, 1000),
			vk::DescriptorPoolSize(vk::DescriptorType::eUniformBufferDynamic, 1000),
			vk::DescriptorPoolSize(vk::DescriptorType::eStorageBufferDynamic, 1000),
			vk::DescriptorPoolSize(vk::DescriptorType::eInputAttachment, 1000)
		};

		// TODO: pass max sets as a parameter
		descriptor_pool = kobra::make_descriptor_pool(*device, pool_sizes);
	
		render_pass = kobra::make_render_pass(*device,
			{context.swapchain_format}, {vk::AttachmentLoadOp::eLoad},
			context.depth_format, vk::AttachmentLoadOp::eClear
		);

		ImGui_ImplVulkan_InitInfo init_info = {};
		init_info.Instance = *kobra::get_vulkan_instance();
		init_info.PhysicalDevice = **phdev;
		init_info.Device = **device;
		init_info.Queue = *queue;
		init_info.DescriptorPool = *descriptor_pool;
		init_info.MinImageCount = 3;
		init_info.ImageCount = 3;
		init_info.MSAASamples = VK_SAMPLE_COUNT_1_BIT; // TODO: pass as a parameter
		
		ImGui_ImplVulkan_Init(&init_info, *render_pass);
		
		// Load fonts
		command_now(*device, *command_pool,
			[&](const vk::raii::CommandBuffer &cmd) {
				ImGui_ImplVulkan_CreateFontsTexture(*cmd);
			}
		);
		
		// Destroy CPU-side resources
		ImGui_ImplVulkan_DestroyFontUploadObjects();
	}

	// Render ImGUI
	void render(const vk::raii::CommandBuffer &cmd,
			const vk::raii::Framebuffer &framebuffer,
			const vk::Extent2D &extent,
			const RenderArea &ra = RenderArea::full()) {
		// Apply the render area
		ra.apply(cmd, extent);

		// Start render pass
		std::array <vk::ClearValue, 2> clear_values {
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

		cmd.beginRenderPass(
			vk::RenderPassBeginInfo {
				*render_pass,
				*framebuffer,
				vk::Rect2D {
					vk::Offset2D {0, 0},
					extent
				},
				static_cast <uint32_t> (clear_values.size()),
				clear_values.data()
			},
			vk::SubpassContents::eInline
		);

		// Start ImGUI frame
		ImGui_ImplVulkan_NewFrame();
		ImGui_ImplGlfw_NewFrame();

		// Render ImGUI
		ImGui::NewFrame();

		for (auto &attachment : attachments)
			attachment->render();

		ImGui::Render();
	
		// Write to the command buffer
		ImGui_ImplVulkan_RenderDrawData(
			ImGui::GetDrawData(),
			*cmd
		);

		cmd.endRenderPass();
	}

	// Add attachments
	void attach(std::shared_ptr <ui::ImGUIAttachment> attachment) {
		attachments.emplace_back(attachment);
	}

	// Set font
	void set_font(const std::string &path, float size) {
		ImGuiIO &io = ImGui::GetIO();
		io.Fonts->Clear();
		io.Fonts->AddFontFromFileTTF(path.c_str(), size);

		// Rebuild font texture
		command_now(*device, *command_pool,
			[&](const vk::raii::CommandBuffer &cmd) {
				ImGui_ImplVulkan_CreateFontsTexture(*cmd);
			}
		);

		// Destroy CPU-side resources
		ImGui_ImplVulkan_DestroyFontUploadObjects();
	}
};

}

}

#endif

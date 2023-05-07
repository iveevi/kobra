#ifndef KOBRA_LAYERS_IMGUI_H_
#define KOBRA_LAYERS_IMGUI_H_

// Standard headers
#include <vector>

// ImGUI headers
#include <imgui/imgui.h>
#include <imgui/backends/imgui_impl_glfw.h>
#include <imgui/backends/imgui_impl_vulkan.h>

// Engine headers
#include "../backend.hpp"
#include "../ui/attachment.hpp"
#include "common.hpp"

namespace kobra {

namespace layers {

// UI rendering backend
struct UserInterface {
	// TODO: GraphicalLayer abstract for these common structures
	// 	.initialize(phdev, dev, ...)

	// TODO: m_...
	vk::raii::PhysicalDevice *phdev = nullptr;
	vk::raii::Device *device = nullptr;
	vk::raii::CommandPool *command_pool = nullptr;

	vk::raii::DescriptorPool descriptor_pool = nullptr;
	vk::raii::RenderPass render_pass = nullptr;
	
	std::vector <std::shared_ptr <ui::ImGuiAttachment>> attachments;

	// Default constructor
	UserInterface() = default;

	// Constructor
	UserInterface(const Context &context,
			const Window &window,
			const vk::raii::Queue &queue,
			const std::pair <std::string, size_t> font,
			vk::AttachmentLoadOp load_op = vk::AttachmentLoadOp::eLoad) {
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
			{context.swapchain_format}, {load_op},
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
		
		// Load font
		ImGuiIO &io = ImGui::GetIO();
		io.Fonts->AddFontFromFileTTF(font.first.c_str(), font.second);

		vk::raii::Queue temp_queue {*device, 0, 0};
		submit_now(*device, temp_queue, *command_pool,
			[&](const vk::raii::CommandBuffer &cmd) {
				ImGui_ImplVulkan_CreateFontsTexture(*cmd);
			}
		);
		
		// Destroy CPU-side resources
		ImGui_ImplVulkan_DestroyFontUploadObjects();
	}

	// Render ImGUI
	struct RenderOptions {
		bool dockspace;
	};

	void render(const vk::raii::CommandBuffer &cmd,
			const vk::raii::Framebuffer &framebuffer,
			const vk::Extent2D &extent,
			const RenderArea &ra = RenderArea::full(),
			const RenderOptions &options = RenderOptions {false}) {
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

		// If the dockspace is enabled, create it
		if (options.dockspace)
			ImGui::DockSpaceOverViewport(ImGui::GetMainViewport());

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
	void attach(std::shared_ptr <ui::ImGuiAttachment> attachment) {
		attachments.emplace_back(attachment);
	}
};

inline void start_user_interface(const UserInterface *ui, const RenderContext &rc, bool dockspace = false)
{
        // Apply the render area
        rc.render_area.apply(rc.cmd, rc.extent);

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

        rc.cmd.beginRenderPass(
                vk::RenderPassBeginInfo {
                        *ui->render_pass,
                        *rc.framebuffer,
                        vk::Rect2D {
                                vk::Offset2D {0, 0},
                                rc.extent
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

        // If the dockspace is enabled, create it
        if (dockspace)
                ImGui::DockSpaceOverViewport(ImGui::GetMainViewport());
}

inline void end_user_interface(const UserInterface *ui, const RenderContext &rc)
{
        ImGui::Render();

        // Write to the command buffer
        ImGui_ImplVulkan_RenderDrawData(
                ImGui::GetDrawData(),
                *rc.cmd
        );

        rc.cmd.endRenderPass();
}

}

}

#endif

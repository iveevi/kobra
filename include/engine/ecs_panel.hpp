#ifndef KOBRA_ENGINE_ECS_PANEL_H_
#define KOBRA_ENGINE_ECS_PANEL_H_

// Engine headers
#include "../ecs.hpp"
#include "../layers/font_renderer.hpp"

namespace kobra {

namespace engine {

class ECSPanel {
	const ECS		*ecs;
	layers::FontRenderer	font_renderer;

	// Vulkan resources
	vk::raii::RenderPass	render_pass = nullptr;
public:
	// Constructor
	ECSPanel(const Context &ctx, const ECS &ecs_, vk::AttachmentLoadOp load = vk::AttachmentLoadOp::eLoad)
			: ecs(&ecs_) {
		// Create render pass
		render_pass = make_render_pass(*ctx.device,
			ctx.swapchain_format,
			ctx.depth_format, load
		);

		// Create font renderer
		font_renderer = layers::FontRenderer(ctx, render_pass, "resources/fonts/noto_sans.ttf");
	}

	// Render
	void render(const vk::raii::CommandBuffer &cmd, const vk::raii::Framebuffer &framebuffer, const vk::Extent2D &extent) {
		std::vector <Text> texts;

		// Add all names
		size_t x = 10;
		size_t y = 10;

		for (int i = 0; i < ecs->size(); i++) {
			std::string name = ecs->name(i);

			// TODO: method to calculate text dimensions (font
			// renderer method)
			Text text {.text = name, .anchor = {x, y}, .size = 0.6f};
			text.color = {0.7, 0.7, 0.9};
			y += font_renderer.size(text).y;

			texts.push_back(text);
		}

		// Start render pass
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

		cmd.beginRenderPass(
			vk::RenderPassBeginInfo {
				*render_pass,
				*framebuffer,
				vk::Rect2D {
					vk::Offset2D {0, 0},
					extent,
				},
				static_cast <uint32_t> (clear_values.size()),
				clear_values.data()
			},
			vk::SubpassContents::eInline
		);

		font_renderer.render(cmd, texts);

		cmd.endRenderPass();
	}
};

}

}

#endif

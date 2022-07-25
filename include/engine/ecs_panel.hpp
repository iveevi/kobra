#ifndef KOBRA_ENGINE_ECS_PANEL_H_
#define KOBRA_ENGINE_ECS_PANEL_H_

// Engine headers
#include "../app.hpp"
#include "../ui/button.hpp"
#include "../ecs.hpp"
#include "../layers/font_renderer.hpp"
#include "../layers/shape_renderer.hpp"

namespace kobra {

namespace engine {

class ECSPanel {
	const ECS		*ecs;
	layers::FontRenderer	font_renderer;
	layers::ShapeRenderer	shape_renderer;
	std::vector <ui::Button>	buttons;
	App::IO			*io;

	// Vulkan resources
	vk::raii::RenderPass	render_pass = nullptr;
public:
	// Constructor
	ECSPanel(const Context &ctx, const ECS &ecs_, App::IO &app_io, vk::AttachmentLoadOp load = vk::AttachmentLoadOp::eLoad)
			: ecs(&ecs_), io(&app_io) {
		// Create render pass
		render_pass = make_render_pass(*ctx.device,
			ctx.swapchain_format,
			ctx.depth_format, load
		);

		// Create font renderer
		font_renderer = layers::FontRenderer(ctx, render_pass, "resources/fonts/noto_sans.ttf");
		shape_renderer = layers::ShapeRenderer(ctx, render_pass);
	}

	// Render
	void render(const vk::raii::CommandBuffer &cmd, const vk::raii::Framebuffer &framebuffer, const vk::Extent2D &extent) {
		std::vector <ui::Text> texts;
		std::vector <ui::Rect *> rects;

		// Add all names
		size_t x = 10;
		size_t y = 10;

		if (buttons.size() < ecs->size())
			buttons.resize(ecs->size());

		for (int i = 0; i < ecs->size(); i++) {
			std::string name = ecs->get_entity(i).name;
			size_t cy = y;

			// TODO: method to calculate text dimensions (font
			// renderer method)
			ui::Text text {.text = name, .anchor = {x, y}, .size = 0.6f};
			text.color = {0.7, 0.7, 0.9};
			y += font_renderer.size(text).y;

			texts.push_back(text);

			// Button
			auto handler = [name](void *user) {
				std::cout << "Clicked on (text) entitiy \"" << name << "\"" << std::endl;
			};

			ui::Button::Args bargs {
				.min = {x, cy},
				.max = {x + font_renderer.size(text).x, y},

				.idle = {0.5, 0.5, 0.5},
				.hover = {0.7, 0.7, 0.9},
				.pressed = {0.9, 0.9, 0.9},

				.on_click = {{nullptr, handler}}
			};

			buttons[i] = ui::Button(io->mouse_events, bargs);
			rects.push_back(buttons[i].shape());

			y += 10;
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

		shape_renderer.render(cmd, rects);
		font_renderer.render(cmd, texts);

		cmd.endRenderPass();
	}
};

}

}

#endif

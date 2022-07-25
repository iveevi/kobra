#include "include/app.hpp"
#include "include/backend.hpp"
#include "include/common.hpp"
#include "include/ecs.hpp"
#include "include/engine/ecs_panel.hpp"
#include "include/io/event.hpp"
#include "include/layers/font_renderer.hpp"
#include "include/layers/raster.hpp"
#include "include/layers/raytracer.hpp"
#include "include/layers/shape_renderer.hpp"
#include "include/logger.hpp"
#include "include/renderer.hpp"
#include "include/scene.hpp"
#include "include/transform.hpp"
#include "include/types.hpp"
#include "include/ui/button.hpp"
#include "include/ui/slider.hpp"
#include "tinyfiledialogs.h"
#include <glslang/Public/ShaderLang.h>

using namespace kobra;

// Scene path
std::string scene_path = "scenes/scene.kobra";

constexpr char color_shader[] = R"(
#version 450

layout (location = 0) in vec3 in_color;
layout (location = 1) in vec2 in_uv;

layout(location = 0) out vec4 fragment;

layout (push_constant) uniform PushConstant {
	vec2 center;
	float width;
	float height;
	float radius;
	float thickness;
};

// Main function
void main()
{
	vec2 uv = (in_uv - center);
	fragment = vec4(1, 0, 0, 1);
}
)";

// Test app
struct ECSApp : public BaseApp {
	layers::Raster	rasterizer;
	layers::Raytracer raytracer;
	layers::FontRenderer font_renderer;
	layers::ShapeRenderer shape_renderer;
	engine::ECSPanel panel;

	Scene scene;

	// TODO: will later also need a project manager
	Entity camera;

	static constexpr glm::vec2 window_size {1900, 1100};
	static constexpr float scene_graph_width = 400;
	static constexpr float component_panel_width = 400;
	static constexpr float project_explorer_height = 300;
	static constexpr glm::vec2 render_min = {scene_graph_width, 0};
	static constexpr glm::vec2 render_max = {
		window_size.x - component_panel_width,
		window_size.y - project_explorer_height
	};

	// Scene graph
	struct SceneGraph {
		ui::Rect r_background;

		std::vector <ui::Button> b_entities;

		ECS &ecs;
		layers::FontRenderer &fr;
		io::MouseEventQueue &mouse_events;

		SceneGraph(ECS &ecs_, layers::FontRenderer &fr_, io::MouseEventQueue &mouse_events_)
				: ecs(ecs_), fr(fr_), mouse_events(mouse_events_) {
			r_background = ui::Rect {
				.min = {5, 5},
				.max = {scene_graph_width - 5, window_size.y - 5},
				.color = glm::vec3 {0.6f, 0.7, 0.6f},
				.radius = 0.01f
			};
		}

		int selected_entity = -1;
		std::vector <ui::Rect *> shapes() {
			// TODO: rectangle border color
			std::vector <ui::Rect *> shapes {&r_background};

			// TODO: assuming that ECS didnt change if the size
			// didnt
			if (b_entities.size() != ecs.size()) {
				b_entities.resize(ecs.size());

				float x = 10;
				float y = 10;

				for (int i = 0; i < ecs.size(); i++) {
					auto &e = ecs.get_entity(i);

					struct _on_click {
						int index;

						void operator()(void *user) {
							int *i = (int *) user;
							*i = index;
						}
					};

					ui::Button::Args button_args {
						.min = {10, y},
						.max = {scene_graph_width - 10, y + 30.0f},
						.radius = 0.005f,

						.idle = glm::vec3 {0.6, 0.7, 0.6},
						.hover = glm::vec3 {0.7, 0.8, 0.7},
						.pressed = glm::vec3 {0.65, 0.8, 0.65},

						.on_click = {{&selected_entity, _on_click {i}}},
					};

					b_entities[i] = ui::Button(mouse_events, button_args);

					y += 40.0f;
				}
			}

			for (int i = 0; i < b_entities.size(); i++) {
				auto s = b_entities[i].shape();

				if (i == selected_entity)
					s->color = glm::vec3 {0.65, 0.8, 0.65};

				shapes.push_back(s);
			}

			return shapes;
		}

		std::vector <ui::Text> texts() {
			std::vector <ui::Text> texts;

			float x = 10;
			float y = 10;

			float minh = 30.0f;

			for (int i = 0; i < ecs.size(); i++) {
				auto &e = ecs.get_entity(i);

				auto t = ui::Text {
					.text = e.name,
					.anchor = {x + 5.0f, y},
					.color = glm::vec3 {0.5, 0.5, 1.0},
					.size = 0.4f
				};

				// Center from y to y + minh
				float h = fr.size(t).y;
				t.anchor.y = y + (minh - h) / 2.0f;

				y += 40.0f;

				texts.push_back(t);
			}

			return texts;
		}
	};

	ui::Rect color_rect;
	SceneGraph scene_graph;

	ECSApp(const vk::raii::PhysicalDevice &phdev, const std::vector <const char *> &extensions)
			: BaseApp(phdev, "ECSApp",
				vk::Extent2D {(uint32_t) window_size.x, (uint32_t) window_size.y},
				extensions, vk::AttachmentLoadOp::eLoad
			),
			rasterizer(get_context(), vk::AttachmentLoadOp::eClear),
			raytracer(get_context(), &sync_queue, vk::AttachmentLoadOp::eClear),
			font_renderer(get_context(), render_pass, "resources/fonts/noto_sans.ttf"),
			shape_renderer(get_context(), render_pass),
			panel(get_context(), scene.ecs, io),
			scene_graph(scene.ecs, font_renderer, io.mouse_events) {
		scene.load(get_device(), scene_path);
		// raytracer.environment_map(scene.p_environment_map);
		raytracer.environment_map("resources/skies/background_1.jpg");

		// Camera
		camera = scene.ecs.get_entity("Camera");

		// TODO: set camera properties (aspect)
		camera.get <Camera> ().tunings.aspect = (render_max.x - render_min.x)/(render_max.y - render_min.y);

		// Rect with custom shader program
		ShaderProgram program;
		program.set_source(color_shader);

		color_rect = ui::Rect {
			.min = {1000, 200},
			.max = {1100, 300},
			.color = glm::vec3 {0.6f, 0.7, 0.6f},
			.radius = 0.01f
		};

		color_rect.shader_program = program;

		// Input callbacks
		io.mouse_events.subscribe(mouse_callback, this);

		scene.ecs.info <Mesh> ();
	}

	int mode = 0;	// 0 for raster, 1 for raytracer
	bool tab_pressed = false;

	void active_input() {
		float speed = 20.0f * frame_time;

		// Camera movement
		// TODO: remove transform component from camera?
		auto &cam = camera.get <Camera> ();

		glm::vec3 forward = cam.transform.forward();
		glm::vec3 right = cam.transform.right();
		glm::vec3 up = cam.transform.up();

		if (io.input.is_key_down(GLFW_KEY_W))
			cam.transform.move(forward * speed);
		else if (io.input.is_key_down(GLFW_KEY_S))
			cam.transform.move(-forward * speed);

		if (io.input.is_key_down(GLFW_KEY_A))
			cam.transform.move(-right * speed);
		else if (io.input.is_key_down(GLFW_KEY_D))
			cam.transform.move(right * speed);

		if (io.input.is_key_down(GLFW_KEY_E))
			cam.transform.move(up * speed);
		else if (io.input.is_key_down(GLFW_KEY_Q))
			cam.transform.move(-up * speed);

		// Switch mode on tab
		if (io.input.is_key_down(GLFW_KEY_TAB)) {
			if (!tab_pressed) {
				tab_pressed = true;
				mode = (mode + 1) % 2;
			}
		} else {
			tab_pressed = false;
		}
	}

	float fps = 0;
	float time = 0;

	void record(const vk::raii::CommandBuffer &cmd,
			const vk::raii::Framebuffer &framebuffer) override {
		// Time things
		if (frame_time > 0)
			fps = (fps + 1.0f/frame_time) / 2.0f;

		time += frame_time;

		// Text things
		std::vector <ui::Text> texts {
			ui::Text {
				.text = common::sprintf("%.2f fps", fps),
				.anchor = {scene_graph_width + 5, 5},
				.size = 0.4f
			},
		};

		for (auto &t : scene_graph.texts())
			texts.push_back(t);

		// Shape things
		std::vector <ui::Rect *> rects {
			&color_rect
		};

		for (auto &s : scene_graph.shapes())
			rects.push_back(s);

		// Input
		active_input();

		// Begin command buffer
		cmd.begin({});

		if (mode == 1)
			raytracer.render(cmd, framebuffer, scene.ecs, {render_min, render_max});
		else
			rasterizer.render(cmd, framebuffer, scene.ecs, {render_min, render_max});

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

		// TODO: ui layer  will have a push interface each frame

		cmd.endRenderPass();

		cmd.end();

		// V-sync
		sync_queue.push([]() {});
	}

	void terminate() override {
		if (io.input.is_key_down(GLFW_KEY_ESCAPE))
			glfwSetWindowShouldClose(window.handle, true);
	}

	// Mouse callback
	static void mouse_callback(void *us, const io::MouseEvent &event) {
		static const int pan_button = GLFW_MOUSE_BUTTON_MIDDLE;
		static const int alt_pan_button = GLFW_MOUSE_BUTTON_LEFT;
		static const int select_button = GLFW_MOUSE_BUTTON_LEFT;

		static const float sensitivity = 0.001f;

		static bool first_movement = true;
		static bool dragging = false;
		static bool dragging_select = false;
		static bool gizmo_dragging = false;

		static float px = 0.0f;
		static float py = 0.0f;

		static glm::vec2 previous_dir {0.0f, 0.0f};

		static float yaw = 0.0f;
		static float pitch = 0.0f;

		auto &app = *static_cast <ECSApp *> (us);
		auto &cam = app.camera.get <Camera> ();

		// Deltas and directions
		float dx = event.xpos - px;
		float dy = event.ypos - py;
		glm::vec2 dir {dx, dy};

		// Dragging only with the drag button
		// TODO: alt left dragging as ewll
		bool is_drag_button = (event.button == pan_button);
		if (event.action == GLFW_PRESS && is_drag_button)
			dragging = true;
		else if (event.action == GLFW_RELEASE && is_drag_button)
			dragging = false;

		// Pan only when draggign
		if (dragging) {
			yaw -= dx * sensitivity;
			pitch -= dy * sensitivity;

			if (pitch > 89.0f)
				pitch = 89.0f;
			if (pitch < -89.0f)
				pitch = -89.0f;

			cam.transform.rotation.x = pitch;
			cam.transform.rotation.y = yaw;
		}

		// Update previous position
		px = event.xpos;
		py = event.ypos;

		previous_dir = dir;
	}
};

int main()
{
	auto extensions = {
		VK_KHR_SWAPCHAIN_EXTENSION_NAME,
	};

	auto predicate = [&extensions](const vk::raii::PhysicalDevice &dev) {
		return physical_device_able(dev, extensions);
	};

	// Choose a physical device
	// TODO: static lambda (FIRST)
	auto phdev = pick_physical_device(predicate);

	/* auto camera = Camera {
		Transform { {2, 2, 6}, {-0.1, 0.3, 0} },
		Tunings { 45.0f, 800, 800 }
	}; */

	// Create the app and run it
	ECSApp app(phdev, extensions);
	// RTApp app(phdev, extensions);
	// engine::RTCapture app(phdev, {1000, 1000}, extensions, scene_path, camera);

	// Run the app
	app.run();
}

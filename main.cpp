#include "global.hpp"
#include "include/app.hpp"
#include "include/backend.hpp"
#include "include/button.hpp"
#include "include/common.hpp"
#include "include/ecs.hpp"
#include "include/engine/ecs_panel.hpp"
#include "include/engine/rt_capture.hpp"
#include "include/gui/button.hpp"
#include "include/gui/layer.hpp"
#include "include/gui/rect.hpp"
#include "include/gui/sprite.hpp"
#include "include/io/event.hpp"
#include "include/layers/font_renderer.hpp"
#include "include/layers/raster.hpp"
#include "include/layers/raytracer.hpp"
#include "include/layers/shape_renderer.hpp"
#include "include/renderer.hpp"
#include "include/transform.hpp"
#include "include/types.hpp"
#include "tinyfiledialogs.h"

using namespace kobra;

// Scene path
std::string scene_path = "scenes/room_simple.kobra";

// Test app
struct ECSApp : public BaseApp {
	layers::Raster	rasterizer;
	layers::Raytracer raytracer;
	layers::FontRenderer font_renderer;
	layers::ShapeRenderer shape_renderer;
	engine::ECSPanel panel;

	ECS ecs;

	Entity camera;
	Button button;

	ECSApp(const vk::raii::PhysicalDevice &phdev, const std::vector <const char *> &extensions)
			: BaseApp(phdev, "ECSApp", {1000, 1000}, extensions, vk::AttachmentLoadOp::eLoad),
			rasterizer(get_context(), vk::AttachmentLoadOp::eClear),
			raytracer(get_context(), &sync_queue, vk::AttachmentLoadOp::eClear),
			font_renderer(get_context(), render_pass, "resources/fonts/noto_sans.ttf"),
			shape_renderer(get_context(), render_pass),
			panel(get_context(), ecs, io) {
		// Add entities
		auto e1 = ecs.make_entity("box");
		auto e2 = ecs.make_entity("plane");
		auto e3 = ecs.make_entity("mirror");
		camera = ecs.make_entity("Camera");

		// Set transforms
		e2.get <Transform> ().position = {0, -2, 0};
		e2.get <Transform> ().scale = {5, 0.1, 5};

		e3.get <Transform> ().position = {-2, 1, 0};
		e3.get <Transform> ().scale = {1, 0.01, 1};
		e3.get <Transform> ().rotation = {0, 0, 70};

		// Add meshes to e1 and e2
		auto box = KMesh::make_box({0, 0, 0}, {1, 1, 1});
		auto submesh = Submesh(box.vertices(), box.indices());
		auto mesh = Mesh({submesh});

		e1.add <Mesh> (mesh);
		e2.add <Mesh> (mesh);
		e3.add <Mesh> (mesh);

		// Add rasterizers for e1 and e2
		e1.add <Rasterizer> (get_device(), e1.get <Mesh> ());
		e2.add <Rasterizer> (get_device(), e2.get <Mesh> ());

		e1.get <Rasterizer> ().mode = RasterMode::ePhong;
		e2.get <Rasterizer> ().mode = RasterMode::ePhong;

		e1.get <Rasterizer> ().material.Kd = {0.5, 0.5, 0.5};

		e2.get <Rasterizer> ().material.albedo_source = "resources/wood_floor_albedo.jpg";
		e2.get <Rasterizer> ().material.normal_source = "resources/wood_floor_normal.jpg";

		// Add raytracers for e1, e2 and e3
		e1.add <Raytracer> (&e1.get <Mesh> ());
		e2.add <Raytracer> (&e2.get <Mesh> ());
		e3.add <Raytracer> (&e3.get <Mesh> ());

		e1.get <Raytracer> ().material.Kd = {0.5, 0.5, 0.5};
		e2.get <Raytracer> ().material.Kd = {0.8, 0.6, 0.4};
		e3.get <Raytracer> ().material = {
			.Kd = {0.8, 0.8, 0.8},
			.type = eReflection
		};

		// Add camera
		auto c = Camera(Transform({0, 3, 10}), Tunings(45.0f, 1000, 1000));
		camera.add <Camera> (c);

		// Lights
		Entity light1 = ecs.make_entity("Light 1");
		Entity light2 = ecs.make_entity("Light 2");

		light1.get <Transform> ().position = {-5, 10, 0};
		light2.get <Transform> ().position = {5, 10, 0};

		light1.add <Light> (Light {.type = Light::Type::eArea, .color = {1, 0, 0}, .power = 8});
		light2.add <Light> (Light {.type = Light::Type::eArea, .color = {1, 1, 1}, .power = 3});

		// Add button
		auto handler = [](void *user) {
			std::cout << "Clicked" << std::endl;
		};

		auto args = Button::Args {
			.min = {400, 300},
			.max = {450, 350},

			.idle = {0.9, 0.7, 0.7},
			.hover = {0.7, 0.7, 0.9},
			.pressed = {0.7, 0.9, 0.7},

			.handlers = {{nullptr, handler}}
		};

		button = Button(io.mouse_events, args);

		// Input callbacks
		io.mouse_events.subscribe(mouse_callback, this);
	}

	void move_camera() {
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
	}

	float fps = 0;
	float time = 0;

	void record(const vk::raii::CommandBuffer &cmd,
			const vk::raii::Framebuffer &framebuffer) override {
		if (frame_time > 0)
			fps = (fps + 1.0f/frame_time) / 2.0f;

		std::vector <Text> texts {
			Text {
				.text = common::sprintf("%.2f fps", fps),
				.anchor = {10, 10},
				.size = 1.0f
			}
		};

		std::vector <Rect> rects {
			Rect {.min = {500, 500}, .max = {600, 600}, .color = {1, 0, 0}},
			button.shape()
		};

		time += frame_time;

		// Move camera
		move_camera();

		// Begin command buffer
		cmd.begin({});

		// rasterizer.render(cmd, framebuffer, ecs);
		raytracer.render(cmd, framebuffer, ecs);

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

		cmd.end();
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

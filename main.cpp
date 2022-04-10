#include "global.hpp"

// Scene path
std::string scene_path = "scene.kobra";

// Experimental GUI app
class GUIApp : public BaseApp {
	gui::Layer layer;

	// Elements
	gui::Text *text_expl;
	gui::Text *text_input;

	// Keyboard input
	// TODO: macro to generate signature
	static void keyboard_handler(void *user, const io::KeyboardEvent &event) {
		// Delete key delays
		static const int delete_key_delay_max = 2;
		static int delete_key_delay = 0;

		GUIApp *app = (GUIApp *) user;

		// If escape, then terminate
		if (event.key == GLFW_KEY_ESCAPE) {
			app->terminate_now();
			return;
		}

		// If the character is printable, then add it to the input
		bool pressed_only = (event.action == GLFW_PRESS);

		if (std::isprint(event.key) && pressed_only) {
			char c = std::tolower(event.key);
			if (event.mods == GLFW_MOD_SHIFT)
				c = std::toupper(c);

			app->text_input->str += c;
		}

		// Delete last character
		if (event.key == GLFW_KEY_BACKSPACE) {
			if (pressed_only) {
				app->text_input->str.pop_back();
				delete_key_delay = 0;
			} else {
				delete_key_delay++;
				if (delete_key_delay > delete_key_delay_max) {
					app->text_input->str.pop_back();
					delete_key_delay = 0;
				}
			}
		}
	}
public:
	GUIApp(Vulkan *vk) : BaseApp({
		vk,
		1000, 1000, 2,
		"GUI App"
	}, true) {
		// Initialize layer and load all fonts
		layer = gui::Layer(window, VK_ATTACHMENT_LOAD_OP_CLEAR);
		layer.load_font("default", "resources/fonts/noto_sans.ttf");

		// Create text
		text_expl = layer.text_render("default")->text(
			"Input: ",
			window.coordinates(10, 10),
			{1, 1, 1, 1}, 0.5
		);

		text_input = layer.text_render("default")->text(
			"",
			window.coordinates(80, 10),
			{1, 1, 1, 1}, 0.5
		);

		// Add elements
		layer.add(std::vector <gui::_element *> {
			text_expl,
			text_input
		});

		// Bind keyboard handler
		window.keyboard_events->subscribe(keyboard_handler, this);
	}

	void record(const VkCommandBuffer &cmd, const VkFramebuffer &framebuffer) override {
		Vulkan::begin(cmd);
		layer.render(cmd, framebuffer);
		Vulkan::end(cmd);
	}
};

int main()
{
	Vulkan *vulkan = new Vulkan();

	// GUIApp gui_app {vulkan};
	RTApp main_app {vulkan};
	ProfilerApp profiler_app {vulkan, &Profiler::one()};

	std::thread t1 {
		[&]() {
			main_app.run();
		}
	};

	std::thread t2 {
		[&]() {
			profiler_app.run();
		}
	};

	t1.join();

	profiler_app.terminate_now();
	t2.join();

	delete vulkan;
}

////////////////////
// Input handlers //
////////////////////

// Keyboard listener
void RTApp::keyboard_handler(void *user, const io::KeyboardEvent &event)
{
	RTApp *app = (RTApp *) user;

	if (event.action == GLFW_PRESS) {
		// Tab
		if (event.key == GLFW_KEY_TAB)
			app->raster = !app->raster;

		// 1, 2, 3 to switch rasterization mode
		if (event.key == GLFW_KEY_1)
			app->raster_layer.set_mode(raster::Layer::Mode::ALBEDO);
		if (event.key == GLFW_KEY_2)
			app->raster_layer.set_mode(raster::Layer::Mode::NORMAL);
		if (event.key == GLFW_KEY_3)
			app->raster_layer.set_mode(raster::Layer::Mode::BLINN_PHONG);

		// Editting mode
		if (event.key == GLFW_KEY_G)
			app->edit_mode = !app->edit_mode;

		// Selecting objects
		if (app->edit_mode) {
			if (app->highlight == -1)
				app->highlight = 0;

			// Left and right arrow keys
			int count = 0;
			if (event.key == GLFW_KEY_LEFT_BRACKET) {
				app->highlight = (app->highlight + 1)
					% app->rt_layer.size();
			}

			if (event.key == GLFW_KEY_RIGHT_BRACKET) {
				app->highlight = (app->highlight
					+ app->rt_layer.size() - 1)
					% app->rt_layer.size();
			}

			// Up down left right to select
			// transform property
			if (event.key == GLFW_KEY_UP)
				app->transform_row = (app->transform_row + 1) % 3;

			if (event.key == GLFW_KEY_DOWN)
				app->transform_row = (app->transform_row + 2) % 3;

			if (event.key == GLFW_KEY_LEFT)
				app->transform_col = (app->transform_col + 2) % 3;

			if (event.key == GLFW_KEY_RIGHT)
				app->transform_col = (app->transform_col + 1) % 3;

			// Reset object transforms
			if (event.key == GLFW_KEY_R) {
				auto eptr = app->raster_layer[app->highlight];
				eptr->transform() = Transform();
			}
		}

		// Saving the scene with Ctrl
		if (event.key == GLFW_KEY_K &&
			event.mods == GLFW_MOD_CONTROL) {
			KOBRA_LOG_FILE(notify) << "Saving scene...\n";
			app->scene.save(scene_path);
		}

		// Refresh rasterization
		if (event.key == GLFW_KEY_R &&
			event.mods == GLFW_MOD_CONTROL) {
			KOBRA_LOG_FILE(notify) << "Force refreshing scene...\n";
			app->scene.save(scene_path);
			app->scene = Scene(app->context, app->window.command_pool, scene_path);
			app->raster_layer = raster::Layer(app->window, VK_ATTACHMENT_LOAD_OP_CLEAR);
			app->raster_layer.add_scene(app->scene);
			app->raster_layer.set_active_camera(app->camera);
			app->raster_layer.set_mode(raster::Layer::Mode::BLINN_PHONG);
		}

		// Toggle mouse visibility
		if (event.key == GLFW_KEY_M) {
			app->show_mouse = !app->show_mouse;
			app->window.cursor_mode(
				app->show_mouse ? GLFW_CURSOR_NORMAL
					: GLFW_CURSOR_HIDDEN
			);
		}

		// Start a capture
		if (event.key == GLFW_KEY_C) {
			app->capturer = new engine::RTCapture(
				app->context.vk,
				scene_path,
				app->camera
			);

			app->capturer_thread = new std::thread([&]() {
				app->capturer->run();
			});
		}
	}
}

// Mouse movement
void RTApp::mouse_movement(void *user, const io::MouseEvent &event)
{
	static const int drag_button = GLFW_MOUSE_BUTTON_MIDDLE;

	static const float sensitivity = 0.001f;

	static bool first_movement = true;
	static bool dragging = false;

	static float px = 0.0f;
	static float py = 0.0f;

	static float yaw = 0.0f;
	static float pitch = 0.0f;

	// Dragging only with the drag button
	bool is_drag_button = (event.button == drag_button);
	if (event.action == GLFW_PRESS && is_drag_button)
		dragging = true;
	else if (event.action == GLFW_RELEASE && is_drag_button)
		dragging = false;

	// Only if dragging
	if (dragging) {
		float dx = event.xpos - px;
		float dy = event.ypos - py;

		Camera *camera = (Camera *) user;

		yaw -= dx * sensitivity;
		pitch -= dy * sensitivity;

		if (pitch > 89.0f)
			pitch = 89.0f;
		if (pitch < -89.0f)
			pitch = -89.0f;

		camera->transform.rotation.x = pitch;
		camera->transform.rotation.y = yaw;
	}

	// Update previous position
	px = event.xpos;
	py = event.ypos;
}

#include "global.hpp"

// Scene path
std::string scene_path = "scene.kobra";

int main()
{
	Vulkan *vulkan = new Vulkan();

	RTApp rt_app {vulkan};
	// ProfilerApplication papp {vulkan, &Profiler::one()};

	std::thread t1 {
		[&rt_app]() {
			rt_app.run();
		}
	};

	/* std::thread t2 {
		[&papp]() {
			papp.run();
		}
	}; */

	t1.join();
	// t2.join();	// TODO: terminate now method

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
	static const int drag_button = GLFW_MOUSE_BUTTON_LEFT;

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

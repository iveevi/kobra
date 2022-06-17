#include "global.hpp"
#include "include/app.hpp"
#include "include/backend.hpp"
#include "include/gui/button.hpp"
#include "include/gui/layer.hpp"
#include "include/gui/rect.hpp"
#include "include/gui/sprite.hpp"
#include "tinyfiledialogs.h"

using namespace kobra;

// Scene path
std::string scene_path = "scenes/room.kobra";

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
	GUIApp(vk::raii::PhysicalDevice &phdev_,
			const vk::Extent2D &extent_,
			const std::vector <const char *> &extensions_)
			: BaseApp(phdev_, extent_, extensions_) {
		layer = std::move(gui::Layer(phdev, device,
			command_pool, descriptor_pool, extent,
			swapchain.format, depth_buffer.format,
			vk::AttachmentLoadOp::eClear
		));

		// Load all fonts
		// TODO: later add layouts to tightly pack text and other gui elements
		layer.load_font("default", "resources/fonts/noto_sans.ttf");

		// Create text
		std::cout << "Coords: " << coordinates(10, 10).x << " " << coordinates(10, 10).y << std::endl;
		text_expl = layer.text_render("default")->text(
			"Input: ",
			coordinates(100, 10),
			{1, 1, 1, 1}, 1
		);

		text_input = layer.text_render("default")->text(
			"",
			coordinates(220, 10),
			{1, 1, 1, 1}, 1
		);

		// Add elements
		layer.add(std::vector <gui::_element *> {
			text_expl,
			text_input
		});

		// Bind keyboard handler
		io.keyboard_events.subscribe(keyboard_handler, this);

		// Rectangle
		auto rect = new gui::Rect(phdev, device,
			coordinates(0, 0),
			coordinates(100, 100),
			{1, 1, 1}
		);

		// layer.add(rect);

		// Button
		auto button_info = gui::Button::RectButton {
			.pos = coordinates(400, 400),
			.size = coordinates(100, 100),

			.button = GLFW_MOUSE_BUTTON_LEFT,

			.idle = {0.5, 0.5, 1},
			.hover = {1, 0.5, 0.5},
			.active = {1, 1, 1}
		};

		auto button = new gui::Button(phdev, device, io, button_info);

		layer.add(button);

		// Sprite
		auto sprite = new gui::Sprite(phdev, device,
			command_pool,
			coordinates(0, 0),
			coordinates(100, 100),
			"resources/icons/mesh.png"
		);

		layer.add(sprite);
	}

	void record(const vk::raii::CommandBuffer &cmd,
			const vk::raii::Framebuffer &framebuffer) override {
		cmd.begin({});
		layer.render(cmd, framebuffer);
		cmd.end();
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

	// Create a GUI app
	RTApp app(phdev, extensions);

	// Run the app
	app.run();
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
		if (event.key == GLFW_KEY_1) {
			app->raster_layer.set_mode(raster::Layer::Mode::ALBEDO);
			app->raster = true;
		}

		if (event.key == GLFW_KEY_2) {
			app->raster_layer.set_mode(raster::Layer::Mode::NORMAL);
			app->raster = true;
		}

		if (event.key == GLFW_KEY_3) {
			app->raster_layer.set_mode(raster::Layer::Mode::BLINN_PHONG);
			app->raster = true;
		}

		if (event.key == GLFW_KEY_4) {
			app->rt_layer.set_mode(rt::Layer::NORMALS);
			app->raster = false;
		}

		if (event.key == GLFW_KEY_0) {
			app->rt_layer.set_mode(rt::Layer::HEATMAP);
			app->raster = false;
		}

		if (event.key == GLFW_KEY_5) {
			app->rt_layer.set_mode(rt::Layer::FAST_PATH_TRACER);
			app->raster = false;
		}

		if (event.key == GLFW_KEY_6) {
			app->rt_layer.set_mode(rt::Layer::PATH_TRACER);
			app->raster = false;
		}

		if (event.key == GLFW_KEY_7) {
			app->rt_layer.set_mode(rt::Layer::MIS_PATH_TRACER);
			app->raster = false;
		}

		if (event.key == GLFW_KEY_8) {
			app->rt_layer.set_mode(rt::Layer::BIDIRECTIONAL_PATH_TRACE);
			app->raster = false;
		}

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
			// app->scene.save(scene_path);
			app->save(scene_path);
			app->gizmo_handle->deselect();
		}

		// Importing scenes with Ctrl+O
		if (event.key == GLFW_KEY_O &&
			event.mods == GLFW_MOD_CONTROL) {
			KOBRA_LOG_FILE(notify) << "Importing scene...\n";
			// app->scene.load(scene_path);

			// Load an object
			std::string str = tinyfd_openFileDialog(
				"Open scene", scene_path.c_str(),
				0, 0, 0, 0
			);

			std::cout << "Selected file: " << str << std::endl;
			app->add_mesh(str);
		}

		// Duplicating objects with Ctrl+P
		if (event.key == GLFW_KEY_P &&
			event.mods == GLFW_MOD_CONTROL) {
			// First update the scene
			app->update_scene();

			// Get currently selected object
			auto eptr = app->gizmo_handle->get_object();
			auto name = app->duplicate_object(eptr->name());

			app->raster_layer.clear_highlight();

			auto ptr = app->raster_layer[name];
			ptr->highlight = true;

			glm::vec3 position = ptr->center();
			app->gizmo_handle->set_position(position);
			app->gizmo_handle->bind(ptr);
		}

		// Delete objects with X
		if (event.key == GLFW_KEY_X && app->edit.selected != nullptr) {
			auto name = app->edit.selected->name();

			std::cout << "Deleting object: " << name << std::endl;
			app->raster_layer.erase(name);
			app->scene.erase(name);

			app->edit.selected = nullptr;
			app->gizmo_handle->deselect();
		}

		// Refresh rasterization
		if (event.key == GLFW_KEY_R &&
			event.mods == GLFW_MOD_CONTROL) {
			KOBRA_LOG_FILE(notify) << "Force refreshing scene...\n";
			throw int(1);

			/* app->save(scene_path);
			app->scene = Scene(app->phdev, app->device, app->command_pool, scene_path);
			app->raster_layer = raster::Layer(app->phdev, app->device,
				app->command_pool, app->descriptor_pool,
				app->extent, app->swapchain.format,
				app->depth_buffer.format,
				vk::AttachmentLoadOp::eLoad
			);

			app->raster_layer.add_scene(app->scene);
			app->raster_layer.set_active_camera(app->camera);
			app->raster_layer.set_mode(raster::Layer::Mode::BLINN_PHONG);
			app->gizmo_handle->deselect(); */
		}

		// Toggle mouse visibility
		if (event.key == GLFW_KEY_M) {
			app->show_mouse = !app->show_mouse;
			app->window.set_cursor_mode(
				app->show_mouse ? GLFW_CURSOR_NORMAL
					: GLFW_CURSOR_HIDDEN
			);
		}

		// Start a capture
		if (event.key == GLFW_KEY_C) {
			// TODO: store extensions in use
			app->capturer = new engine::RTCapture(
				app->phdev,
				app->extent,
				{},
				scene_path,
				app->camera
			);

			app->capturer_thread = new std::thread([&]() {
				app->capturer->run();
			});
		}

		// R and T to switch gizmo type
		if (event.key == GLFW_KEY_T)
			app->edit.gizmo_mode = 1;

		if (event.key == GLFW_KEY_R)
			app->edit.gizmo_mode = 2;
	}
}

// Mouse movement
void RTApp::mouse_movement(void *user, const io::MouseEvent &event)
{
	// TODO: refactor to pan
	static const int drag_button = GLFW_MOUSE_BUTTON_MIDDLE;
	static const int alt_drag_button = GLFW_MOUSE_BUTTON_LEFT;
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

	RTApp *app = (RTApp *) user;

	// Deltas and directions
	float dx = event.xpos - px;
	float dy = event.ypos - py;
	glm::vec2 dir {dx, dy};

	// Is alt pressed?
	bool alt = app->io.input.is_key_down(GLFW_KEY_LEFT_ALT);

	// Dragging only with the drag button
	// TODO: alt left dragging as ewll
	bool is_drag_button = (event.button == drag_button);
	if (event.action == GLFW_PRESS && is_drag_button)
		dragging = true;
	else if (event.action == GLFW_RELEASE && is_drag_button)
		dragging = false;

	// Dragging select
	bool is_select_button = (event.button == select_button) && !alt;
	if (event.action == GLFW_PRESS && is_select_button)
		dragging_select = true;
	else if (event.action == GLFW_RELEASE && is_select_button)
		dragging_select = false;

	// Clicking (shoots a ray)
	if (event.action == GLFW_PRESS && event.button == select_button && !alt) {
		// Convert to window coordinates
		glm::mat4 proj = app->camera.projection();

		// Clicked coordinates
		float x = event.xpos;
		float y = (app->extent.height - event.ypos);

		bool activated = app->gizmo_handle->handle_select(proj,
			x, y, app->extent.width,
			app->extent.height
		);

		// Rotation gizmo
		float _x = event.xpos / (float) app->extent.width;
		float _y = event.ypos / (float) app->extent.height;
		Ray ray = app->camera.generate_ray(_x, _y);

		auto d_x = closest_distance(*app->edit.gizmo_x, ray);
		auto d_y = closest_distance(*app->edit.gizmo_y, ray);
		auto d_z = closest_distance(*app->edit.gizmo_z, ray);

		int min_axis = -1;
		float min_distance = -1.0f;

		if (d_x.distance < d_y.distance && d_x.distance < d_z.distance) {
			min_axis = 2;
			min_distance = d_x.distance;
		} else if (d_y.distance < d_z.distance && d_y.distance < d_x.distance) {
			min_axis = 1;
			min_distance = d_y.distance;
		} else if (d_z.distance < d_x.distance && d_z.distance < d_y.distance) {
			min_axis = 0;
			min_distance = d_z.distance;
		}

		if (app->edit.gizmo_mode == 1 && app->gizmo_handle->get_object() != nullptr && activated) {
			gizmo_dragging = true;
		} else if (app->edit.gizmo_mode == 2 && app->edit.selected != nullptr && min_distance < 0.1f) {
			gizmo_dragging = true;
			app->edit.rot_axis = min_axis;
		} else {
			// Select an object instead
			float x = event.xpos / (float) app->extent.width;
			float y = event.ypos / (float) app->extent.height;
			Ray ray = app->camera.generate_ray(x, y);

			float t = std::numeric_limits <float> ::max();
			std::string name = "";

			for (auto &obj : app->raster_layer) {
				float t_ = obj->intersect(ray);
				if (t_ >= 0.0 && t_ < t) {
					t = t_;
					name = obj->name();
				}
			}

			// TODO: also poppup window with info (name, transform,
			// material, ect)
			// TODO: make a poppup info window template for
			// materials
			if (name.length() > 0) {
				app->raster_layer.clear_highlight();

				auto ptr = app->raster_layer[name];
				ptr->highlight = true;

				glm::vec3 position = ptr->center();
				app->gizmo_handle->set_position(position);
				app->gizmo_handle->bind(ptr);
				app->edit.selected = ptr;
			} else {
				auto ptr = app->gizmo_handle->get_object();
				if (ptr != nullptr) {
					app->raster_layer[ptr->name()]->highlight = false;
					app->gizmo_handle->deselect();
					app->edit.selected = nullptr;
				}
			}
		}
	} else if (event.action == GLFW_RELEASE && event.button == select_button) {
		gizmo_dragging = false;
	}

	// Dragging in gizmo for translation
	if (dragging_select && gizmo_dragging && app->gizmo_handle->get_object() != nullptr
			&& app->edit.gizmo_mode == 1) {
		glm::mat4 proj = app->camera.projection();
		app->gizmo_handle->handle_drag(proj,
			dx, -dy,
			app->extent.width, app->extent.height
		);
	}

	// Dragging in gizmo for rotation
	if (dragging_select && gizmo_dragging
			&& app->gizmo_handle->get_object() != nullptr
			&& app->edit.gizmo_mode == 2) {
		glm::vec3 pos = app->edit.selected->transform().position;

		float xp = (event.xpos - dx) / (float) app->extent.width;
		float yp = (event.ypos - dy) / (float) app->extent.height;

		float x = event.xpos / (float) app->extent.width;
		float y = event.ypos / (float) app->extent.height;

		Ray ray0 = app->camera.generate_ray(xp, yp);
		Ray ray1 = app->camera.generate_ray(x, y);

		// Closest points to pos
		glm::vec3 oc0 = closest_point(ray0, pos);
		glm::vec3 oc1 = closest_point(ray1, pos);

		glm::vec3 c0 = closest_point(ray0, pos) - pos;
		glm::vec3 c1 = closest_point(ray1, pos) - pos;

		float degrees = 0;

		// TODO: fix duplicate code
		if (app->edit.rot_axis == 0) {
			// Project onto yz plane
			glm::vec3 pc0 = glm::normalize(glm::vec3(0, c0.y, c0.z));
			glm::vec3 pc1 = glm::normalize(glm::vec3(0, c1.y, c1.z));

			float dot = glm::dot(pc0, pc1);

			if (std::fabs(dot - 1.0f) > 1e-6) {
				degrees = glm::acos(dot);
				degrees *= (glm::cross(pc0, pc1).x < 0 ? -1 : 1);
				degrees = degrees * 180.0f / glm::pi<float>();
			}
		} else if (app->edit.rot_axis == 1) {
			// Project onto xz plane
			glm::vec3 pc0 = glm::normalize(glm::vec3(c0.x, 0, c0.z));
			glm::vec3 pc1 = glm::normalize(glm::vec3(c1.x, 0, c1.z));

			float dot = glm::dot(pc0, pc1);
			if (std::fabs(dot - 1.0f) > 1e-6) {
				degrees = glm::acos(dot);
				degrees *= (glm::cross(pc0, pc1).y < 0 ? -1 : 1);
				degrees *= 180.0f / glm::pi<float>();
			}
		} else if (app->edit.rot_axis == 2) {
			// Project onto xy plane
			glm::vec3 pc0 = glm::normalize(glm::vec3(c0.x, c0.y, 0));
			glm::vec3 pc1 = glm::normalize(glm::vec3(c1.x, c1.y, 0));

			float dot = glm::dot(pc0, pc1);

			if (std::fabs(dot - 1.0f) > 1e-6) {
				degrees = glm::acos(dot);
				degrees *= (glm::cross(pc0, pc1).z < 0 ? -1 : 1);
				degrees = degrees * 180.0f / glm::pi<float>();
			}
		}

		app->edit.selected->transform()
			.rotation[app->edit.rot_axis] += degrees;
	}

	// Only if dragging
	if (dragging || alt) {
		Camera &camera = app->camera;

		yaw -= dx * sensitivity;
		pitch -= dy * sensitivity;

		if (pitch > 89.0f)
			pitch = 89.0f;
		if (pitch < -89.0f)
			pitch = -89.0f;

		camera.transform.rotation.x = pitch;
		camera.transform.rotation.y = yaw;
	}

	// Update previous position
	px = event.xpos;
	py = event.ypos;

	previous_dir = dir;
}

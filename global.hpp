#ifndef GLOBAL_H_
#define GLOBAL_H_

// Standard headers
#include <cstring>
#include <iostream>
#include <thread>
#include <vulkan/vulkan_core.h>

// Engine macros
// #define KOBRA_VALIDATION_LAYERS
#define KOBRA_ERROR_ONLY
#define KOBRA_THROW_ERROR

// Engine headers
#include "include/app.hpp"
#include "include/backend.hpp"
#include "include/capture.hpp"
#include "include/engine/gizmo.hpp"
#include "include/engine/rt_capture.hpp"
#include "include/gui/gui.hpp"
#include "include/gui/layer.hpp"
#include "include/gui/rect.hpp"
#include "include/io/event.hpp"
#include "include/model.hpp"
#include "include/profiler.hpp"
#include "include/raster/layer.hpp"
#include "include/raytracing/layer.hpp"
#include "include/raytracing/mesh.hpp"
#include "include/raytracing/sphere.hpp"
#include "include/scene.hpp"
#include "include/types.hpp"
#include "profiler.hpp"

using namespace kobra;

// Scene path
extern std::string scene_path;

// TODO: focus on adding objects to the scene (spheres, boxes, models)
// TODO: RT sampling sphere lights
// TODO: input window for RT Capture (etner # of samples, ect)

// Main class
class RTApp :  public BaseApp {
	// Application camera
	Camera		camera;

	// RT or Raster
	bool			raster = true;
	bool			modified = false;
	bool			show_mouse = false;

	// Layers
	rt::Layer		rt_layer;
	raster::Layer		raster_layer;
	gui::Layer		gui_layer;

	// Capturer
	engine::RTCapture	*capturer = nullptr;
	std::thread		*capturer_thread = nullptr;

	// Current scene
	Scene			scene;

	// Raster state
	int			highlight = -1;
	bool			edit_mode = false;

	// Raytracing state
	rt::Batch		batch;
	rt::BatchIndex		batch_index;

	// GUI state
	// TODO: gui window abstraction (uses screen info returns as well)
	struct {
		// Statistics
		gui::Text	*text_frame_rate;
		gui::Text	*layer_info;
		gui::Rect	*stats_bg;

		// Help
		gui::Text	*text_help;
		gui::Rect	*help_bg;

		// Editing
		gui::Text	*text_position;
		gui::Text	*text_rotation;
		gui::Text	*text_scale;
		gui::Rect	*edit_bg;
	} gui;

	// Initialize GUI elements
	void initialize_gui() {
		// TODO: rounded corners
		// TODO: color type

		// TODO: method to add gui elements
		// TODO: add_scene for gui layer

		// Fonts
		gui_layer.load_font("default", "resources/fonts/noto_sans.ttf");

		// Statistics
		gui.text_frame_rate = gui_layer.text_render("default")->text(
			"fps",
			window.coordinates(10, 10),
			{1, 1, 1, 1}
		);

		gui.layer_info = gui_layer.text_render("default")->text(
			"",
			window.coordinates(10, 60),
			{1, 1, 1, 1}
		);

		gui.stats_bg = new gui::Rect(
			window.coordinates(0, 0),
			window.coordinates(0, 0),
			{0.4, 0.4, 0.4}
		);

		// TODO: direct method to add gui elements
		gui.stats_bg->children.push_back(gui::Element(gui.text_frame_rate));
		gui.stats_bg->children.push_back(gui::Element(gui.layer_info));

		glm::vec4 bounds = gui::get_bounding_box(gui.stats_bg->children);
		bounds += 0.01f * glm::vec4 {-1, -1, 1, 1};
		gui.stats_bg->set_bounds(bounds);

		gui_layer.add(gui.stats_bg);

		// Help
		gui.text_help = gui_layer.text_render("default")->text(
			"[h] Help, [c] Capture, [g] Edit objects, [tab] Toggle RT preview",
			window.coordinates(10, 750),
			{1, 1, 1, 1}, 0.5
		);

		gui.help_bg = new gui::Rect(
			window.coordinates(0, 0),
			window.coordinates(0, 0),
			{0.4, 0.4, 0.4}
		);

		gui.help_bg->children.push_back(gui::Element(gui.text_help));
		bounds = gui::get_bounding_box(gui.help_bg->children);
		// bounds += 0.01f * glm::vec4 {1, 1, -1, -1};
		gui.help_bg->set_bounds(bounds);

		gui_layer.add(gui.help_bg);

		// Editing
		gui.text_position = gui_layer.text_render("default")->text(
			"Position: (0, 0, 0)",
			window.coordinates(450, 630),
			{1, 1, 1, 1}, 0.5
		);

		gui.text_rotation = gui_layer.text_render("default")->text(
			"Rotation: (0, 0, 0)",
			window.coordinates(450, 650),
			{1, 1, 1, 1}, 0.5
		);

		gui.text_scale = gui_layer.text_render("default")->text(
			"Scale: (1, 1, 1)",
			window.coordinates(450, 670),
			{1, 1, 1, 1}, 0.5
		);

		gui.edit_bg = new gui::Rect(
			window.coordinates(0, 0),
			window.coordinates(0, 0),
			{0.4, 0.4, 0.4}
		);

		gui.edit_bg->children.push_back(gui::Element(gui.text_position));
		gui.edit_bg->children.push_back(gui::Element(gui.text_rotation));
		gui.edit_bg->children.push_back(gui::Element(gui.text_scale));

		bounds = gui::get_bounding_box(gui.edit_bg->children);
		bounds += 0.01f * glm::vec4 {-1, -1, 1, 1};
		gui.edit_bg->set_bounds(bounds);

		gui_layer.add(gui.edit_bg);
	}

	// Update GUI elements
	void update_gui() {
		static char buffer[1024];

		// Overlay statistics
		// TODO: statistics should be made a standalone layer
		std::sprintf(buffer, "time: %.2f ms, fps: %.2f",
			1000 * frame_time,
			1.0f/frame_time
		);

		gui.text_frame_rate->str = buffer;

		// RT layer statistics
		gui.layer_info->str = std::to_string(rt_layer.triangle_count()) + " triangles";

		// Update bounds
		glm::vec4 bounds = gui::get_bounding_box(gui.stats_bg->children);
		bounds += 0.01f * glm::vec4 {-1, -1, 1, 1};
		gui.stats_bg->set_bounds(bounds);

		// Edit mode
		if (edit_mode) {
			auto eptr = raster_layer[highlight];
			auto transform = eptr->transform();

			// Update text
			std::sprintf(buffer, "Position: (%.2f, %.2f, %.2f)",
				transform.position.x,
				transform.position.y,
				transform.position.z
			);

			gui.text_position->str = buffer;

			std::sprintf(buffer, "Rotation: (%.2f, %.2f, %.2f)",
				transform.rotation.x,
				transform.rotation.y,
				transform.rotation.z
			);

			gui.text_rotation->str = buffer;

			std::sprintf(buffer, "Scale: (%.2f, %.2f, %.2f)",
				transform.scale.x,
				transform.scale.y,
				transform.scale.z
			);

			gui.text_scale->str = buffer;

			// Update bounds
			bounds = gui::get_bounding_box(gui.edit_bg->children);
			bounds += 0.01f * glm::vec4 {-1, -1, 1, 1};
			gui.edit_bg->set_bounds(bounds);
		}
	}

	// Input handling for edit mode
	// TODO: text to indicate mode and name of selected object
	int transform_row = 0;
	int transform_col = 0;

	void handle_edit_mode() {
		// Assume edit mode
		auto eptr = raster_layer[highlight];
		auto &transform = eptr->transform();
		float speed = 0.01f;

		// TODO: determine a reference to a float
		float *fptr = nullptr;
		if (transform_row == 0) {
			fptr = &transform.position[transform_col];
		} else if (transform_row == 1) {
			fptr = &transform.rotation[transform_col];
			speed = 0.1f;
		} else if (transform_row == 2) {
			fptr = &transform.scale[transform_col];
		}

		if (input.is_key_down('=')) {
			*fptr += speed;
			modified = true;
		} else if (input.is_key_down('-')) {
			*fptr -= speed;
			modified = true;
		}

		// Save with corresponding scene object
		std::string name = eptr->name();
		scene[name]->transform() = transform;
	}

	// Save element currently in raster_layer
	void save(const std::string &path) {
		for (auto &e : raster_layer) {
			scene[e->name()]->transform() = e->transform();
			// TODO: later will need to copy material, etc
		}

		scene.save(path);
	}

	// Keyboard handler
	static void keyboard_handler(void *, const io::KeyboardEvent &);

	// Mouve camera
	static void mouse_movement(void *, const io::MouseEvent &);

	// Gizmo things
	engine::Gizmo::Handle		gizmo_handle;
	engine::Gizmo			gizmo_set;
public:
	// TODO: just remove the option of no depth buffer (always use depth buffer)
	RTApp(Vulkan *vk) : BaseApp({
		vk,
		1000, 1000, 2,
		"RT App"
	}) {
		// Construct camera
		camera = Camera {
			Transform { {0, 6, 16}, {-0.2, 0, 0} },
			Tunings { 45.0f, 800, 800 }
		};

		// Load scene
		// create_scene();
		Profiler::one().frame("Loading scene");
		scene = Scene(context, window.command_pool, scene_path);
		Profiler::one().end();

		for (auto &obj : scene)
			std::cout << "Scene object: " << obj->name() << std::endl;

		///////////////////////
		// Initialize layers //
		///////////////////////

		// Ray tracing layer
		rt_layer = rt::Layer(window);
		rt_layer.add_scene(scene);

		// Initialize batch and batch index
		batch = rt::Batch(1000, 1000, 50, 50, 1);
		batch_index = batch.make_batch_index(0, 0);

		// TODO: m8 gotta really fix auto channels
		rt_layer.set_environment_map(
			load_image_texture("resources/skies/background_3.jpg", 4)
		);

		// Rasterization layer
		raster_layer = raster::Layer(window, VK_ATTACHMENT_LOAD_OP_CLEAR);
		raster_layer.set_mode(raster::Layer::Mode::BLINN_PHONG);
		raster_layer.add_scene(scene);

		// GUI layer
		// TODO: be able to load gui elements from scene
		gui_layer = gui::Layer(window, VK_ATTACHMENT_LOAD_OP_LOAD);
		initialize_gui();

		// Gizmo layer
		// TODO: all layer type constructor should take a mandatory load
		// operation
		gizmo_set = engine::Gizmo(window, VK_ATTACHMENT_LOAD_OP_LOAD);
		gizmo_handle = gizmo_set.transform_gizmo();

		// Add event listeners
		window.keyboard_events->subscribe(keyboard_handler, this);
		window.mouse_events->subscribe(mouse_movement, this);

		// Show results of profiling
		auto frame = Profiler::one().pop();
		// std::cout << Profiler::pretty(frame);
	}

	// Destructor
	~RTApp() {
		if (capturer) {
			capturer->terminate_now();
			capturer_thread->join();
			delete capturer_thread;
			delete capturer;
		}
	}

	// Override record method
	// TODO: preview raytraced scene with a very low resolution
	void record(const VkCommandBuffer &cmd, const VkFramebuffer &framebuffer) override {
		static float time = 0.0f;

		// Start recording command buffer
		Vulkan::begin(cmd);

		// WASDEQ movement
		// TODO: method
		float speed = 20.0f * frame_time;

		glm::vec3 forward = camera.transform.forward();
		glm::vec3 right = camera.transform.right();
		glm::vec3 up = camera.transform.up();

		if (input.is_key_down(GLFW_KEY_W))
			camera.transform.move(forward * speed);
		else if (input.is_key_down(GLFW_KEY_S))
			camera.transform.move(-forward * speed);

		if (input.is_key_down(GLFW_KEY_A))
			camera.transform.move(-right * speed);
		else if (input.is_key_down(GLFW_KEY_D))
			camera.transform.move(right * speed);

		if (input.is_key_down(GLFW_KEY_E))
			camera.transform.move(up * speed);
		else if (input.is_key_down(GLFW_KEY_Q))
			camera.transform.move(-up * speed);

		// Input
		if (edit_mode)
			handle_edit_mode();

		// Copy camera contents to each relevant layer
		rt_layer.set_active_camera(camera);
		raster_layer.set_active_camera(camera);
		gizmo_set.set_camera(camera);

		/* Highlight appropriate object
		raster_layer.clear_highlight();
		if (edit_mode)
			raster_layer.set_highlight(highlight, true); */

		// Render appropriate layer
		if (raster) {
			raster_layer.render(cmd, framebuffer);
		} else {
			if (modified) {
				// TODO: fix clear method
				// rt_layer.clear();
				// rt_layer.add_scene(scene);
				KOBRA_LOG_FILE(notify) << "Reconstructing RT layer\n";
				scene.save(scene_path);
				scene = Scene(context, window.command_pool, scene_path);

				rt_layer = rt::Layer(window);
				rt_layer.add_scene(scene);
				rt_layer.set_active_camera(camera);
				modified = false;
			}

			rt_layer.render(cmd, framebuffer, batch_index);
			batch.increment(batch_index);
		}

		// Render GUI
		// update_gui();
		// gui_layer.render(cmd, framebuffer);

		// Render gizmo
		if (gizmo_handle->get_object() != nullptr)
			gizmo_set.render(cmd, framebuffer);

		// End recording command buffer
		Vulkan::end(cmd);

		// Update time
		time += frame_time;
	}

	// Termination method
	void terminate() override {
		// Check input
		if (window.input->is_key_down(GLFW_KEY_ESCAPE))
			glfwSetWindowShouldClose(surface.window, true);
	}
};

#endif

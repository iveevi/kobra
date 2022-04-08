// Standard headers
#include <cstring>
#include <iostream>
#include <thread>
#include <vulkan/vulkan_core.h>

// Engine macros
#define KOBRA_VALIDATION_LAYERS
#define KOBRA_ERROR_ONLY
#define KOBRA_THROW_ERROR

// Engine headers
#include "include/app.hpp"
#include "include/backend.hpp"
#include "include/capture.hpp"
#include "include/gui/gui.hpp"
#include "include/gui/layer.hpp"
#include "include/gui/rect.hpp"
#include "include/io/event.hpp"
#include "include/model.hpp"
#include "include/raster/layer.hpp"
#include "include/raytracing/layer.hpp"
#include "include/raytracing/mesh.hpp"
#include "include/raytracing/sphere.hpp"
#include "include/scene.hpp"
#include "include/types.hpp"

using namespace kobra;

// RT capture class
// TODO: turn in an engine module
class RTCapture : public BaseApp {
	Camera		camera;
	rt::Layer	layer;
	rt::Batch	batch;
	rt::BatchIndex	index;
	bool		term = false;
public:
	// Constructor from scene file and camera
	RTCapture(Vulkan *vk, const char *scene_file, const Camera &camera)
			: BaseApp({
				vk,
				800, 800, 2,
				"RT Capture",
			}, true),
			camera(camera) {
		// Load scene
		Scene scene = Scene(context, window.command_pool, scene_file);

		// Create raster layer
		layer = rt::Layer(window);
		layer.add_scene(scene);
		layer.set_active_camera(camera);
		layer.set_environment_map(
			load_image_texture("resources/skies/background_3.jpg", 4)
		);

		// Create batch
		// TODO: a method to generate optimal batch sizes (eg 50x50 is
		// faster than 10x10)
		batch = rt::Batch(800, 800, 100, 100, 1);
		index = batch.make_batch_index(0, 0, 16, 1);
	}

	// Render loop
	void record(const VkCommandBuffer &cmd, const VkFramebuffer &framebuffer) override {
		static float time = 0.0f;

		// Start recording command buffer
		Vulkan::begin(cmd);

		// Render scene
		layer.render(cmd, framebuffer, index);

		// End recording command buffer
		Vulkan::end(cmd);

		// Track progress
		time += frame_time;
		float progress = batch.progress();
		float eta = time * (1.0f - progress) / progress;
		std::printf("Progress: %.2f%%, Total time: %.2fs (+%.2fs), ETA: %.2fs\n",
			progress * 100.0f, time, frame_time, eta);

		// Next batch
		batch.increment(index);
	}

	// Treminator
	void terminate() override {
		bool b = batch.completed();
		if (term || b) {
			glfwSetWindowShouldClose(surface.window, GLFW_TRUE);
			auto buffer = layer.pixels();

			// TODO: make an easier an more straight forward way to
			// save a buffer to an image
			Image img {
				.width = 800,
				.height = 800
			};

			Capture::snapshot(buffer, img);
			img.write("capture.png");
			KOBRA_LOG_FUNC(notify) << "Capture saved to capture.png\n";
		}
	}

	// Terminate now
	// TODO: base app should have a terminate function
	void terminate_now() {
		term = true;
	}
};

// Main class
class RTApp :  public BaseApp {
	// Application camera
	Camera		camera;

	// RT or Raster
	bool		raster = true;
	bool		modified = false;
	bool		show_mouse = false;

	// Layers
	rt::Layer	rt_layer;
	raster::Layer	raster_layer;
	gui::Layer	gui_layer;

	// Capturer
	RTCapture	*capturer = nullptr;
	std::thread	*capturer_thread = nullptr;

	// Current scene
	Scene		scene;

	// Raster state
	int		highlight = -1;
	bool		edit_mode = false;

	// GUI state
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
	int row = 0;
	int col = 0;

	void handle_edit_mode() {
		// Assume edit mode
		auto eptr = raster_layer[highlight];
		auto &transform = eptr->transform();
		float speed = 0.01f;

		// TODO: determine a reference to a float
		float *fptr = nullptr;
		if (row == 0) {
			fptr = &transform.position[col];
		} else if (row == 1) {
			fptr = &transform.rotation[col];
			speed = 0.1f;
		} else if (row == 2) {
			fptr = &transform.scale[col];
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

	// Keyboard listener
	static void keyboard_handler(void *user, const io::KeyboardEvent &event) {
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
					app->row = (app->row + 1) % 3;

				if (event.key == GLFW_KEY_DOWN)
					app->row = (app->row + 2) % 3;

				if (event.key == GLFW_KEY_LEFT)
					app->col = (app->col + 2) % 3;

				if (event.key == GLFW_KEY_RIGHT)
					app->col = (app->col + 1) % 3;

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
				app->scene.save("scene.kobra");
			}

			// Refresh rasterization
			if (event.key == GLFW_KEY_R &&
				event.mods == GLFW_MOD_CONTROL) {
				KOBRA_LOG_FILE(notify) << "Force refreshing scene...\n";
				app->scene.save("scene.kobra");
				app->scene = Scene(app->context, app->window.command_pool, "scene.kobra");
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
				app->capturer = new RTCapture(app->context.vk, "scene.kobra", app->camera);
				app->capturer_thread = new std::thread([&]() {
					app->capturer->run();
				});
			}
		}
	}

	// Mouve camera
	static void mouse_movement(void *user, const io::MouseEvent &event) {
		static const float sensitivity = 0.001f;

		static bool first_movement = true;

		static float px = 0.0f;
		static float py = 0.0f;

		static float yaw = 0.0f;
		static float pitch = 0.0f;

		float dx = event.xpos - px;
		float dy = event.ypos - py;

		px = event.xpos;
		py = event.ypos;
		if (first_movement) {
			first_movement = false;
			return;
		}

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

	void create_scene() {
		Model model("resources/benchmark/suzanne.obj");

		Mesh box = Mesh::make_box({1, -1, 3.0}, {1, 1, 1});
		box.transform().rotation = {0, 30, 0};

		// rt::Mesh *mesh0 = new rt::Mesh(box);
		rt::Sphere *mesh0 = new rt::Sphere(1.0);
		mesh0->transform().position = {-1, 0, 3.0};

		rt::Sphere *sphere1 = new rt::Sphere(1.0);
		sphere1->transform().position = {2, 2, 0.0};

		rt::Mesh *mesh1 = new rt::Mesh(model[0]);
		mesh1->transform().position = {2, -1, 3};
		mesh1->transform().rotation = {0, -30, 0};

		// Box entire scene
		rt::Mesh *wall1 = new rt::Mesh(Mesh::make_box({0, -2, 0}, {5, 0.1, 5}));
		rt::Mesh *wall2 = new rt::Mesh(Mesh::make_box({0, 8, 0}, {5, 0.1, 5}));
		rt::Mesh *wall3 = new rt::Mesh(Mesh::make_box({-5, 3, 0}, {0.1, 5, 5}));
		rt::Mesh *wall4 = new rt::Mesh(Mesh::make_box({5, 3, 0}, {0.1, 5, 5}));
		rt::Mesh *wall5 = new rt::Mesh(Mesh::make_box({0, 3, -5}, {5, 5, 0.1}));

		// Square light source
		glm::vec3 center {0, 7.5, 3.0};
		Mesh light_mesh(
			VertexList {
				Vertex { {center.x - 0.5, center.y, center.z + 0.5}, {0, 0, 0} },
				Vertex { {center.x + 0.5, center.y, center.z + 0.5}, {0, 0, 0} },
				Vertex { {center.x + 0.5, center.y, center.z - 0.5}, {0, 0, 0} },
				Vertex { {center.x - 0.5, center.y, center.z - 0.5}, {0, 0, 0} },
			},

			IndexList {
				0, 1, 2,
				0, 2, 3
			}
		);

		rt::Mesh *light1 = new rt::Mesh(light_mesh);
		rt::Mesh *light2 = new rt::Mesh(light_mesh);

		light2->transform().position = {0, 10, 0.0};

		light1->set_material(Material {
			.albedo = {1, 1, 1},
			.shading_type = SHADING_TYPE_EMISSIVE
		});

		light2->set_material(Material {
			.albedo = {1, 1, 1},
			.shading_type = SHADING_TYPE_EMISSIVE
		});

		// mesh0->transform().scale = glm::vec3 {0.1f};
		mesh0->transform().move({0.25, -0.6, -2});

		Material mat {
			.albedo = {0, 0, 0},
			.shading_type = SHADING_TYPE_REFRACTION,
			.ior = 1.3
		};

		mat.set_albedo(context,
			window.command_pool,
			"resources/wood_floor_albedo.jpg"
		);

		// mesh1->transform().scale = glm::vec3 {0.1f};

		mesh0->set_material(mat);
		mat.ior = 1.0;

		mat.shading_type = SHADING_TYPE_DIFFUSE;
		sphere1->set_material(mat);

		// Set wall materials
		mat.albedo = {0.7, 0.7, 0.7};
		mat.shading_type = SHADING_TYPE_DIFFUSE;

		mat.set_albedo(context,
			window.command_pool,
			"resources/sky.jpg"
		);

		wall1->set_material(mat);
		mat.albedo_sampler = nullptr;

		wall2->set_material(mat);
		wall5->set_material(mat);

		mat.albedo = {1.0, 0.5, 0.5};
		wall3->set_material(mat);

		mat.albedo = {0.5, 0.5, 1.0};
		wall4->set_material(mat);

		mat.albedo = {0.5, 1.0, 0.5};
		mesh1->set_material(mat);

		Scene scene({
			mesh0, sphere1,
			wall1, mesh1,
			wall2, wall3, wall4, wall5,
			light1
		});

		scene.save("scene.kobra");
	}
public:
	// TODO: app to distinguish the number fo attachments
	// TODO: just remove the option of no depth buffer (always use depth buffer)
	RTApp(Vulkan *vk) : BaseApp({
		vk,
		800, 800, 2,
		"RTApp"
	}, true) {
		// Construct camera
		camera = Camera {
			Transform { {0, 6, 16}, {-0.2, 0, 0} },
			Tunings { 45.0f, 800, 800 }
		};

		// Load scene
		// create_scene();
		scene = Scene(context, window.command_pool, "scene.kobra");

		for (auto &obj : scene)
			std::cout << "Scene object: " << obj->name() << std::endl;

		///////////////////////
		// Initialize layers //
		///////////////////////

		// Ray tracing layer
		rt_layer = rt::Layer(window);
		rt_layer.add_scene(scene);

		// TODO: m8 gotta really fix auto channels
		rt_layer.set_environment_map(
			load_image_texture("resources/skies/background_3.jpg", 4)
		);

		// Rasterization layer
		raster_layer = raster::Layer(window, VK_ATTACHMENT_LOAD_OP_CLEAR);
		raster_layer.set_mode(raster::Layer::Mode::BLINN_PHONG);
		raster_layer.add_scene(scene);

		for (int i = 0; i < raster_layer.size(); i++)
			std::cout << "Rasterization layer object: " << raster_layer[i]->name() << std::endl;

		// GUI layer
		// TODO: be able to load gui elements from scene
		// TODO: naming objects
		gui_layer = gui::Layer(window, VK_ATTACHMENT_LOAD_OP_LOAD);
		initialize_gui();

		// Add event listeners
		window.keyboard_events->subscribe(keyboard_handler, this);
		window.mouse_events->subscribe(mouse_movement, &camera);
		window.cursor_mode(GLFW_CURSOR_DISABLED);
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

		// Highlight appropriate object
		raster_layer.clear_highlight();
		if (edit_mode)
			raster_layer.set_highlight(highlight, true);

		// Render appropriate layer
		if (raster) {
			raster_layer.render(cmd, framebuffer);
		} else {
			if (modified) {
				// TODO: fix clear method
				// rt_layer.clear();
				// rt_layer.add_scene(scene);
				KOBRA_LOG_FILE(notify) << "Reconstructing RT layer\n";
				scene.save("scene.kobra");
				scene = Scene(context, window.command_pool, "scene.kobra");

				rt_layer = rt::Layer(window);
				rt_layer.add_scene(scene);
				rt_layer.set_active_camera(camera);
				modified = false;
			}

			rt_layer.render(cmd, framebuffer);
		}

		// Render GUI
		update_gui();
		gui_layer.render(cmd, framebuffer);

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

int main()
{
	Vulkan *vulkan = new Vulkan();
	RTApp rt_app {vulkan};

	std::thread t1 {
		[&rt_app]() {
			rt_app.run();
		}
	};

	t1.join();

	delete vulkan;
}

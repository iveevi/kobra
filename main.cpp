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
#include "include/gui/layer.hpp"
#include "include/io/event.hpp"
#include "include/model.hpp"
#include "include/raster/layer.hpp"
#include "include/raytracing/layer.hpp"
#include "include/raytracing/mesh.hpp"
#include "include/raytracing/sphere.hpp"
#include "include/scene.hpp"
#include "include/types.hpp"

using namespace kobra;

class RTApp :  public BaseApp {
	// Application camera
	Camera		camera;

	// RT or Raster
	bool		raster = true;

	// Layers
	rt::Layer	rt_layer;
	raster::Layer	raster_layer;
	gui::Layer	gui_layer;

	// GUI elements
	gui::Text	*text_frame_rate;
	gui::Text	*layer_info;

	// Keyboard listener
	static void keyboard_handler(void *user, const io::KeyboardEvent &event) {
		RTApp *app = (RTApp *) user;

		if (event.key == GLFW_KEY_TAB && event.action == GLFW_PRESS)
			app->raster = !app->raster;
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
		rt::Sphere *mesh0 = new rt::Sphere({-1, 0, 3.0}, 1.0);
		rt::Sphere *sphere1 = new rt::Sphere({2, 2, 0.0}, 1.0);
		rt::Mesh *mesh1 = new rt::Mesh(box);

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
		Scene scene(context, window.command_pool, "scene.kobra");

		///////////////////////
		// Initialize layers //
		///////////////////////

		// Ray tracing layer
		rt_layer = rt::Layer(window);
		rt_layer.add_scene(scene);

		// Rasterization layer
		raster_layer = raster::Layer(window, VK_ATTACHMENT_LOAD_OP_CLEAR);
		raster_layer.set_mode(raster::Layer::Mode::BLINN_PHONG);
		raster_layer.add(scene);

		// GUI layer
		gui_layer = gui::Layer(window, VK_ATTACHMENT_LOAD_OP_LOAD);

		// TODO: method to add gui elements
		// TODO: add_scene for gui layer
		gui_layer.load_font("default", "resources/fonts/noto_sans.ttf");

		text_frame_rate = gui_layer.text_render("default")->text(
			"fps",
			window.coordinates(0, 0),
			{1, 1, 1, 1}
		);

		layer_info = gui_layer.text_render("default")->text(
			"",
			window.coordinates(0, 50),
			{1, 1, 1, 1}
		);

		gui_layer.add(text_frame_rate);
		gui_layer.add(layer_info);

		// Add event listeners
		window.keyboard_events->subscribe(keyboard_handler, this);
		window.mouse_events->subscribe(mouse_movement, &camera);
		window.cursor_mode(GLFW_CURSOR_DISABLED);
	}

	// Override record method
	// TODO: preview raytraced scene with a very low resolution
	void record(const VkCommandBuffer &cmd, const VkFramebuffer &framebuffer) override {
		static char buffer[1024];
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

		// Copy camera contents to each relevant layer
		rt_layer.set_active_camera(camera);
		raster_layer.set_active_camera(camera);

		// Render appropriate layer
		if (raster)
			raster_layer.render(cmd, framebuffer);
		else
			rt_layer.render(cmd, framebuffer);

		// Overlay statistics
		// TODO: method to update gui elements
		// TODO: should be made a standalone layer
		std::sprintf(buffer, "time: %.2f ms, fps: %.2f",
			1000 * frame_time,
			1.0f/frame_time
		);

		text_frame_rate->str = buffer;

		// RT layer statistics
		// TODO: why can i set str to an integer
		// layer_info->str = std::to_string(rt_layer.triangle_count()) + " triangles";

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

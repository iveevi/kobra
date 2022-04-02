// Standard headers
#include <cstring>
#include <iostream>
#include <thread>

#define KOBRA_VALIDATION_LAYERS
#define KOBRA_ERROR_ONLY
#define KOBRA_THROW_ERROR

#include "include/app.hpp"
#include "include/backend.hpp"
#include "include/gui/layer.hpp"
#include "include/model.hpp"
#include "include/raytracing/layer.hpp"
#include "include/raytracing/mesh.hpp"

using namespace kobra;

class RTApp :  public BaseApp {
	rt::Layer	rt_layer;

	Camera 		*active_camera;

	// GUI elements
	gui::Layer	gui_layer;

	gui::Text	*text_frame_rate;
	gui::Text	*layer_info;

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
public:
	RTApp(Vulkan *vk) : BaseApp({
		vk,
		800, 800, 2,
		"RTApp"
	}) {
		// Add RT elements
		rt_layer = rt::Layer(window);

		Camera camera {
			Transform { {0, 2, 15}, {0, 0, 0} },
			Tunings { 45.0f, 800, 800 }
		};

		rt_layer.add_camera(camera);
		active_camera = rt_layer.activate_camera(0);

		Model model("resources/benchmark/suzanne.obj");

		Mesh box = Mesh::make_box({-1, 2.0, 3.0}, {1, 1, 1});
		box.transform().rotation = {10, 30, 40};

		rt::Mesh *mesh0 = new rt::Mesh(box);
		rt::Mesh *mesh1 = new rt::Mesh(model[0]);

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
			.shading_type = SHADING_TYPE_EMMISIVE
		});
		
		light2->set_material(Material {
			.albedo = {1, 1, 1},
			.shading_type = SHADING_TYPE_EMMISIVE
		});

		// mesh0->transform().scale = glm::vec3 {0.1f};
		mesh0->transform().move({0.25, -0.6, -2});

		Material mat {
			.albedo = {0.5, 0.8, 0.5},
			.ior = 1.76
		};
		
		// mesh1->transform().scale = glm::vec3 {0.1f};

		mesh0->set_material(mat);
		mat.ior = 1.0;

		rt_layer.add(mesh0);
		// rt_layer.add(mesh1);

		// Set wall materials
		mat.albedo = {0.7, 0.7, 0.7};
		wall1->set_material(mat);
		wall2->set_material(mat);
		wall5->set_material(mat);
		
		mat.albedo = {1.0, 0.5, 0.5};
		wall3->set_material(mat);

		mat.albedo = {0.5, 0.5, 1.0};
		wall4->set_material(mat);

		// Walls
		rt_layer.add(wall1);
		rt_layer.add(wall2);
		rt_layer.add(wall3);
		rt_layer.add(wall4);
		rt_layer.add(wall5);

		rt_layer.add(light1);
		// rt_layer.add(light2);

		// Add GUI elements
		gui_layer = gui::Layer(window, VK_ATTACHMENT_LOAD_OP_LOAD);
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
		window.mouse_events->subscribe(mouse_movement, active_camera);
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
		float speed = 20.0f * frame_time;

		glm::vec3 forward = active_camera->transform.forward();
		glm::vec3 right = active_camera->transform.right();
		glm::vec3 up = active_camera->transform.up();

		if (input.is_key_down(GLFW_KEY_W))
			active_camera->transform.move(forward * speed);
		else if (input.is_key_down(GLFW_KEY_S))
			active_camera->transform.move(-forward * speed);

		if (input.is_key_down(GLFW_KEY_A))
			active_camera->transform.move(-right * speed);
		else if (input.is_key_down(GLFW_KEY_D))
			active_camera->transform.move(right * speed);

		if (input.is_key_down(GLFW_KEY_E))
			active_camera->transform.move(up * speed);
		else if (input.is_key_down(GLFW_KEY_Q))
			active_camera->transform.move(-up * speed);

		// Render RT layer
		rt_layer.render(cmd, framebuffer);

		// Overlay statistics
		// TODO: should be made a standalone layer
		std::sprintf(buffer, "time: %.2f ms, fps: %.2f",
			1000 * frame_time,
			1.0f/frame_time
		);

		text_frame_rate->str = buffer;

		// RT layer statistics
		// TODO: why can i set str to an integer
		layer_info->str = std::to_string(rt_layer.triangle_count()) + " triangles";

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

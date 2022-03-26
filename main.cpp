// Standard headers
#include <cstring>
#include <iostream>
#include <thread>

#define KOBRA_VALIDATION_LAYERS

// Local headers
#include "include/logger.hpp"
#include "include/model.hpp"
#include "include/vertex.hpp"
#include "include/raster/mesh.hpp"
#include "include/raster/layer.hpp"
#include "profiler.hpp"

#include <glm/gtx/rotate_vector.hpp>

using namespace kobra;

// Rasterization app
class RasterApp : public BaseApp {
	raster::Layer layer;
public:
	RasterApp(Vulkan *vk) : BaseApp({
		vk,
		800, 800, 2,
		"Rasterization"
	}) {
		// Load meshes
		Model model1("resources/benchmark/bunny_res_1.ply");
		Model model2("resources/benchmark/suzanne.obj");

		// Plane mesh
		Mesh plane(VertexList {
			Vertex {.position = {-10, -1, -10}, .normal = {0, 1, 0}, .tex_coords = {0, 0}},
			Vertex {.position = {-10, -1, 10}, .normal = {0, 1, 0}, .tex_coords = {0, 1}},
			Vertex {.position = {10, -1, 10}, .normal = {0, 1, 0}, .tex_coords = {1, 1}},
			Vertex {.position = {10, -1, -10}, .normal = {0, 1, 0}, .tex_coords = {1, 0}}
		}, {
			0, 1, 2,
			0, 2, 3
		});

		// Cube mesh (1x1x1)
		Mesh cube(VertexList {
			Vertex {.position = {-0.5, -0.5, -0.5}, .normal = {1, 0, 0}, .tex_coords = {0, 0}},
			Vertex {.position = {-0.5, 0.5, -0.5}, .normal = {1, 0, 0}, .tex_coords = {0, 1}},
			Vertex {.position = {0.5, 0.5, -0.5}, .normal = {1, 0, 0}, .tex_coords = {1, 1}},
			Vertex {.position = {0.5, -0.5, -0.5}, .normal = {1, 0, 0}, .tex_coords = {1, 0}},
			
			Vertex {.position = {-0.5, -0.5, 0.5}, .normal = {1, 0, 0}, .tex_coords = {0, 0}},
			Vertex {.position = {-0.5, 0.5, 0.5}, .normal = {1, 0, 0}, .tex_coords = {0, 1}},
			Vertex {.position = {0.5, 0.5, 0.5}, .normal = {1, 0, 0}, .tex_coords = {1, 1}},
			Vertex {.position = {0.5, -0.5, 0.5}, .normal = {1, 0, 0}, .tex_coords = {1, 0}},
		}, {
			// Account for culling
			0, 2, 1,
			0, 3, 2,

			4, 6, 5,
			4, 7, 6,

			0, 5, 4,
			0, 1, 5,

			1, 6, 5,
			1, 2, 6,

			2, 6, 7,
			2, 7, 3,

			3, 4, 7,
			3, 0, 4
		});

		raster::Mesh *mesh1 = new raster::Mesh(window.context, model1[0]);
		raster::Mesh *mesh2 = new raster::Mesh(window.context, model2[0]);
		raster::Mesh *mesh3 = new raster::Mesh(window.context, plane);
		raster::Mesh *mesh4 = new raster::Mesh(window.context, cube);
		raster::Mesh *mesh5 = new raster::Mesh(window.context, cube);
		raster::Mesh *mesh6 = new raster::Mesh(window.context, cube);
		raster::Mesh *mesh7 = new raster::Mesh(window.context, model2[0]);

		mesh5->transform().move(glm::vec3(0, 0, -5));
		mesh6->transform().move(glm::vec3(-5, 0, 0));

		mesh1->transform() = Transform({0.0f, 0.0f, -4.0f});
		mesh1->transform().scale(10);

		mesh2->transform() = Transform({0.0f, 4.0f, -4.0f});
		mesh7->transform() = Transform({0.0f, 4.0f, 4.0f});
		// mesh2->transform().scale = glm::vec3(1/10.0f);

		KOBRA_LOG_FILE(notify) << "Loaded all models and meshes\n";

		// Initialize layer
		Camera camera {
			Transform { glm::vec3(0.0f, 0.0f, -1.0f) },
			Tunings { 45.0f, 800, 600 }
		};

		layer = raster::Layer(window, camera, VK_ATTACHMENT_LOAD_OP_CLEAR);
		// layer.add(mesh1);
		layer.add(mesh2);
		layer.add(mesh3);
		layer.add(mesh4);
		layer.add(mesh5);
		layer.add(mesh6);
		layer.add(mesh7);

		auto mouse_movement = [](void *user, const io::MouseEvent &event) {
			static const float sensitivity = 0.001f;

			static bool first_movement = true;

			static float px = 0.0f;
			static float py = 0.0f;

			float dx = event.xpos - px;
			float dy = event.ypos - py;

			px = event.xpos;
			py = event.ypos;
			if (first_movement) {
				first_movement = false;
				return;
			}

			Camera *camera = (Camera *) user;
			
			camera->transform.rotate(glm::vec3(0.0f, -dx * sensitivity, 0.0f));
			camera->transform.rotate(glm::vec3(dy * sensitivity, 0.0f, 0.0f));
		};

		// Add to event handlers
		window.mouse_events->subscribe(mouse_movement, &layer.camera());

		// Disable cursor
		window.cursor_mode(GLFW_CURSOR_DISABLED);
	}

	// Override record method
	void record(const VkCommandBuffer &cmd, const VkFramebuffer &framebuffer) override {
		static float time = 0.0f;

		// Start recording command buffer
		Vulkan::begin(cmd);

		// Camera movement
		glm::vec3 forward = layer.camera().transform.forward();
		glm::vec3 right = layer.camera().transform.right();
		glm::vec3 up = layer.camera().transform.up();

		// WASDEQ movement
		float speed = 0.01f;
		if (input.is_key_down(GLFW_KEY_W))
			layer.camera().transform.move(forward * speed);
		else if (input.is_key_down(GLFW_KEY_S))
			layer.camera().transform.move(-forward * speed);

		if (input.is_key_down(GLFW_KEY_A))
			layer.camera().transform.move(-right * speed);
		else if (input.is_key_down(GLFW_KEY_D))
			layer.camera().transform.move(right * speed);

		if (input.is_key_down(GLFW_KEY_E))
			layer.camera().transform.move(up * speed);
		else if (input.is_key_down(GLFW_KEY_Q))
			layer.camera().transform.move(-up * speed);


		// Record commands
		layer.render(cmd, framebuffer);

		// End recording command buffer
		Vulkan::end(cmd);

		// Progress time
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
	// Redirect logger to file
	// Logger::switch_file("kobra.log");

	std::string bunny_obj = "resources/benchmark/bunny_res_1.ply";
	Model model(bunny_obj);
	Logger::ok("Model loaded");

	// Initialize Vulkan
	Vulkan *vulkan = new Vulkan();

	// Create and launch profiler app
	Profiler *pf = new Profiler();
	// ProfilerApplication app {vulkan, pf};
	/* std::thread thread {
		[&]() { app.run(); }
	}; */

	// Create and launch raster app
	RasterApp raster_app {vulkan};
	/* std::thread raster_thread {
		[&]() { raster_app.run(); }
	}; */

	raster_app.run();


	// Wait for all to finish
	// thread.join();
	// raster_thread.join();

	delete vulkan;
}

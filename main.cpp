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
		std::string bunny_obj = "resources/benchmark/bunny_res_1.ply";
		Model <VERTEX_TYPE_POSITION> model(bunny_obj);

		raster::Mesh <VERTEX_TYPE_POSITION> *mesh = new raster::Mesh <VERTEX_TYPE_POSITION> (window.context, model[0]);
		mesh->transform() = Transform({0.0f, 0.0f, -4.0f});
		mesh->transform().scale = glm::vec3(10.0f);

		KOBRA_LOG_FILE(notify) << "Loaded all models and meshes\n";

		// Initialize layer
		Camera camera {
			Transform { glm::vec3(0.0f, 0.0f, -1.0f) },
			Tunings { 45.0f, 800, 600 }
		};

		layer = raster::Layer(window, camera, VK_ATTACHMENT_LOAD_OP_CLEAR);
		layer.add(mesh);

		// Bind camera movement
		// TODO: would be smoother using input object
		auto key_movement = [&](void *user, const io::KeyboardEvent &event) {
			float speed = 0.25f;

			// TODO: define key constants (keys.hpp)
			Camera *camera = (Camera *) user;

			// TODO: move functions
			if (event.key == GLFW_KEY_W)
				camera->transform.position += camera->transform.forward * speed;
			else if (event.key == GLFW_KEY_S)
				camera->transform.position -= camera->transform.forward * speed;

			if (event.key == GLFW_KEY_A)
				camera->transform.position += camera->transform.right * speed;
			else if (event.key == GLFW_KEY_D)
				camera->transform.position -= camera->transform.right * speed;

			if (event.key == GLFW_KEY_E)
				camera->transform.position += camera->transform.up * speed;
			else if (event.key == GLFW_KEY_Q)
				camera->transform.position -= camera->transform.up * speed;
		};

		auto mouse_movement = [&](void *user, const io::MouseEvent &event) {
			static const float sensitivity = 0.001f;
			static bool first_mouse = true;
			static float last_x = window.width / 2.0f;
			static float last_y = window.height / 2.0f;
			
			Camera *camera = (Camera *) user;
	
			// if (!mouse_tracking)
			//	return;

			if (first_mouse) {
				first_mouse = false;
				last_x = event.xpos;
				last_y = event.ypos;
				return;
			}

			// Store pitch and yaw
			static float pitch = 0.0f;
			static float yaw = 0.0f;

			float xoffset = event.xpos - last_x;
			float yoffset = event.ypos - last_y;

			xoffset *= sensitivity;
			yoffset *= sensitivity;

			yaw += xoffset;
			pitch += yoffset;

			if (pitch > 89.0f)
				pitch = 89.0f;
			else if (pitch < -89.0f)
				pitch = -89.0f;

			// Update camera
			camera->transform.set_euler(pitch, yaw);

			last_x = event.xpos;
			last_y = event.ypos;
		};

		// Add to event handlers
		window.keyboard_events->subscribe(key_movement, &layer.camera());
		window.mouse_events->subscribe(mouse_movement, &layer.camera());
	}

	// Override record method
	void record(const VkCommandBuffer &cmd, const VkFramebuffer &framebuffer) override {
		// Start recording command buffer
		Vulkan::begin(cmd);

		// Record commands
		layer.render(cmd, framebuffer);

		// End recording command buffer
		Vulkan::end(cmd);
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
	Model <VERTEX_TYPE_POSITION> model(bunny_obj);
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

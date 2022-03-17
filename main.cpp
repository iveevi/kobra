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

		layer = raster::Layer(window, camera);
		layer.add(mesh);
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

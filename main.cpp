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

	// GUI elements
	gui::Layer	gui_layer;

	gui::Text	*text_frame_rate;
public:
	RTApp(Vulkan *vk) : BaseApp({
		vk,
		800, 800, 2,
		"RTApp"
	}) {
		// Add RT elements
		rt_layer = rt::Layer(window);
		
		Camera camera {
			Transform { {0, 0, 4} },
			Tunings { 45.0f, 800, 800 }
		};

		rt_layer.add_camera(camera);
		rt_layer.activate_camera(0);

		Model model("resources/benchmark/suzanne.obj");

		rt::Mesh *mesh = new rt::Mesh(model[0]);

		rt_layer.add(mesh);

		// Add GUI elements
		gui_layer = gui::Layer(window, VK_ATTACHMENT_LOAD_OP_LOAD);
		gui_layer.load_font("default", "resources/fonts/noto_sans.ttf");
		
		text_frame_rate = gui_layer.text_render("default")->text(
			"fps",
			window.coordinates(0, 0),
			{1, 1, 1, 1}
		);

		gui_layer.add(text_frame_rate);
	}

	// Override record method
	void record(const VkCommandBuffer &cmd, const VkFramebuffer &framebuffer) override {
		static char buffer[1024];

		// Start recording command buffer
		Vulkan::begin(cmd);

		// Render RT layer
		rt_layer.render(cmd, framebuffer);

		// Overlay statistics
		// TODO: should be made a standalone layer
		std::sprintf(buffer, "time: %.2f ms, fps: %.2f",
			1000 * frame_time,
			1.0f/frame_time
		);

		text_frame_rate->str = buffer;

		gui_layer.render(cmd, framebuffer);

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

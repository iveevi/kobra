#include "../../include/engine/rt_capture.hpp"

namespace kobra {

namespace engine {

// Constructor from scene file and camera
RTCapture::RTCapture(Vulkan *vk, const std::string &scene_file, const Camera &camera)
		: BaseApp({
			vk,
			800, 800, 2,
			"RT Capture",
		}),
		camera(camera)
{
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
	batch = rt::Batch(800, 800, 50, 50, 1024);
	index = batch.make_batch_index(0, 0, 8, 16);
	index.accumulate = true;

	// Create GUI
	gui_layer = gui::Layer(window, VK_ATTACHMENT_LOAD_OP_LOAD);

	// Fonts
	gui_layer.load_font("default", "resources/fonts/noto_sans.ttf");

	// Progress indicator
	text_progress = gui_layer.text_render("default")->text(
		"Progress: ",
		window.coordinates(0, window.height - 10),
		{1, 1, 1, 1}, 0.3
	);

	gui_layer.add(text_progress);
}

// Render loop
void RTCapture::record(const VkCommandBuffer &cmd, const VkFramebuffer &framebuffer)
{
	static char buffer[1024];
	static float time = 0.0f;

	// Start recording command buffer
	Vulkan::begin(cmd);

	// Render scene
	layer.render(cmd, framebuffer, batch, index);

	// Track progress
	time += frame_time;
	float progress = batch.progress();
	float eta = time * (1.0f - progress) / progress;

	std::sprintf(buffer,
		"Progress: %.2f%%, Total time: %.2fs (+%.2fs), ETA: %.2fs\n",
		progress * 100.0f, time, frame_time, eta
	);

	// Update and render GUI
	text_progress->str = buffer;
	gui_layer.render(cmd, framebuffer);

	// End recording command buffer
	Vulkan::end(cmd);

	// Next batch
	batch.increment(index);
}

// Treminator
void RTCapture::terminate()
{
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

}

}

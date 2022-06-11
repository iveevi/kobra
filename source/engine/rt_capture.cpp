#include "../../include/engine/rt_capture.hpp"

namespace kobra {

namespace engine {

// Constructor from scene file and camera
RTCapture::RTCapture(const vk::raii::PhysicalDevice &phdev,
		const vk::Extent2D &extent,
		const std::vector <const char *> &extensions_,
		const std::string &scene_file,
		const Camera &camera)
		: BaseApp(phdev, extent, extensions_),
		camera(camera),
		layer(phdev, device,
			command_pool,
			descriptor_pool,
			extent,
			swapchain.format,
			depth_buffer.format
		),
		gui_layer(phdev, device,
			command_pool,
			descriptor_pool,
			extent,
			swapchain.format,
			depth_buffer.format,
			vk::AttachmentLoadOp::eLoad
		),
		dimensions(extent)
{
	// Load scene
	Scene scene = Scene(phdev, device, command_pool, scene_file);

	// Create raytracing layer
	layer.add_scene(scene);
	layer.set_active_camera(camera);
	layer.set_mode(rt::Layer::Mode::MIS_PATH_TRACER);

	// Set background texture
	// TODO: default arguments for this function
	ImageData background = make_image(phdev, device,
		command_pool,
		"resources/skies/background_1.jpg",
		vk::ImageTiling::eOptimal,
		vk::ImageUsageFlagBits::eTransferDst
			| vk::ImageUsageFlagBits::eSampled,
		vk::MemoryPropertyFlagBits::eDeviceLocal,
		vk::ImageAspectFlagBits::eColor
	);

	layer.set_environment_map(std::move(background));

	// Create batch
	// TODO: a method to generate optimal batch sizes (eg 50x50 is
	// faster than 10x10)
	batch = rt::Batch(800, 800, 25, 25, 100);

	index = batch.make_batch_index(0, 0, 1, 100);
	index.surface_samples = 1;
	index.accumulate = true;

	// Fonts
	gui_layer.load_font("default", "resources/fonts/noto_sans.ttf");

	// Progress indicator
	text_progress = gui_layer.text_render("default")->text(
		"Progress: ",
		coordinates(0, extent.height - 20),
		{0, 1, 0, 1}, 0.5
	);

	gui_layer.add(text_progress);
}

// Render loop
void RTCapture::record(const vk::raii::CommandBuffer &cmd, const vk::raii::Framebuffer &framebuffer)
{
	static char buffer[1024];
	static float time = 0.0f;

	// Start recording command buffer
	cmd.begin({});

	// Render scene
	layer.render(cmd, framebuffer, dimensions, batch, index);

	// Track progress
	time += frame_time;
	float progress = batch.progress();

	float eta_s = time * (1.0f - progress) / progress;
	float eta_m = 0;
	float eta_h = 0;

	if (eta_s > 60.0f) {
		eta_m = int(eta_s / 60.0f);
		eta_s = fmod(eta_s, 60.0f);
	}

	if (eta_m > 60.0f) {
		eta_h = int(eta_m / 60.0f);
		eta_m = fmod(eta_m, 60.0f);
	}

	std::sprintf(buffer,
		"Progress: %.2f%%, Total time: %.2fs (+%.2fs), ETA: %.2fh %.2fm %.2fs",
		progress * 100.0f, time, frame_time, eta_h, eta_m, eta_s
	);

	// Update and render GUI
	text_progress->str = buffer;
	gui_layer.render(cmd, framebuffer);

	// End recording command buffer
	cmd.end();

	// Next batch
	batch.increment(index);
}

// Treminator
void RTCapture::terminate()
{
	bool b = batch.completed();
	if (term || b) {
		glfwSetWindowShouldClose(window.handle, GLFW_TRUE);
		const auto &buffer = layer.pixels();

		/* TODO: make an easier an more straight forward way to
		// save a buffer to an image
		Image img {
			.width = 800,
			.height = 800
		};

		Capture::snapshot(buffer, img);
		img.write("capture.png");
		KOBRA_LOG_FUNC(notify) << "Capture saved to capture.png\n"; */
	}
}

}

}

#include <iostream>

// OpenCV for video capture
#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>

// Engine headers
#include "include/app.hpp"
#include "include/capture.hpp"
#include "include/core/interpolation.hpp"
#include "include/cuda/color.cuh"
#include "include/layers/denoiser.cuh"
#include "include/layers/font_renderer.hpp"
#include "include/layers/framer.hpp"
#include "include/layers/hybrid_tracer.cuh"
#include "include/layers/optix_tracer.cuh"
#include "include/layers/basilisk.cuh"
#include "include/optix/options.cuh"
#include "include/optix/parameters.cuh"
#include "include/renderer.hpp"
#include "include/scene.hpp"
#include "include/texture.hpp"
#include "include/asmodeus/backend.cuh"
#include "include/asmodeus/wsris.cuh"
#include "include/project.hpp"

#include "sequences.hpp"

#define DENOISER 0
#define SAMPLES_PER_FRAME 16
#define CAPTURE_PATH "capture.mp4"
#define ENVIRONMENT_MAP "resources/skies/background_1.jpg"
// #define BASILISK_MODE kobra::optix::eVoxel
#define BENCHMARK_SCENE "/home/venki/models/bistro_interior.kobra"
#define STILL_IMAGE

struct Bench : public kobra::BaseApp {
	// TODO: let the scene run on any virtual device?
	kobra::Entity camera;
	kobra::Scene scene;

	// Layers
	kobra::layers::Framer framer;

#if DENOISER
	kobra::layers::Denoiser denoiser;
#endif

#ifdef BASILISK_MODE
	kobra::layers::Basilisk tracer;
#else
	kobra::asmodeus::GridBasedReservoirs grid_based;
#endif

	// Buffers
	CUdeviceptr b_traced;
	std::vector <uint32_t> b_traced_cpu;

	// Capture
	cv::VideoWriter capture;

	kobra::core::Sequence <glm::vec3> positions = CAMERA_POSITION_SEQUENCE;
	kobra::core::Sequence <glm::vec3> rotations = CAMERA_ROTATION_SEQUENCE;

	Bench(const vk::raii::PhysicalDevice &phdev,
			const std::vector <const char *> &extensions,
			const std::string &scene_path)
			: BaseApp(phdev, "Bench",
				vk::Extent2D {1000, 1000},
				extensions, vk::AttachmentLoadOp::eLoad
			) {
		// Load scene and camera
		scene.load(get_device(), scene_path);
		camera = scene.ecs.get_entity("Camera");

		// Initialize layers
		framer = kobra::layers::Framer::make(get_context());

#if DENOISER
		denoiser = kobra::layers::Denoiser::make(
			extent,
			kobra::layers::Denoiser::eNormal
				| kobra::layers::Denoiser::eAlbedo
		);
#endif

#ifdef BASILISK_MODE
		tracer = kobra::layers::Basilisk
			::make(get_context(), {1000, 1000});

#ifdef ENVIRONMENT_MAP
		kobra::layers::set_envmap(tracer, ENVIRONMENT_MAP);
#endif

		// tracer.launch_params.options.indirect_only = true;
		tracer.launch_params.options.reprojected_reuse = true;

#else
		grid_based = kobra::asmodeus::GridBasedReservoirs
			::make(get_context(), {1000, 1000});

#ifdef ENVIRONMENT_MAP
		grid_based.set_envmap(ENVIRONMENT_MAP);
#endif

#endif

#ifdef BASILISK_MODE
		int width = tracer.extent.width;
		int height = tracer.extent.height;
#else
		int width = grid_based.extent.width;
		int height = grid_based.extent.height;
#endif
		
		// Setup capture
		capture.open(CAPTURE_PATH,
			cv::VideoWriter::fourcc('A', 'V', 'C', '1'),
			60, cv::Size(width, height)
		);

		assert(capture.isOpened());

		// Allocate buffers
		size_t size = extent.width * extent.height;
		b_traced = kobra::cuda::alloc(size * sizeof(uint32_t));
		b_traced_cpu.resize(size);
	}

	// Path tracing kernel
	float time = 0.0f;

	void record(const vk::raii::CommandBuffer &cmd,
			const vk::raii::Framebuffer &framebuffer) override {
#ifdef STILL_IMAGE
		time = 0.0f;
#endif

		// Move the camera
		auto &transform = camera.get <kobra::Transform> ();

		// Interpolate camera position
		glm::vec3 pos = kobra::core::piecewise_linear(positions, time);
		glm::vec3 rot = kobra::core::piecewise_linear(rotations, time);

		transform.position = pos;
		transform.rotation = rot;
		
		// Render the frame
		int frames = 0;
		while (frames < SAMPLES_PER_FRAME) {
#ifdef BASILISK_MODE
			kobra::layers::compute(tracer, scene.ecs,
				camera.get <kobra::Camera> (),
				camera.get <kobra::Transform> (),
				BASILISK_MODE, (frames++ > 0)
			);
#else
			grid_based.render(scene.ecs,
				camera.get <kobra::Camera> (),
				camera.get <kobra::Transform> (),
				(frames++ > 0)
			);
#endif
		}
	
#if DENOISER
		kobra::layers::denoise(denoiser, {
#ifdef BASILISK_MODE
			.color = kobra::layers::color_buffer(tracer),
			.normal = kobra::layers::normal_buffer(tracer),
			.albedo = kobra::layers::albedo_buffer(tracer)
#else
			.color = grid_based.color_buffer(),
			.normal = grid_based.normal_buffer(),
			.albedo = grid_based.albedo_buffer()
#endif
		});
#endif

		// Now trace and render
#ifdef BASILISK_MODE
		int width = tracer.extent.width;
		int height = tracer.extent.height;
#else
		int width = grid_based.extent.width;
		int height = grid_based.extent.height;
#endif

		cmd.begin({});
			float4 *d_output = 0;

#ifdef BASILISK_MODE
			d_output = (float4 *) kobra::layers::color_buffer(tracer);
#else
			d_output = (float4 *) grid_based.color_buffer();
#endif

#if DENOISER
			d_output = (float4 *) denoiser.result;
#endif

			kobra::cuda::hdr_to_ldr(
				d_output,
				(uint32_t *) b_traced,
				width, height,
				kobra::cuda::eTonemappingACES
			);

			kobra::cuda::copy(b_traced_cpu, b_traced, width * height);
			kobra::layers::render(framer, b_traced_cpu, cmd, framebuffer);
		cmd.end();

		// Write the frame to the video
		cv::Mat mat(width, height, CV_8UC4, b_traced_cpu.data());
		cv::cvtColor(mat, mat, cv::COLOR_BGRA2RGB);
		capture.write(mat);

		if (time > CAMERA_TIMES.back()) {
			capture.release();
			terminate_now();
		}
		
#ifdef STILL_IMAGE
		
		std::string capture_path = "capture.png";
		stbi_write_png(capture_path.c_str(),
			width, height, 4,
			b_traced_cpu.data(),
			width * 4
		);

#endif

		// Update time (fixed)
		time += 1/60.0f;
	}
};

int main()
{
	// Load Vulkan physical device
	auto extensions = {
		VK_KHR_SWAPCHAIN_EXTENSION_NAME,
		VK_KHR_GET_MEMORY_REQUIREMENTS_2_EXTENSION_NAME,
		VK_KHR_EXTERNAL_MEMORY_EXTENSION_NAME,
		VK_KHR_EXTERNAL_SEMAPHORE_EXTENSION_NAME,
	};

	auto predicate = [&extensions](const vk::raii::PhysicalDevice &dev) {
		return kobra::physical_device_able(dev, extensions);
	};

	// Choose a physical device
	// TODO: static lambda (GREEDY)
	auto phdev = kobra::pick_physical_device(predicate);

	std::cout << "Extensions:" << std::endl;
	for (auto str : extensions)
		std::cout << "\t" << str << std::endl;

	// Create and launch the application
	Bench app(phdev, {
		VK_KHR_SWAPCHAIN_EXTENSION_NAME,
		VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME,
		VK_KHR_EXTERNAL_MEMORY_EXTENSION_NAME,
	}, BENCHMARK_SCENE);

	app.run();
}

#ifndef MOTION_CAPTURE_H_
#define MOTION_CAPTURE_H_

// GLM headers
#include <glm/glm.hpp>

// OpenCV for video capture
#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/imgproc.hpp>

// Engine headers
#include "include/app.hpp"
#include "include/capture.hpp"
#include "include/core/interpolation.hpp"
#include "include/layers/hybrid_tracer.cuh"
#include "include/layers/optix_tracer.cuh"
#include "include/optix/options.cuh"
#include "include/scene.hpp"

// TODO: do base app without inheritance (simple struct..., app and baseapp not
// related)
struct MotionCapture : public kobra::BaseApp {
	// TODO: let the scene run on any virtual device?
	kobra::Entity camera;
	kobra::Scene scene;

	kobra::layers::OptixTracer tracer;
	kobra::layers::HybridTracer hybrid_tracer;

	// Capture
	cv::VideoWriter capture;
	std::vector <byte> frame;

	const std::vector <glm::vec3> camera_pos {
		{-13.81, 23.40, 24.29},
		{24.87, 21.49, 22.52},
		{59.04, 21.49, 4.70},
		{65.30, 21.49, -20.18},
		{57.76, 21.49, -45.75},
		{0.61, 24.94, -53.59},
		{-16.36, 23.35, -35.79},
		{-18.45, 21.34, -13.25},
		{0.40, 14.35, -3.78},
		{18.99, 8.53, -6.17},
		{37.84, 6.21, -16.40},
		{32.84, 6.21, -27.41},
		{29.18, 16.56, -26.53},
		{18.24, 15.16, -27.63},
		{6.65, 15.16, -22.16}
	};

	const std::vector <glm::vec3> camera_rot {
		{-0.24, -0.44, 0.00},
		{-0.29, 0.09, 0.00},
		{-0.29, 0.85, 0.00},
		{-0.33, 1.55, 0.00},
		{-0.39, 2.41, 0.00},
		{-0.34, 3.67, 0.00},
		{-0.37, 4.11, 0.00},
		{-0.35, 4.71, 0.00},
		{-0.35, 4.84, 0.00},
		{-0.24, 4.74, 0.00},
		{-0.21, 7.67, 0.00},
		{-0.23, 8.51, 0.00},
		{-0.54, 8.56, 0.00},
		{-0.27, 9.26, 0.00},
		{-0.49, 10.19, 0.00}
	};

	std::vector <float> times;

	kobra::core::Sequence <glm::vec3> camera_pos_seq {
		.values = camera_pos,
		.times = times
	};

	kobra::core::Sequence <glm::vec3> camera_rot_seq {
		.values = camera_rot,
		.times = times
	};
	
	MotionCapture(const vk::raii::PhysicalDevice &phdev,
			const std::vector <const char *> &extensions,
			const std::string &scene_path)
			: BaseApp(phdev, "MotionCapture",
				vk::Extent2D {1600, 1200},
				extensions, vk::AttachmentLoadOp::eLoad
			),
			tracer(get_context(),
				vk::AttachmentLoadOp::eClear,
				1000, 1000
			) {
		// Load scene and camera
		scene.load(get_device(), scene_path);
		camera = scene.ecs.get_entity("Camera");

		// Setup tracer
		tracer.environment_map("resources/skies/background_1.jpg");
		tracer.sampling_strategy = kobra::optix::eSpatioTemporal;
		tracer.denoiser_enabled = false;

		// Setup hybrid tracer
		KOBRA_LOG_FILE(kobra::Log::INFO) << "Hybrid tracer setup\n";
		hybrid_tracer = kobra::layers::HybridTracer::make(get_context());
		kobra::layers::set_envmap(hybrid_tracer, "resources/skies/background_1.jpg");

		// Setup capture
		capture.open(
			"animation.mp4",
			cv::VideoWriter::fourcc('A', 'V', 'C', '1'),
			60, cv::Size(1000, 1000)
		);

		if (!capture.isOpened())
			std::cout << "Failed to open capture" << std::endl;
		else
			std::cout << "Capture opened" << std::endl;

		// Fill in time intervals
		for (int i = 0; i < camera_pos.size(); i++)
			times.push_back(i);

		camera_pos_seq.times = times;
		camera_rot_seq.times = times;
	}

	float time = 0.0f;
	void record(const vk::raii::CommandBuffer &cmd,
			const vk::raii::Framebuffer &framebuffer) override {
		// Move the camera
		auto &transform = camera.get <kobra::Transform> ();
		
		// Interpolate camera position
		if (time > camera_pos.size() - 1)
			terminate_now();

		glm::vec3 pos = kobra::core::piecewise_linear(camera_pos_seq, time);
		glm::vec3 rot = kobra::core::piecewise_linear(camera_rot_seq, time);

		transform.position = pos;
		transform.rotation = rot;

		// Now trace and render
		cmd.begin({});
			/* for (int i = 0; i < 2; i++)
				tracer.compute(scene.ecs);
			tracer.render(cmd, framebuffer); */

			kobra::layers::compute(hybrid_tracer,
				scene.ecs,
				camera.get <kobra::Camera> (),
				camera.get <kobra::Transform> ()
			);

			kobra::layers::render(hybrid_tracer,
				cmd, framebuffer
			);
		cmd.end();

		// Write the frame to the video
		tracer.capture(frame);

		cv::Mat mat(1000, 1000, CV_8UC4, frame.data());
		cv::cvtColor(mat, mat, cv::COLOR_BGRA2RGB);
		/* capture.write(mat);

		if (time > 5) {
			capture.release();
			terminate_now();
		} */

		// Update time (fixed)
		time += 1/60.0f;
	}
};

#endif

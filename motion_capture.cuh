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
#include "include/layers/optix_tracer.cuh"
#include "include/optix/options.cuh"
#include "include/scene.hpp"
#include "include/capture.hpp"

// TODO: do base app without inheritance (simple struct..., app and baseapp not
// related)
struct MotionCapture : public kobra::BaseApp {
	// TODO: let the scene run on any virtual device?
	kobra::Entity camera;
	kobra::Scene scene;
	kobra::layers::OptixTracer tracer;

	// Capture
	cv::VideoWriter capture;
	std::vector <byte> frame;
	
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
	}

	float time = 0.0f;
	void record(const vk::raii::CommandBuffer &cmd,
			const vk::raii::Framebuffer &framebuffer) override {
		// Move the camera
		auto &transform = camera.get <kobra::Transform> ();
		
		static const glm::vec3 origin {10, 5, -10};
		transform.position = glm::vec3 {
			5.0f * std::sin(time) + 7,
			6.0f,
			5.0f * std::cos(time) - 10
		};

		// Look at the origin always
		glm::vec3 eye = transform.position;
		glm::vec3 dir = glm::normalize(origin - eye);
		
		transform.look(dir);

		// Now trace and render
		cmd.begin({});
			for (int i = 0; i < 4; i++)
				tracer.compute(scene.ecs);
			tracer.render(cmd, framebuffer);
		cmd.end();

		// Write the frame to the video
		tracer.capture(frame);

		cv::Mat mat(1000, 1000, CV_8UC4, frame.data());
		cv::cvtColor(mat, mat, cv::COLOR_BGRA2RGB);
		capture.write(mat);

		if (time > 5) {
			capture.release();
			terminate_now();
		}

		// Update time (fixed)
		time += 1/60.0f;
	}
};

#endif

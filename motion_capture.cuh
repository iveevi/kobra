#ifndef MOTION_CAPTURE_H_
#define MOTION_CAPTURE_H_

// GLM headers
#include <glm/glm.hpp>

// OpenCV for video capture
#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>

// Engine headers
#include "include/app.hpp"
#include "include/capture.hpp"
#include "include/core/interpolation.hpp"
#include "include/layers/font_renderer.hpp"
#include "include/layers/hybrid_tracer.cuh"
#include "include/layers/optix_tracer.cuh"
#include "include/layers/wadjet.cuh"
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
	kobra::layers::Wadjet wadjet_tracer;
	kobra::layers::FontRenderer font_renderer;

	// Capture
	cv::VideoWriter capture;
	std::vector <byte> frame;

	/* const std::vector <glm::vec3> camera_pos {
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
	}; */
	
	const std::vector <glm::vec3> camera_pos {
		{76.80, 115.63, 83.22},
		{89.34, 92.42, 71.24},
		{93.42, 76.90, 59.02},
		{100.20, 68.60, 52.30},
		{101.17, 64.25, 44.44},
		{102.42, 58.69, 32.65},
		{104.05, 48.85, 14.16},
		{112.92, 44.75, 11.08},
		{125.01, 40.79, 8.21},
		{126.03, 38.04, -0.29},
		{121.78, 36.74, -5.45},
		{107.80, 36.30, -8.05},
		{89.78, 36.01, -9.72},
		{72.60, 35.70, -10.48},
		{58.29, 35.34, -10.93},
		{47.93, 34.95, -10.92},
		{38.73, 34.77, -5.04},
		{43.25, 34.96, 2.55},
		{44.47, 35.05, 16.51},
		{59.43, 34.80, 24.61},
		{75.37, 34.80, 32.55},
		{93.17, 34.80, 32.98},
		{93.17, 34.80, 32.98},
		{98.94, 28.53, 23.61},
		{118.68, 15.86, 14.19},
		{148.76, 9.78, 12.65},
		{148.62, 8.16, 10.27},
		{126.98, 9.01, 4.85},
		{109.89, 9.01, 1.87},
		{109.89, 9.01, 1.87},
		{104.52, 8.56, 7.67},
		{90.71, 7.54, 12.84},
		{61.25, 7.45, 4.71},
		{40.58, 7.25, 6.55},
		{45.33, 6.55, -6.13},
		{60.03, 6.87, -7.99},
		{80.66, 6.26, -7.84},
		{103.09, 10.09, -7.18}
	};
	
	const std::vector <glm::vec3> camera_rot {
		{-0.94, 0.06, 0.00},
		{-0.91, -0.35, 0.00},
		{-0.83, -0.29, 0.00},
		{-0.54, -0.13, 0.00},
		{-0.48, -0.12, 0.00},
		{-0.42, -0.10, 0.00},
		{-0.48, -1.21, 0.00},
		{-0.39, -1.22, 0.00},
		{-0.32, -0.21, 0.00},
		{-0.27, 0.11, 0.00},
		{-0.05, 1.32, 0.00},
		{-0.02, 1.46, 0.00},
		{-0.02, 1.52, 0.00},
		{-0.02, 1.53, 0.00},
		{-0.04, 1.56, 0.00},
		{-0.04, 1.61, 0.00},
		{-0.00, 2.34, 0.00},
		{0.04, 3.66, 0.00},
		{-0.05, 5.20, 0.00},
		{-0.07, 5.62, 0.00},
		{-0.08, 5.99, 0.00},
		{-0.10, 6.39, 0.00},
		{-0.41, 5.98, 0.00},
		{-0.58, 5.59, 0.00},
		{-0.36, 4.85, 0.00},
		{-0.09, 4.76, 0.00},
		{-0.23, 8.52, 0.00},
		{-0.14, 9.25, 0.00},
		{0.22, 9.32, 0.00},
		{-0.01, 8.98, 0.00},
		{-0.07, 8.58, 0.00},
		{-0.06, 8.03, 0.00},
		{0.06, 7.86, 0.00},
		{-0.17, 6.09, 0.00},
		{-0.02, 5.01, 0.00},
		{0.03, 4.80, 0.00},
		{-0.04, 4.68, 0.00},
		{-0.16, 4.61, 0.00},
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
				vk::Extent2D {1000, 1000},
				extensions, vk::AttachmentLoadOp::eLoad
			),
			tracer(get_context(),
				vk::AttachmentLoadOp::eClear,
				1000, 1000
			),
			font_renderer(get_context(),
				render_pass,
				"resources/fonts/noto_sans.ttf"
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
		
		// Setup Wadjet tracer
		KOBRA_LOG_FILE(kobra::Log::INFO) << "Hybrid tracer setup\n";
		wadjet_tracer = kobra::layers::Wadjet::make(get_context());
		kobra::layers::set_envmap(wadjet_tracer, "resources/skies/background_1.jpg");

#ifdef RECORD

		// Ask for destination
		std::string video_name;
		std::cout << "Enter video name: ";
		std::cin >> video_name;

		// Setup capture
		capture.open(
			("data/" + video_name + ".mp4"),
			cv::VideoWriter::fourcc('A', 'V', 'C', '1'),
			60, cv::Size(1000, 1000)
		);

		if (!capture.isOpened())
			std::cout << "Failed to open capture" << std::endl;
		else
			std::cout << "Capture opened" << std::endl;

#endif

		// Fill in time intervals
		for (int i = 0; i < camera_pos.size(); i++)
			times.push_back(i);

		camera_pos_seq.times = times;
		camera_rot_seq.times = times;
			
		// Input callbacks
		io.mouse_events.subscribe(mouse_callback, this);
		io.keyboard_events.subscribe(keyboard_callback, this);
	}

	float time = 0.0f;
	int mode = 0;

	std::queue <bool> events;
	std::mutex events_mutex;

	void record(const vk::raii::CommandBuffer &cmd,
			const vk::raii::Framebuffer &framebuffer) override {
		// Move the camera
		auto &transform = camera.get <kobra::Transform> ();
		
		// Interpolate camera position
		/* glm::vec3 pos = kobra::core::piecewise_linear(camera_pos_seq, time);
		glm::vec3 rot = kobra::core::piecewise_linear(camera_rot_seq, time);

		transform.position = pos;
		transform.rotation = rot; */

		float speed = 20.0f * frame_time;
		
		glm::vec3 forward = transform.forward();
		glm::vec3 right = transform.right();
		glm::vec3 up = transform.up();

		bool accumulate = true;

		if (io.input.is_key_down(GLFW_KEY_W)) {
			transform.move(forward * speed);
			accumulate = false;
		} else if (io.input.is_key_down(GLFW_KEY_S)) {
			transform.move(-forward * speed);
			accumulate = false;
		}

		if (io.input.is_key_down(GLFW_KEY_A)) {
			transform.move(-right * speed);
			accumulate = false;
		} else if (io.input.is_key_down(GLFW_KEY_D)) {
			transform.move(right * speed);
			accumulate = false;
		}

		if (io.input.is_key_down(GLFW_KEY_E)) {
			transform.move(up * speed);
			accumulate = false;
		} else if (io.input.is_key_down(GLFW_KEY_Q)) {
			transform.move(-up * speed);
			accumulate = false;
		}

		// Also check our events
		events_mutex.lock();
		if (!events.empty())
			accumulate = false; // Means that camera direction
					    // changed

		events = std::queue <bool> (); // Clear events
		events_mutex.unlock();

		// Now trace and render
		cmd.begin({});
			if (mode) {
				for (int i = 0; i < 1; i++)
					tracer.compute(scene.ecs);
				tracer.render(cmd, framebuffer);
				tracer.capture(frame);
			} else {
				kobra::layers::compute(wadjet_tracer,
					scene.ecs,
					camera.get <kobra::Camera> (),
					camera.get <kobra::Transform> (),
					accumulate
				);

				kobra::layers::render(wadjet_tracer, cmd, framebuffer);
				kobra::layers::capture(wadjet_tracer, frame);
			}

			// Text to render
			kobra::ui::Text t_fps(
				kobra::common::sprintf("%.2f fps", 1.0f/frame_time),
				{5, 5}, glm::vec3 {0.8, 0.8, 1}, 0.7f
			);
		
			// Start render pass
			// TODO: Make this a function
			std::array <vk::ClearValue, 2> clear_values = {
				vk::ClearValue {
					vk::ClearColorValue {
						std::array <float, 4> {0.0f, 0.0f, 0.0f, 1.0f}
					}
				},
				vk::ClearValue {
					vk::ClearDepthStencilValue {
						1.0f, 0
					}
				}
			};

			cmd.beginRenderPass(
				vk::RenderPassBeginInfo {
					*render_pass,
					*framebuffer,
					vk::Rect2D {
						vk::Offset2D {0, 0},
						extent,
					},
					static_cast <uint32_t> (clear_values.size()),
					clear_values.data()
				},
				vk::SubpassContents::eInline
			);

			font_renderer.render(cmd, {t_fps});

			cmd.endRenderPass();
		cmd.end();

#ifdef RECORD

		// Write the frame to the video
		cv::Mat mat(1000, 1000, CV_8UC4, frame.data());
		cv::cvtColor(mat, mat, cv::COLOR_BGRA2RGB);
		capture.write(mat);

		if (time > camera_pos.size() - 1) {
			capture.release();
			terminate_now();
		}

#endif

		// Update time (fixed)
		time += 1/60.0f;
	}
	
	// Mouse callback
	static void mouse_callback(void *us, const kobra::io::MouseEvent &event) {
		static const int pan_button = GLFW_MOUSE_BUTTON_MIDDLE;

		static const float sensitivity = 0.001f;

		static float px = 0.0f;
		static float py = 0.0f;

		static glm::vec2 previous_dir {0.0f, 0.0f};

		static float yaw = 0.0f;
		static float pitch = 0.0f;

		auto &app = *static_cast <MotionCapture *> (us);
		auto &transform = app.camera.get <kobra::Transform> ();

		// Deltas and directions
		float dx = event.xpos - px;
		float dy = event.ypos - py;
		glm::vec2 dir {dx, dy};
		
		// Check if panning
		static bool dragging = false;

		bool is_drag_button = (event.button == pan_button);
		if (event.action == GLFW_PRESS && is_drag_button)
			dragging = true;
		else if (event.action == GLFW_RELEASE && is_drag_button)
			dragging = false;

		// Pan only when dragging
		if (dragging) {
			yaw -= dx * sensitivity;
			pitch -= dy * sensitivity;

			if (pitch > 89.0f)
				pitch = 89.0f;
			if (pitch < -89.0f)
				pitch = -89.0f;

			transform.rotation.x = pitch;
			transform.rotation.y = yaw;

			// Add to event queue
			app.events_mutex.lock();
			app.events.push(true);
			app.events_mutex.unlock();
		}

		// Update previous position
		px = event.xpos;
		py = event.ypos;

		previous_dir = dir;
	}
	
	// Keyboard callback
	static void keyboard_callback(void *us, const kobra::io::KeyboardEvent &event) {
		auto &app = *static_cast <MotionCapture *> (us);
		auto &transform = app.camera.get <kobra::Transform> ();

		// M to switch between modes
		if (event.key == GLFW_KEY_M && event.action == GLFW_PRESS)
			app.mode = 1 - app.mode;
	}

};

#endif

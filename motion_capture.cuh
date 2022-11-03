#ifndef MOTION_CAPTURE_H_
#define MOTION_CAPTURE_H_

// Standard headers
#include <thread>

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

// TODO: do base app without inheritance (simple struct..., app and baseapp not
// related)
struct MotionCapture : public kobra::BaseApp {
	// TODO: let the scene run on any virtual device?
	kobra::Entity camera;
	kobra::Scene scene;

	// Necessary layers
	kobra::layers::Basilisk tracer;
	kobra::layers::Denoiser denoiser;
	kobra::layers::Framer framer;
	kobra::layers::FontRenderer font_renderer;
	// TODO: denoising layer which takes CUDA buffer as input

	// Buffers
	CUdeviceptr b_traced;
	std::vector <uint32_t> b_traced_cpu;

	// Capture
	cv::VideoWriter capture;
	std::vector <byte> frame;

	std::string capture_path;
	int max_samples = 0;

	// Threads
	std::thread *compute_thread = nullptr;

	kobra::Timer compute_timer;
	float compute_time;

	std::queue <bool> events;
	std::mutex events_mutex;
	bool kill = false;

	MotionCapture(const vk::raii::PhysicalDevice &phdev,
			const std::vector <const char *> &extensions,
			const std::string &scene_path)
			: BaseApp(phdev, "MotionCapture",
				vk::Extent2D {1000, 1000},
				extensions, vk::AttachmentLoadOp::eLoad
			),
			font_renderer(get_context(),
				render_pass,
				"resources/fonts/noto_sans.ttf"
			) {
		// Load scene and camera
		scene.load(get_device(), scene_path);

		// Add extra lights
		float width = 10.0f;
		float height = 10.0f;
		float depth = 10.0f;

		float stride = 5.0f;

		/* auto lighter = [&](float x, float y, float z) {
			kobra::Entity light = scene.ecs.make_entity();

			light.get <kobra::Transform> ().position = glm::vec3 {x, y, z};
			light.get <kobra::Transform> ().scale = glm::vec3 {0.1f};

			// Add emissive box
			kobra::Mesh box = kobra::Mesh::box({},
					{1, 1, 1});

			light.add <kobra::Mesh> (box);

			kobra::Mesh *mesh = &light.get <kobra::Mesh> ();
			mesh->submeshes[0].material.emission = glm::vec3 {100.0f};
			mesh->submeshes[0].material.diffuse = {0, 0, 0};
			mesh->submeshes[0].material.type = Shading::eEmissive;

			light.add <kobra::Rasterizer> (get_device(), mesh);

			// TODO: material update method for rasterizers,
			// adds tasks to a queue or something...?
			kobra::Rasterizer *rasterizer = &light.get <kobra::Rasterizer> ();
		};

		for (float x = 0; x <= width; x += stride) {
			for (float y = 0; y <= height; y += stride) {
				for (float z = 0; z <= depth; z += stride)
					lighter(x, y, z);
			}
		} */

		camera = scene.ecs.get_entity("Camera");

		// Setup Wadjet tracer
		KOBRA_LOG_FILE(kobra::Log::INFO) << "Hybrid tracer setup\n";
		tracer = kobra::layers::Basilisk::make(get_context());
		kobra::layers::set_envmap(tracer, "resources/skies/background_1.jpg");

		// Create the denoiser layer
		denoiser = kobra::layers::Denoiser::make(
			extent,
			kobra::layers::Denoiser::eNormal
				| kobra::layers::Denoiser::eAlbedo
		);

		framer = kobra::layers::Framer::make(get_context());

#if 1

		std::cout << "Enter capture path: ";
		std::cin >> capture_path;

		if (capture_path.empty()) {
			// Assume defaults
			capture_path = "capture.png";
			max_samples = 100000;
		} else {
			std::cout << "Enter max samples: ";
			std::cin >> max_samples;
		}

#else

		capture_path = "capture.png";
		max_samples = 100000;

#endif

		std::cout << "Path tracing " << max_samples << " samples to " << capture_path << "\n";

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
			
		// Input callbacks
		io.mouse_events.subscribe(mouse_callback, this);
		io.keyboard_events.subscribe(keyboard_callback, this);
			
		// NOTE: we need this precomputation step to load all the
		// resources before rendering; we need some system to allocate
		// queues so that we dont need to do this...
		kobra::layers::compute(tracer,
			scene.ecs,
			camera.get <kobra::Camera> (),
			camera.get <kobra::Transform> (),
			mode, false 
		);
			
		// Launch compute thread
		compute_thread = new std::thread(
			&MotionCapture::path_trace_kernel, this
		);

		// Allocate buffers
		size_t size = kobra::layers::size(tracer);

		b_traced = kobra::cuda::alloc(size * sizeof(uint32_t));
		b_traced_cpu.resize(size);
	}

	// Destructor
	~MotionCapture() {
		// Wait for compute thread to finish
		if (compute_thread) {
			// Send kill signal
			kill = true;
			compute_thread->join();
			delete compute_thread;
		}
	}

	// Path tracing kernel
	void path_trace_kernel() {
		compute_timer.start();
		while (!kill) {
			bool accumulate = true;

			// Also check our events
			events_mutex.lock();
			if (!events.empty())
				accumulate = false; // Means that camera direction
						    // changed

			events = std::queue <bool> (); // Clear events
			events_mutex.unlock();

			kobra::layers::compute(tracer,
				scene.ecs,
				camera.get <kobra::Camera> (),
				camera.get <kobra::Transform> (),
				mode, accumulate
			);

			/* kobra::layers::denoise(denoiser, {
				.color = kobra::layers::color_buffer(tracer),
				.normal = kobra::layers::normal_buffer(tracer),
				.albedo = kobra::layers::albedo_buffer(tracer)
			}); */

			compute_time = compute_timer.lap()/1e6;
		}
	}

	float time = 0.0f;
	unsigned int mode = kobra::optix::eRegular;

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
			events.push(true);
		} else if (io.input.is_key_down(GLFW_KEY_S)) {
			transform.move(-forward * speed);
			events.push(true);
		}

		if (io.input.is_key_down(GLFW_KEY_A)) {
			transform.move(-right * speed);
			events.push(true);
		} else if (io.input.is_key_down(GLFW_KEY_D)) {
			transform.move(right * speed);
			events.push(true);
		}

		if (io.input.is_key_down(GLFW_KEY_E)) {
			transform.move(up * speed);
			events.push(true);
		} else if (io.input.is_key_down(GLFW_KEY_Q)) {
			transform.move(-up * speed);
			events.push(true);
		}

		// Now trace and render
		cmd.begin({});
			int width = tracer.extent.width;
			int height = tracer.extent.height;

			kobra::cuda::hdr_to_ldr(
				(float4 *) kobra::layers::color_buffer(tracer),
				(uint32_t *) b_traced,
				width, height,
				kobra::cuda::eTonemappingACES
			);
			

			kobra::cuda::copy(b_traced_cpu, b_traced, width * height);
			kobra::layers::render(framer, b_traced_cpu, cmd, framebuffer);

			// kobra::layers::capture(tracer, frame);

			// Text to render
			kobra::ui::Text t_fps(
				kobra::common::sprintf("%.2f fps", 1.0f/compute_time),
				{5, 5}, glm::vec3 {1, 0.6, 0.6}, 0.7f
			);

			kobra::ui::Text t_samples(
				kobra::common::sprintf("%d samples", tracer.launch_params.samples),
				{0, 5}, glm::vec3 {1, 0.6, 0.6}, 0.7f
			);

			kobra::ui::Text t_mode(
				kobra::common::sprintf("Mode: %s", kobra::optix::str_modes[mode]),
				{5, 45}, glm::vec3 {1, 0.6, 0.6}, 0.5f
			);

			glm::vec2 size = font_renderer.size(t_samples);
			t_samples.anchor.x = 1000 - size.x - 5;
		
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

			font_renderer.render(cmd, {t_fps, t_samples, t_mode});

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

	void terminate() override {
		if (tracer.launch_params.samples > max_samples) {
			// Get data to save
			int width = tracer.extent.width;
			int height = tracer.extent.height;

			kill = true;
			compute_thread->join();

			stbi_write_png(capture_path.c_str(),
				width, height, 4, b_traced_cpu.data(),
				width * 4
			);
		
			KOBRA_LOG_FILE(kobra::Log::INFO) << "Saved image to "
				<< capture_path << "\n";

			terminate_now();
		}
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
		static bool alt_dragging = false;

		bool is_drag_button = (event.button == pan_button);
		if (event.action == GLFW_PRESS && is_drag_button)
			dragging = true;
		else if (event.action == GLFW_RELEASE && is_drag_button)
			dragging = false;

		bool is_alt_down = app.io.input.is_key_down(GLFW_KEY_LEFT_ALT);
		if (!alt_dragging && is_alt_down)
			alt_dragging = true;
		else if (alt_dragging && !is_alt_down)
			alt_dragging = false;

		// Pan only when dragging
		if (dragging || alt_dragging) {
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
		if (event.key == GLFW_KEY_M && event.action == GLFW_PRESS) {
			app.mode = (app.mode + 1) % kobra::optix::eCount;
			
			// Add to event queue
			app.events_mutex.lock();
			app.events.push(true);
			app.events_mutex.unlock();
		}

		// Numbers 1-3 for the same function
		if (event.key >= GLFW_KEY_1 && event.key <= GLFW_KEY_3 && event.action == GLFW_PRESS) {
			app.mode = event.key - GLFW_KEY_1;
			
			// Add to event queue
			app.events_mutex.lock();
			app.events.push(true);
			app.events_mutex.unlock();
		}

		// I for info
		if (event.key == GLFW_KEY_I && event.action == GLFW_PRESS) {
			std::cout << "Camera transform:\n";
			std::cout << "\tPosition: " << transform.position.x << ", " << transform.position.y << ", " << transform.position.z << "\n";
			std::cout << "\tRotation: " << transform.rotation.x << ", " << transform.rotation.y << ", " << transform.rotation.z << "\n";
		}
	}

};

#endif

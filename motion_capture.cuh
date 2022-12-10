#ifndef MOTION_CAPTURE_H_
#define MOTION_CAPTURE_H_

// Standard headers
#include <set>
#include <string>
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
#include "include/cuda/alloc.cuh"
#include "include/cuda/color.cuh"
#include "include/layers/basilisk.cuh"
#include "include/layers/denoiser.cuh"
#include "include/layers/font_renderer.hpp"
#include "include/layers/framer.hpp"
#include "include/layers/hybrid_tracer.cuh"
#include "include/layers/imgui.hpp"
#include "include/layers/wssr.cuh"
#include "include/optix/options.cuh"
#include "include/optix/parameters.cuh"
#include "include/renderer.hpp"
#include "include/scene.hpp"
#include "include/texture.hpp"
#include "include/ui/attachment.hpp"
#include "include/ui/framerate.hpp"

// #include "include/amadeus/backend.cuh"

// TODO: do base app without inheritance (simple struct..., app and baseapp not related)
// TODO: and then insert layers with .attach() method
struct MotionCapture : public kobra::BaseApp {
	// TODO: let the scene run on any virtual device?
	kobra::Entity camera;
	kobra::Scene scene;

	// Necessary layers
	kobra::layers::Basilisk tracer;
	kobra::layers::Denoiser denoiser;
	kobra::layers::Framer framer;
	kobra::layers::ImGUI imgui;

	// TODO: GPU utilization (mem and compute) monitor

	kobra::asmodeus::GridBasedReservoirs grid_based;

	// Buffers
	CUdeviceptr b_traced;
	std::vector <uint8_t> b_traced_cpu;

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
	bool capture_now = false;
	bool lock_motion = false;

	static constexpr vk::Extent2D raytracing_extent = {1000, 1000};
	static constexpr vk::Extent2D rendering_extent = {1920, 1080};
	// TODO: try different rendering extent

	struct CaptureInterface : kobra::ui::ImGUIAttachment {
		std::string m_mode_description;
		std::set <std::string> m_additional_descriptions;
		int m_samples = 1;
		int m_captured_samples = -1;
		char m_capture_path[256] = "capture.png";

		MotionCapture *m_parent = nullptr;

		void render() override {
			ImGui::Begin("Capture");

			ImGui::Text("Mode: %s", m_mode_description.c_str());

			// TODO: check boxes...
			std::string additional_description;
			for (auto &description : m_additional_descriptions)
				additional_description += description + ", ";

			ImGui::Text("Additional: %s", additional_description.c_str());

			// Slider to control the number of samples
			ImGui::SliderInt("Samples", &m_samples, 1, 4096);

			// Text field to control the capture path
			ImGui::InputText("Capture path", m_capture_path, 256);

			if (m_captured_samples < 0 && ImGui::Button("Capture")) {
				m_captured_samples = m_samples;
				m_parent->events_mutex.lock();
				m_parent->events.push(true);
				m_parent->events_mutex.unlock();
			} else if (m_captured_samples >= 0) {
				// Progress bar
				ImGui::ProgressBar((float) m_captured_samples/m_samples, ImVec2(0.0f, 0.0f));
			}

			ImGui::End();
		}
	};

	std::shared_ptr <CaptureInterface> capture_interface;

	MotionCapture(const vk::raii::PhysicalDevice &phdev,
			const std::vector <const char *> &extensions,
			const std::string &scene_path)
			: BaseApp(phdev, "MotionCapture",
				rendering_extent,
				extensions, vk::AttachmentLoadOp::eLoad
			),
			framer(get_context()) {
		// Load scene and camera
		scene.load(get_device(), scene_path);

		// TODO: save mesh source...
		// scene.save("scene.kobra");

		camera = scene.ecs.get_entity("Camera");

		// TODO: test lower resolution...
		tracer = kobra::layers::Basilisk::make(get_context(), raytracing_extent);
		
		kobra::layers::set_envmap(tracer, "resources/skies/background_1.jpg");

		// Create the denoiser layer
		denoiser = kobra::layers::Denoiser::make(
			extent,
			kobra::layers::Denoiser::eNormal
				| kobra::layers::Denoiser::eAlbedo
		);

		// Create Asmodeus backend
		grid_based = kobra::asmodeus::GridBasedReservoirs::make(
			get_context(), raytracing_extent
		);

		grid_based.set_envmap("resources/skies/background_1.jpg");

#if 0

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
		
		// Initialize ImGUI
		// TODO: ui/core method (initialize)
		ImGui::CreateContext();
		ImPlot::CreateContext();

		ImGui_ImplGlfw_InitForVulkan(window.handle, true);

		imgui = kobra::layers::ImGUI(get_context(), window, graphics_queue);
		imgui.set_font(KOBRA_DIR "/resources/fonts/NotoSans.ttf", 30);

		capture_interface = std::make_shared <CaptureInterface> ();
		imgui.attach(std::make_shared <kobra::ui::FramerateAttachment> ());
		imgui.attach(capture_interface);

		capture_interface->m_parent = this;
		
		// NOTE: we need this precomputation step to load all the
		// resources before rendering; we need some system to allocate
		// queues so that we dont need to do this...
		kobra::layers::compute(tracer,
			scene.ecs,
			camera.get <kobra::Camera> (),
			camera.get <kobra::Transform> (),
			mode, false 
		);

		/* grid_based.render(scene.ecs,
			camera.get <kobra::Camera> (),
			camera.get <kobra::Transform> (),
			false
		); */

		compute_thread = new std::thread(
			&MotionCapture::path_trace_kernel, this
		);

		// Allocate buffers
		size_t size = grid_based.size(); // kobra::layers::size(tracer);

		b_traced = kobra::cuda::alloc(size * sizeof(uint32_t));
		b_traced_cpu.resize(size);

		mode_map.at(5)();
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
	int integrator = 0;

	void path_trace_kernel() {	
		compute_timer.start();
		while (!kill) {
			// Wait for the latest capture if any
			while (!capture_now);

			bool accumulate = true;

			// Also check our events
			events_mutex.lock();
			if (!events.empty())
				accumulate = false; // Means that camera direction
						    // changed

			events = std::queue <bool> (); // Clear events
			events_mutex.unlock();

			if (integrator == 0) {
				kobra::layers::compute(tracer,
					scene.ecs,
					camera.get <kobra::Camera> (),
					camera.get <kobra::Transform> (),
					mode, accumulate
				);
			} else {
				grid_based.render(scene.ecs,
					camera.get <kobra::Camera> (),
					camera.get <kobra::Transform> (),
					accumulate
				);
			}
			
			kobra::layers::denoise(denoiser, {
				.color = grid_based.color_buffer(),
				.normal = grid_based.normal_buffer(),
				.albedo = grid_based.albedo_buffer()
			});

			compute_time = compute_timer.lap()/1e6;

			if (capture_interface->m_captured_samples >= 0)
				capture_now = (--capture_interface->m_captured_samples <= 0);
		}
	}

	float time = 0.0f;

	unsigned int mode = kobra::optix::eRegular;

	// Mode map for Basilisk
	// TODO: turn into keybindings
	const std::unordered_map <int, std::function <void ()>> mode_map {
		{1, [&]() {
			mode = kobra::optix::eRegular;
			capture_interface->m_mode_description = "Regular";
		}},

		{2, [&]() {
			mode = kobra::optix::eReSTIR;
			tracer.launch_params.options.reprojected_reuse = false;
			capture_interface->m_mode_description = "ReSTIR";
		}},

		{3, [&]() {
			mode = kobra::optix::eReSTIR;
			tracer.launch_params.options.reprojected_reuse = true;
			capture_interface->m_mode_description = "ReSTIR (reprojected reuse)";
		}},

		{4, [&]() {
			mode = kobra::optix::eVoxel;
			capture_interface->m_mode_description = "WSSR using K-d tree";
		}},

		{5, [&]() {
			mode = kobra::optix::eReSTIRPT;
			capture_interface->m_mode_description = "ReSTIR Path Tracing";
		}},

		{6, [&]() {
			bool &b = tracer.launch_params.options.indirect_only;
			b = !b;

			if (b)
				capture_interface->m_additional_descriptions.insert("Indirect Only");
			else
				capture_interface->m_additional_descriptions.erase("Indirect Only");
		}},

		{7, [&]() {
			// TODO: disabled accumulation should just reset every
			// frame...
			bool &b = tracer.launch_params.options.disable_accumulation;
			b = !b;

			if (b)
				capture_interface->m_additional_descriptions.insert("No Accumulation");
			else
				capture_interface->m_additional_descriptions.erase("No Accumulation");
		}},

		{8, [&]() {
			integrator = 1 - integrator;

			if (integrator == 1) {
				capture_interface->m_mode_description = "WSSR using Grid";
			} else {
				// TODO: get a string representation of the
				// integrator...
				capture_interface->m_mode_description = "Basilisk";
			}
		}},

		{9, [&]() {
			auto &gb_ris = grid_based.launch_params.gb_ris;
			gb_ris.reproject = !gb_ris.reproject;
		}}
	};

	void record(const vk::raii::CommandBuffer &cmd,
			const vk::raii::Framebuffer &framebuffer) override {
		// Move the camera
		auto &transform = camera.get <kobra::Transform> ();

		// TODO: motion method
		float speed = 20.0f * frame_time;
		
		glm::vec3 forward = transform.forward();
		glm::vec3 right = transform.right();
		glm::vec3 up = transform.up();

		bool accumulate = true;

		if (!lock_motion) {
			// TODO: method...
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
		}

		// Now trace and render
		cmd.begin({});
			unsigned int width = grid_based.extent.width;
			unsigned int height = grid_based.extent.height;

			float4 *d_output = 0;
			if (integrator == 0) {
				d_output = (float4 *) kobra::layers::color_buffer(tracer);
			} else {
				// d_output = (float4 *) kobra::asmodeus::color_buffer(wskdr);
				d_output = (float4 *) grid_based.color_buffer();
			}

			// TODO: denoise here?
			// d_output = (float4 *) denoiser.result;

			kobra::cuda::hdr_to_ldr(
				d_output,
				(uint32_t *) b_traced,
				width, height,
				kobra::cuda::eTonemappingACES
			);

			kobra::cuda::copy(b_traced_cpu, b_traced, width * height * sizeof(uint32_t));

			// TODO: import CUDA to Vulkan and render straight to the image
			framer.render(
				kobra::Image {
					.data = b_traced_cpu,
					.width = width,
					.height = height,
					.channels = 4
				},
				cmd, framebuffer, extent,
				// TODO: embed in a docked ImGui window
				{{420, 0}, {1080 + 420, 1080}}
			);

			imgui.render(cmd, framebuffer, extent);
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

		if (capture_now) {
			int width = raytracing_extent.width;
			int height = raytracing_extent.height;

			stbi_write_png(capture_interface->m_capture_path,
				width, height, 4,
				b_traced_cpu.data(),
				width * 4
			);

			capture_now = false;
		}

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
		static const int pan_button = GLFW_MOUSE_BUTTON_RIGHT;

		static const float sensitivity = 0.001f;

		static float px = 0.0f;
		static float py = 0.0f;

		static float yaw = 0.0f;
		static float pitch = 0.0f;

		auto &app = *static_cast <MotionCapture *> (us);
		auto &transform = app.camera.get <kobra::Transform> ();

		// Deltas and directions
		float dx = event.xpos - px;
		float dy = event.ypos - py;
		
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
		if (dragging | alt_dragging) {
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

		if (event.key >= GLFW_KEY_0 && event.key <= GLFW_KEY_9 && event.action == GLFW_PRESS) {
			int key = event.key - GLFW_KEY_0;

			// Execute mode map function
			if (app.mode_map.count(key) > 0) {
				app.mode_map.at(key)();
			
				// Add to event queue
				app.events_mutex.lock();
				app.events.push(true);
				app.events_mutex.unlock();
			} else {
				KOBRA_LOG_FILE(kobra::Log::WARN) << "No mode map for key "
					<< key << "\n";
			}
		}

		// I for info
		if (event.key == GLFW_KEY_I && event.action == GLFW_PRESS) {
			printf("\n{%.2f, %.2f, %.2f}\n", transform.position.x, transform.position.y, transform.position.z);
			printf("{%.2f, %.2f, %.2f}\n", transform.rotation.x, transform.rotation.y, transform.rotation.z);
		}

		// F1 to lock motion
		if (event.key == GLFW_KEY_F1 && event.action == GLFW_PRESS) {
			app.lock_motion = !app.lock_motion;
		}
	}

};

#endif

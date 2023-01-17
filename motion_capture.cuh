#ifndef MOTION_CAPTURE_H_
#define MOTION_CAPTURE_H_

// Standard headers
#include <set>
#include <string>
#include <thread>
#include <atomic>

// GLM headers
#include <glm/glm.hpp>

// OpenCV for video capture
#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>

// NVIDIA diagnostics
#include <nvml.h>

// Engine headers
#include "include/amadeus/system.cuh"
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
#include "include/layers/ui.hpp"
#include "include/layers/wssr.cuh"
#include "include/optix/options.cuh"
#include "include/optix/parameters.cuh"
#include "include/scene.hpp"
#include "include/texture.hpp"
#include "include/ui/attachment.hpp"
#include "include/ui/framerate.hpp"
#include "include/layers/mesh_memory.hpp"
#include "include/amadeus/armada.cuh"
#include "include/amadeus/path_tracer.cuh"
#include "include/amadeus/repg.cuh"
#include "include/amadeus/restir.cuh"

// TODO: do base app without inheritance (simple struct..., app and baseapp not related)
// TODO: and then insert layers with .attach() method
struct MotionCapture : public kobra::BaseApp {
	// TODO: let the scene run on any virtual device?
	kobra::Entity camera;
	kobra::Scene scene;

	// Necessary layers
	kobra::layers::Denoiser denoiser;
	kobra::layers::Framer framer;
	kobra::layers::UI ui;
	
	std::shared_ptr <kobra::layers::MeshMemory> mesh_memory;
	std::shared_ptr <kobra::amadeus::System> amadeus;

	std::shared_ptr <kobra::amadeus::ArmadaRTX> armada_rtx;

	// Buffers
	CUdeviceptr b_traced;
	std::vector <uint8_t> b_traced_cpu;

	// Threads
	std::thread *compute_thread = nullptr;

	kobra::Timer compute_timer;
	float compute_time;

	std::queue <bool> events;
	std::mutex events_mutex;
	std::atomic <int> semaphore = 0;
	bool kill = false;
	bool capture_now = false;
	bool lock_motion = false;

	static constexpr vk::Extent2D raytracing_extent = {1000, 1000};
	static constexpr vk::Extent2D rendering_extent = {1920, 1080};

	struct GPUUsageMonitor : kobra::ui::ImGuiAttachment {
		void render() {
			// TODO: graph memory usage over time
			nvmlDevice_t device;
			nvmlReturn_t result = nvmlDeviceGetHandleByIndex(0, &device);

			nvmlUtilization_t utilization;
			result = nvmlDeviceGetUtilizationRates(device, &utilization);

			nvmlMemory_t memory;
			result = nvmlDeviceGetMemoryInfo(device, &memory);

			ImGui::Begin("GPU Usage");
			ImGui::Text("GPU usage: %d%%", utilization.gpu);
			ImGui::Text("Memory used: %llu/%llu MiB",
				memory.used/(1024ull * 1024ull),
				memory.total/(1024ull * 1024ull)
			);

			ImGui::End();
		}
	};

	struct CaptureInterface : kobra::ui::ImGuiAttachment {
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

		// Create the layers
		amadeus = std::make_shared <kobra::amadeus::System> ();
		mesh_memory = std::make_shared <kobra::layers::MeshMemory> (get_context());

		// TODO: test lower resolution...
		armada_rtx = std::make_shared <kobra::amadeus::ArmadaRTX> (
			get_context(), amadeus,
			mesh_memory, raytracing_extent
		);

		armada_rtx->attach(
			"Path Tracer",
			std::make_shared <kobra::amadeus::PathTracer> ()
		);
		
		armada_rtx->attach(
			"ReSTIR",
			std::make_shared <kobra::amadeus::ReSTIR> ()
		);

		armada_rtx->attach(
			"RePG",
			std::make_shared <kobra::amadeus::RePG> ()
		);
		
		armada_rtx->set_envmap(KOBRA_DIR "/resources/skies/background_1.jpg");

		// Create the denoiser layer
		denoiser = kobra::layers::Denoiser::make(
			extent,
			kobra::layers::Denoiser::eNormal
				| kobra::layers::Denoiser::eAlbedo
		);
			
		// Input callbacks
		io.mouse_events.subscribe(mouse_callback, this);
		io.keyboard_events.subscribe(keyboard_callback, this);
		
		// Initialize ImGUI
		// TODO: ui/core method (initialize)
		ImGui::CreateContext();
		ImPlot::CreateContext();

		ImGui_ImplGlfw_InitForVulkan(window.handle, true);

		ui = kobra::layers::UI(get_context(), window, graphics_queue);
		ui.set_font(KOBRA_DIR "/resources/fonts/NotoSans.ttf", 30);

		capture_interface = std::make_shared <CaptureInterface> ();

		ui.attach(capture_interface);
		ui.attach(std::make_shared <GPUUsageMonitor> ());
		ui.attach(std::make_shared <kobra::ui::FramerateAttachment> (
			[&]() {
				return 1.0f/compute_time;
			}
		));

		capture_interface->m_parent = this;
		
		// NOTE: we need this precomputation step to load all the
		// resources before rendering; we need some system to allocate
		// queues so that we dont need to do this...
		armada_rtx->render(
			scene.ecs,
			camera.get <kobra::Camera> (),
			camera.get <kobra::Transform> (),
			false 
		);

		compute_thread = new std::thread(
			&MotionCapture::path_trace_kernel, this
		);
		
		KOBRA_LOG_FILE(kobra::Log::INFO) << "Launched path tracing thread\n";

		// Allocate buffers
		size_t size = armada_rtx->size();

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
	int integrator = 0;

	void path_trace_kernel() {	
		compute_timer.start();

		while (!kill) {	
			// Wait for the latest capture if any
			/* while (!capture_now) {
				std::cout << "\twaiting for capture, capture_now: " << std::boolalpha << capture_now << "\n";
			} */
			bool accumulate = true;

			// Also check our events
			events_mutex.lock();
			if (!events.empty())
				accumulate = false; // Means that camera direction
						    // changed

			events = std::queue <bool> (); // Clear events
			events_mutex.unlock();

			armada_rtx->render(
				scene.ecs,
				camera.get <kobra::Camera> (),
				camera.get <kobra::Transform> (),
				accumulate
			);
			
			/* kobra::layers::denoise(denoiser, {
				.color = grid_based.color_buffer(),
				.normal = grid_based.normal_buffer(),
				.albedo = grid_based.albedo_buffer()
			}); */

			compute_time = compute_timer.lap()/1e6;
			if (capture_interface->m_captured_samples >= 0)
				capture_now = (--capture_interface->m_captured_samples <= 0);
		}
	}

	float time = 0.0f;

	// Keybindings
	// TODO: separate header/class
	const std::unordered_map <int, std::function <void ()>> mode_map {
		{1, [&]() {
			armada_rtx->activate("Path Tracer");
		}},
		{2, [&]() {
			armada_rtx->activate("ReSTIR");
		}},
		{3, [&]() {
			armada_rtx->activate("RePG");
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
			vk::Extent2D rtx_extent = armada_rtx->extent();

			kobra::cuda::hdr_to_ldr(
				(float4 *) armada_rtx->color_buffer(),
				(uint32_t *) b_traced,
				rtx_extent.width, rtx_extent.height,
				kobra::cuda::eTonemappingACES
			);

			kobra::cuda::copy(
				b_traced_cpu, b_traced,
				armada_rtx->size() * sizeof(uint32_t)
			);

			// TODO: import CUDA to Vulkan and render straight to the image
			framer.render(
				kobra::Image {
					.data = b_traced_cpu,
					.width = rtx_extent.width,
					.height = rtx_extent.height,
					.channels = 4
				},
				cmd, framebuffer, extent,
				// TODO: embed in a docked ImGui window
				{{420, 0}, {1080 + 420, 1080}}
			);

			ui.render(cmd, framebuffer, extent);
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

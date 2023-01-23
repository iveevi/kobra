#include "../include/amadeus/armada.cuh"
#include "../include/amadeus/path_tracer.cuh"
#include "../include/amadeus/repg.cuh"
#include "../include/amadeus/restir.cuh"
#include "../include/app.hpp"
#include "../include/backend.hpp"
#include "../include/cuda/color.cuh"
#include "../include/layers/forward_renderer.hpp"
#include "../include/layers/framer.hpp"
#include "../include/layers/ui.hpp"
#include "../include/project.hpp"
#include "../include/scene.hpp"
#include "../include/ui/framerate.hpp"

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

	vk::raii::PhysicalDevice phdev = kobra::pick_physical_device(predicate);

	struct StressApp1 : public kobra::BaseApp {
		kobra::Scene scene;
		kobra::Entity camera;
		kobra::layers::ForwardRenderer forward_renderer;

		StressApp1(const vk::raii::PhysicalDevice &phdev,
				const std::vector <const char *> extensions)
				: kobra::BaseApp {
					phdev, "Stress Test",
					vk::Extent2D {500, 500},
					extensions
				} {
			// Setup forward renderer
			forward_renderer = kobra::layers::ForwardRenderer(get_context());

			kobra::Project project = kobra::Project::load(".kobra/project");
			scene.load(get_context(), project.scene);
			camera = scene.ecs.get_entity("Camera");
		}

		void record(const vk::raii::CommandBuffer &cmd,
				const vk::raii::Framebuffer &framebuffer) override {
			cmd.begin({});
				forward_renderer.render(
					scene.ecs,
					camera.get <kobra::Camera> (),
					camera.get <kobra::Transform> (),
					cmd, framebuffer
				);
			cmd.end();
		}
	};

	struct StressApp2 : public kobra::BaseApp {
		kobra::Scene scene;
		kobra::Entity camera;

		std::shared_ptr <kobra::layers::MeshMemory> mesh_memory;

		std::shared_ptr <kobra::amadeus::System> system;
		std::shared_ptr <kobra::amadeus::ArmadaRTX> armada;
	
		std::shared_ptr <kobra::layers::Framer> framer;
	
		std::shared_ptr <kobra::layers::UI> ui;
	
		// Buffers
		CUdeviceptr b_traced;
		std::vector <uint8_t> b_traced_cpu;

		StressApp2(const vk::raii::PhysicalDevice &phdev,
				const std::vector <const char *> extensions)
				: kobra::BaseApp {
					phdev, "Stress Test",
					vk::Extent2D {500, 500},
					extensions
				} {
			mesh_memory = std::make_shared <kobra::layers::MeshMemory> (get_context());
			system = std::make_shared <kobra::amadeus::System> ();
			armada = std::make_shared <kobra::amadeus::ArmadaRTX> (get_context(), system, mesh_memory, vk::Extent2D {500, 500});
			framer = std::make_shared <kobra::layers::Framer> (get_context());

			// Configure the armada
			armada->attach("Path Tracer", std::make_shared <kobra::amadeus::PathTracer> ());
			armada->attach("ReSTIR", std::make_shared <kobra::amadeus::ReSTIR> ());
			armada->attach("RePG", std::make_shared <kobra::amadeus::RePG> ());
			armada->set_envmap(KOBRA_DIR "/resources/skies/background_1.jpg");
				
			// Initialize ImGUI
			// TODO: ui/core method (initialize)
			ImGui::CreateContext();
			ImPlot::CreateContext();

			ImGui_ImplGlfw_InitForVulkan(window.handle, true);

			std::pair <std::string, size_t> font {KOBRA_DIR "/resources/fonts/NotoSans.ttf", 30};
			ui = std::make_shared <kobra::layers::UI> (get_context(), window, graphics_queue, font);
			ui->attach(std::make_shared <kobra::ui::FramerateAttachment> ());
		
			// Allocate buffers
			size_t size = armada->size();
			b_traced = kobra::cuda::alloc(size * sizeof(uint32_t));
			b_traced_cpu.resize(size);
			
			// Load the scene
			kobra::Project project = kobra::Project::load(".kobra/project");
			scene.load(get_context(), project.scene);
			camera = scene.ecs.get_entity("Camera");
			
			// Keyboard callbacks
			io.keyboard_events.subscribe(
				[](void *us, const kobra::io::KeyboardEvent &event) {
					static const std::vector <std::string> modes {
						"Path Tracer",
						"ReSTIR",
						"RePG"
					};

					StressApp2 *self = (StressApp2 *) us;

					if ((event.key >= GLFW_KEY_0 && event.key <= GLFW_KEY_9)
							&& event.action == GLFW_PRESS) {
						int index = event.key - GLFW_KEY_0;
						if (index <= modes.size())
							self->armada->activate(modes[index - 1]);
					}
				}, this
			);

			// Mouse callbacks
			io.mouse_events.subscribe(
				[&](void *us, const kobra::io::MouseEvent &event) {
					static const int pan_button = GLFW_MOUSE_BUTTON_RIGHT;

					static const float sensitivity = 0.001f;

					static float px = 0.0f;
					static float py = 0.0f;

					static float yaw = 0.0f;
					static float pitch = 0.0f;

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

					bool is_alt_down = io.input->is_key_down(GLFW_KEY_LEFT_ALT);
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

						kobra::Transform &transform = camera.get <kobra::Transform> ();
						transform.rotation.x = pitch;
						transform.rotation.y = yaw;
					}

					// Update previous position
					px = event.xpos;
					py = event.ypos;
				},
				// TODO: free this in the destructor
				this
			);
		}

		~StressApp2() {
			// Wait to finish
			device.waitIdle();

			// Clear handlers
			io.mouse_events.clear();

			// Always destroy the UI before the context
			ui.reset();

			// Clean up local resources
			kobra::cuda::free(b_traced);

			ImGui_ImplGlfw_Shutdown();
			ImGui::DestroyContext();
		}
		
		void record(const vk::raii::CommandBuffer &cmd,
				const vk::raii::Framebuffer &framebuffer) override {
			cmd.begin({});
				armada->render(
					scene.ecs,
					camera.get <kobra::Camera> (),
					camera.get <kobra::Transform> (),
					false
				);
			
				vk::Extent2D rtx_extent = armada->extent();

				kobra::cuda::hdr_to_ldr(
					(float4 *) armada->color_buffer(),
					(uint32_t *) b_traced,
					rtx_extent.width, rtx_extent.height,
					kobra::cuda::eTonemappingACES
				);

				kobra::cuda::copy(
					b_traced_cpu, b_traced,
					armada->size() * sizeof(uint32_t)
				);

				// TODO: import CUDA to Vulkan and render straight to the image
				framer->render(
					kobra::Image {
						.data = b_traced_cpu,
						.width = rtx_extent.width,
						.height = rtx_extent.height,
						.channels = 4
					},
					cmd, framebuffer, extent
				);
			
				ui->render(cmd, framebuffer, extent);
			cmd.end();
		}
	};

	StressApp1 app1 {
		phdev,
		{VK_KHR_SWAPCHAIN_EXTENSION_NAME},
	};
	
	StressApp2 app2 {
		phdev, {
			VK_KHR_SWAPCHAIN_EXTENSION_NAME,
			VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME,
			VK_KHR_EXTERNAL_MEMORY_EXTENSION_NAME
		}
	};

	app2.run();
}

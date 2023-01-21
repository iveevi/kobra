#include "../include/app.hpp"
#include "../include/backend.hpp"
#include "../include/scene.hpp"
#include "../include/project.hpp"
#include "../include/layers/forward_renderer.hpp"
#include "../include/amadeus/armada.cuh"
#include "../include/amadeus/path_tracer.cuh"

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

		StressApp2(const vk::raii::PhysicalDevice &phdev,
				const std::vector <const char *> extensions)
				: kobra::BaseApp {
					phdev, "Stress Test",
					vk::Extent2D {500, 500},
					extensions
				} {
			mesh_memory = std::make_shared <kobra::layers::MeshMemory> (get_context());
			system = std::make_shared <kobra::amadeus::System> ();
			armada = std::make_shared <kobra::amadeus::ArmadaRTX>
				(get_context(), system, mesh_memory, vk::Extent2D {500, 500});
		}
	};

	StressApp1 app {
		phdev,
		{VK_KHR_SWAPCHAIN_EXTENSION_NAME},
	};

	app.run();
}

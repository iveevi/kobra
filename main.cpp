#include "global.hpp"
#include "include/app.hpp"
#include "include/backend.hpp"
#include "include/ecs.hpp"
#include "include/engine/rt_capture.hpp"
#include "include/gui/button.hpp"
#include "include/gui/layer.hpp"
#include "include/gui/rect.hpp"
#include "include/gui/sprite.hpp"
#include "tinyfiledialogs.h"

using namespace kobra;

// Scene path
std::string scene_path = "scenes/room_simple.kobra";

// Test app
struct ECSApp : public BaseApp {
	ECSApp(const vk::raii::PhysicalDevice &phdev, const std::vector <const char *> &extensions)
			: BaseApp(phdev, "ECSApp", {1000, 1000}, extensions) {
		// ECS test
		ECS ecs;

		auto e1 = ecs.make_entity("e1");
		auto e2 = ecs.make_entity("e2");
		auto e3 = ecs.make_entity("e3");

		e2.get <Transform>().position = {1, 2, 3};

		std::cout << "Transform.position e1 = " << glm::to_string(e1.get <Transform> ().position) << std::endl;
		std::cout << "Transform.position e2 = " << glm::to_string(e2.get <Transform> ().position) << std::endl;


		ecs.info <Mesh> ();

		auto box = KMesh::make_box({1, 2, 3}, {4, 5, 6});
		auto submesh = Submesh(box.vertices(), box.indices());
		auto mesh = Mesh({submesh});

		e2.add <Mesh> (mesh);

		std::cout << "Does e1 have a mesh: " << std::boolalpha << e1.exists <Mesh> () << std::endl;
		std::cout << "Does e2 have a mesh: " << std::boolalpha << e2.exists <Mesh> () << std::endl;
		std::cout << "Mesh e2 #vertices = " << e2.get <Mesh> ().vertices() << std::endl;
		// e2.add <

		ecs.info <Mesh> ();
	}

	void record(const vk::raii::CommandBuffer &cmd, const vk::raii::Framebuffer &fb) override {}
};

int main()
{
	auto extensions = {
		VK_KHR_SWAPCHAIN_EXTENSION_NAME,
	};

	auto predicate = [&extensions](const vk::raii::PhysicalDevice &dev) {
		return physical_device_able(dev, extensions);
	};
	
	// Choose a physical device
	// TODO: static lambda (FIRST)
	auto phdev = pick_physical_device(predicate);

	/* auto camera = Camera {
		Transform { {2, 2, 6}, {-0.1, 0.3, 0} },
		Tunings { 45.0f, 800, 800 }
	}; */

	// Create the app and run it
	ECSApp app(phdev, extensions);
	// RTApp app(phdev, extensions);
	// engine::RTCapture app(phdev, {1000, 1000}, extensions, scene_path, camera);

	// Run the app
	app.run();
}

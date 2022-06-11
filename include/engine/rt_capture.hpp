#ifndef KOBRA_ENGINE_RT_CAPTURE_H_
#define KOBRA_ENGINE_RT_CAPTURE_H_

// Engine headers
#include "../app.hpp"
#include "../backend.hpp"
#include "../camera.hpp"
#include "../gui/layer.hpp"
#include "../raytracing/layer.hpp"
#include "../capture.hpp"

namespace kobra {

namespace engine {

// RT capture class
class RTCapture : public BaseApp {
	Camera		camera;

	rt::Layer	layer;
	rt::Batch	batch;
	rt::BatchIndex	index;
	
	bool		term = false;

	// GUI
	gui::Layer	gui_layer;
	gui::Text	*text_progress;

	// Screen dimensions
	vk::Extent2D	dimensions;
public:
	// Constructor from scene file and camera
	RTCapture(const vk::raii::PhysicalDevice &,
			const vk::Extent2D &,
			const std::vector <const char *> &,
			const std::string &,
			const Camera &);

	// Render loop
	void record(const vk::raii::CommandBuffer &, const vk::raii::Framebuffer &);

	// Treminators
	void terminate();
};

}

}

#endif

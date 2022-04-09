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
public:
	// Constructor from scene file and camera
	RTCapture(Vulkan *, const std::string &, const Camera &c);

	// Render loop
	void record(const VkCommandBuffer &, const VkFramebuffer &);

	// Treminators
	void terminate();
};

}

}

#endif

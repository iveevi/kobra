#ifndef CAPTURE_H_
#define CAPTURE_H_

// OpenCV headers
// TODO: switch to ffmpeg
#include <opencv2/core/core.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/videoio.hpp>

// Engine headers
#include "backend.hpp"

namespace mercury {

// Handles image and video capturing of render
namespace capture {

// TODO: pass buffer manager later
// TODO: static captuer function
void write(Vulkan *, const Vulkan::Device &,
	const Vulkan::Buffer &, size_t, size_t,
	const std::string &);

}

// Capture class
class Capture {
	cv::VideoWriter _cap;
public:
	// Constructor
	Capture() {}

	// Starting capture
	void start(const std::string &filename, size_t width, size_t height) {
		_cap.open(filename, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'),
			30, cv::Size(height, width), true);
	}

	// Write frame to file
	void write(Vulkan *vk, const Vulkan::Device &device,
		const Vulkan::Buffer &buffer,
		size_t width, size_t height);
	
	// Flush to file
	void flush() {
		_cap.release();
	}

	// TODO: method to play the video to the renderer
};

}

#endif
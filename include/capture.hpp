#ifndef CAPTURE_H_
#define CAPTURE_H_

// STB image writer
#include <stb/stb_image_write.h>

// Engine headers
#include "backend.hpp"
#include "buffer_manager.hpp"
#include "core.hpp"

namespace mercury {

// Image structure
// TODO: eventually move to another header
struct Image {
	size_t width;
	size_t height;

	// TODO: needs to hold image format

	bytes data;

	// Write image to file
	void write(const std::string &filename) {
		// TODO: later support other formats
		stbi_write_png(
			filename.c_str(),
			width,
			height,
			4,	// TODO: use format later
			data.data(),
			width * 4
		);
	}
};

// Capture class
class Capture {
public:
	// Constructor
	Capture() {}

	// Starting capture
	void start(const std::string &filename, size_t width, size_t height) {
		/* _cap.open(filename, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'),
			30, cv::Size(height, width), true); */
	}

	/* Write frame to file
	void write(Vulkan *vk, const Vulkan::Device &device,
		const Vulkan::Buffer &buffer,
		size_t width, size_t height); */
	
	// Flush to file
	void flush() {
		// _cap.release();
	}

	// TODO: method to play the video to the renderer

	// Get a snapshot
	// TODO: pass image format as well --> bundle in struct
	static void snapshot(const BufferManager <uint> &, Image &);
};

}

#endif
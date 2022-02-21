#include "../include/capture.hpp"

namespace mercury {

namespace capture {

// Convert uint32_t to uint8_ts
std::vector <uint8_t> convert(uint32_t *ptr, size_t size) {
	std::vector <uint8_t> result(size * 4);
	for (size_t i = 0; i < size; i++) {
		result[i * 4 + 2] = (ptr[i] & 0x00FF0000) >> 16;
		result[i * 4 + 1] = (ptr[i] & 0x0000FF00) >> 8;
		result[i * 4] = (ptr[i] & 0x000000FF);
		result[i * 4 + 3] = 255;
	}
	return result;
}

void write(Vulkan *vk, const Vulkan::Device &device,
		const Vulkan::Buffer &buffer,
		size_t width, size_t height,
		const std::string &filename)
{
	// Create OpenCV image
	cv::Mat image(height, width, CV_8UC4);

	// Copy buffer to OpenCV image
	void *bdata = vk->get_buffer_data(device, buffer);
	uint32_t *ptr = static_cast <uint32_t *> (bdata);
	std::vector <uint8_t> data = convert(ptr, width * height);
	std::memcpy(image.data, data.data(), width * height * 4);
	// TODO: dont need width and height, just use buffer size

	// Write OpenCV image to file
	cv::imwrite(filename, image);
}

}

// Capture class
void Capture::write(Vulkan *vk, const Vulkan::Device &device,
	const Vulkan::Buffer &buffer,
	size_t width, size_t height)
{
	// Create OpenCV image
	cv::Mat image(height, width, CV_8UC4);

	// Copy buffer to OpenCV image
	void *bdata = vk->get_buffer_data(device, buffer);
	uint32_t *ptr = static_cast <uint32_t *> (bdata);
	std::vector <uint8_t> data = capture::convert(ptr, width * height);
	std::memcpy(image.data, data.data(), width * height * 4);

	// Write OpenCV image to video capture
	_cap << image;
}

}
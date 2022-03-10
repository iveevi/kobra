#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../include/capture.hpp"

namespace kobra {

// Convert uint32_t to uint8_ts
bytes convert(const uint *ptr, size_t size) {
	std::vector <uint8_t> result(size * 4);
	for (size_t i = 0; i < size; i++) {
		result[i * 4 + 0] = (ptr[i] & 0x00FF0000) >> 16;
		result[i * 4 + 1] = (ptr[i] & 0x0000FF00) >> 8;
		result[i * 4 + 2] = (ptr[i] & 0x000000FF);
		result[i * 4 + 3] = 255;
	}
	return result;
}

// Get a snapshot
void Capture::snapshot(const BufferManager <uint> &pbuf, Image &image)
{
	// Read pixels
	const uint *pixels = pbuf.data();
	bytes data = convert(pixels, pbuf.size());
	image.data = data;
}

}
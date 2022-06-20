#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../include/capture.hpp"

namespace kobra {

namespace capture {

// Convert vec <uint32_t> to vec <uint8_t>
// TODO: more generic image formats?
std::vector <uint8_t> convert(const std::vector <uint32_t> &ptr, size_t size)
{
	std::vector <uint8_t> result(size * 4);
	for (size_t i = 0; i < size; i++) {
		result[i * 4 + 0] = (ptr[i] & 0x000000FF);
		result[i * 4 + 1] = (ptr[i] & 0x0000FF00) >> 8;
		result[i * 4 + 2] = (ptr[i] & 0x00FF0000) >> 16;
		result[i * 4 + 3] = 255;
	}

	return result;
}

void snapshot(const BufferData &buffer, const vk::Extent3D &dim, const std::string &filename)
{
	KOBRA_ASSERT(
		buffer.size == (dim.width * dim.height * dim.depth),
		"Buffer size does not match image dimensions"
	);

	std::vector <uint32_t> data = buffer.download <uint32_t> ();
	std::vector <uint8_t> image = convert(data, dim.width * dim.height);

	stbi_write_png(
		filename.c_str(),
		(size_t) dim.width,
		(size_t) dim.height,
		(size_t) dim.depth,
		image.data(),
		(size_t) dim.width * dim.depth
	);
}

}

}

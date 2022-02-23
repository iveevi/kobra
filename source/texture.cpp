#define STB_IMAGE_IMPLEMENTATION
#include "../include/texture.hpp"

namespace mercury {

bytes load_image_texture(const std::string &filename)
{
	// Check if file exists
	// TODO: function in common.hpp
	std::ifstream file(filename, std::ios::binary);
	if (!file.is_open()) {
		Logger::error("Failed to open file: " + filename);
		return {};
	}

	// Load image
	int width, height, channels;
	byte *image = stbi_load(filename.c_str(), &width, &height, &channels, 0);
	if (!image) {
		Logger::error("Failed to load image: " + filename);
		return {};
	}

	// Create texture
	bytes out(width * height * channels);
	memcpy(out.data(), image, out.size());

	// Free image
	stbi_image_free(image);

	return out;
}

}
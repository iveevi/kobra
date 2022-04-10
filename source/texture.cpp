#define STB_IMAGE_IMPLEMENTATION
#include "../include/texture.hpp"
#include "../include/texture_manager.hpp"

namespace kobra {

Texture load_image_texture(const std::string &filename, int chan)
{
	// Check if file exists
	// TODO: function in common.hpp
	std::ifstream file(filename, std::ios::binary);
	if (!file.is_open()) {
		KOBRA_LOG_FILE(error) << "Failed to open file: \""
			<< filename << "\"" << std::endl;
		return {};
	}

	// Load image
	Profiler::one().frame("Reading texture from source using STB");
	int width, height, channels;
	stbi_set_flip_vertically_on_load(true);
	byte *image = stbi_load(filename.c_str(), &width, &height, &channels, 0);
	if (!image) {
		KOBRA_LOG_FILE(error) << "Failed to load image: \""
			<< filename << "\"" << std::endl;
		Profiler::one().end();
		return {};
	}
	Profiler::one().end();

	// Resize to number of channels if requested
	Profiler::one().frame("Resizing image to requested channels");
	bytes out;
	if (chan > 0 && chan != channels) {
		out = bytes(width * height * chan);

		for (int i = 0; i < width * height; i++) {
			for (int j = 0; j < chan; j++) {
				// If more channels than requested, set zero
				if (j < channels)
					out[i * chan + j] = image[i * channels + j];
				else
					out[i * chan + j] = 0;
			}
		}
	} else {
		// Create texture
		out = bytes(width * height * channels);
		memcpy(out.data(), image, out.size());
	}
	Profiler::one().end();

	// Free image
	stbi_image_free(image);

	return Texture {
		(uint) width,
		(uint) height,
		(uint) channels,
		out
	};
}

}

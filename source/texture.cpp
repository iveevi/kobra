#define STB_IMAGE_IMPLEMENTATION
#include "../include/texture.hpp"

namespace kobra {

Texture load_image_texture(const std::string &filename, int chan)
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
	stbi_set_flip_vertically_on_load(true);  
	byte *image = stbi_load(filename.c_str(), &width, &height, &channels, 0);
	if (!image) {
		Logger::error("Failed to load image: " + filename);
		return {};
	}

	// Resize to number of channels if requested
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

	// Free image
	stbi_image_free(image);

	return Texture {
		(uint) width,
		(uint) height,
		(uint) channels,
		out
	};
}

namespace raytracing {

// Converting bytes to aligned_vec4 array
Buffer bytes_to_floats(const Texture &texture)
{
	// Create buffer
	size_t pixels = texture.data.size() / texture.channels;
	Buffer out(pixels);

	Logger::ok() << "# Channels = " << texture.channels << std::endl;

	// Convert
	size_t c = texture.channels;
	for (size_t i = 0; i < pixels; i++) {
		float r, g, b, a = 1.0f;

		r = (float) texture.data[i * c + 0] / 255.0f;
		if (texture.channels > 1)
			g = (float) texture.data[i * c + 1] / 255.0f;
		if (texture.channels > 2)
			b = (float) texture.data[i * c + 2] / 255.0f;
		if (texture.channels > 3)
			a = (float) texture.data[i * c + 3] / 255.0f;
		
		out[i] = glm::vec4(r, g, b, a);
	}
	// TODO: consider channels

	/* for (size_t i = 0; i < texture.data.size(); i += 4) {
		// Convert each channel to float
		float r, g, b, a;
		r = (float) texture.data[i + 0] / 255.0f;
		g = (float) texture.data[i + 1] / 255.0f;
		b = (float) texture.data[i + 2] / 255.0f;
		a = (float) texture.data[i + 3] / 255.0f;

		Logger::ok() << "texture rgba = " << texture.data[i]
			<< ", " << texture.data[i + 1]
			<< ", " << texture.data[i + 2]
			<< ", " << texture.data[i + 3] << std::endl;

		Logger::ok() << "\tr, g, b, a = " << r << ", "
			<< g << ", " << b << ", " << a << "\n";

		// Store
		size_t pixel = i / 4;
		out[pixel] = glm::vec4 { b, g, r, a };
	} */

	return out;
}

// Reset indices
void TextureUpdate::reset()
{
	textures->reset_push_back();
	texture_info->reset_push_back();
}

// Upload texture data
void TextureUpdate::write(const Texture &texture)
{
	// Get index of texture
	uint index = textures->push_size();

	// Convert bytes to aligned_vec4 array
	const Buffer &buffer = bytes_to_floats(texture);
	textures->push_back(buffer.data(), buffer.size());

	// Update texture info
	texture_info->push_back(glm::uvec4 {
		index,
		texture.width,
		texture.height,
		texture.channels,
	});
}

// Upload texture data to GPU
void TextureUpdate::upload()
{
	// Sync sizes and flush
	textures->sync_size();
	texture_info->sync_size();

	// Upload
	textures->upload();
	texture_info->upload();
}

}

}

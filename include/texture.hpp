#ifndef TEXTURE_H_
#define TEXTURE_H_

// Standard headers
#include <fstream>

// STB image loader
#include <stb/stb_image.h>

// Engine headers
#include "buffer_manager.hpp"
#include "core.hpp"
#include "logger.hpp"

namespace mercury {

// Texture structure
struct Texture {
	// Texture data
	uint width;
	uint height;
	uint channels;

	bytes data;
};

// Load texture as byte array
Texture load_image_texture(const std::string &);

// Textures for ray tracing
namespace raytracing {

// Convert bytes to aligned_vec4 array
Buffer convert_vec4(const Texture &);

// Texture update data
struct TextureUpdate {
	Buffer4f	*textures;
	Buffer4u	*texture_info;

	// Reset indices
	void reset();

	// Write texture data
	void write(const Texture &);

	// Upload texture data
	void upload();
};

}

}

#endif
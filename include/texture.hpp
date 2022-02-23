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

// Load texture as byte array
bytes load_image_texture(const std::string &);

// Textures for ray tracing
namespace raytracing {

// Texture update data
struct TextureUpdate {
	// BufferManager <byte> *_textures;
	Buffer4u *_texture_info;
};

}

}

#endif
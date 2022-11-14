#ifndef KOBRA_TEXTURE_H_
#define KOBRA_TEXTURE_H_

// Standard headers
#include <iostream>
#include <filesystem>

// ImageMagick headers
#include <ImageMagick-7/Magick++.h>

// STB Image headers
#include <stb/stb_image.h>

// Engine headers
#include "core.hpp"

namespace kobra {

// Load an image
byte *load_texture(const std::filesystem::path &, int &, int &, int &);

}

#endif

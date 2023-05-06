#ifndef KOBRA_IMAGE_H_
#define KOBRA_IMAGE_H_

// Standard headers
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <vector>

// ImageMagick headers
// #include <ImageMagick-7/Magick++.h>

// STB headers
#include <stb/stb_image.h>
#include <stb/stb_image_write.h>

// Engine headers
#include "core.hpp"

namespace kobra {

struct RawImage {
	std::vector <uint8_t> data;

	uint32_t width;
	uint32_t height;
	uint32_t channels;

	enum {
		RGBA_8_UI,
		RGBA_32_F
	} type;

        uint32_t size() const {
                return width * height * channels;
        }

	void write(const std::string &filename) {
		stbi_write_png(filename.c_str(),
			width, height, channels,
			data.data(), width * channels
		);
	}
};

// Load an image
RawImage load_texture(const std::filesystem::path &, bool = true);

}

#endif

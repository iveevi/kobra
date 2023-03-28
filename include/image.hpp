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

        uint32_t size() const {
                return width * height * channels;
        }

	void write(const char *filename) {
		stbi_write_png(filename,
			width, height, channels,
			data.data(), width * channels
		);
	}
};

// Load an image
byte *load_texture(const std::filesystem::path &, int &, int &, int &);

}

#endif

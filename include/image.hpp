#ifndef KOBRA_IMAGE_H_
#define KOBRA_IMAGE_H_

// Standard headers
#include <cstdint>
#include <vector>

// STB headers
#include <stb/stb_image_write.h>

namespace kobra {

struct Image {
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

}

#endif
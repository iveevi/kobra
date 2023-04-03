// STBI image implementation
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION

// TinyEXR implementation
#define TINYEXR_IMPLEMENTATION
#define TINYEXR_USE_STB_ZLIB 1

#include <tinyexr/tinyexr.h>

// Engine headers
#include "../include/image.hpp"
#include "../include/logger.hpp"

namespace kobra {

/* Miscellaneous file formats that STB cannot load
static byte *load_texture_magick(const std::filesystem::path &path,
		int &width, int &height, int &channels)
{
	// Load the image
	Magick::Image image;

	try {
		image.read(path.string());
	} catch (Magick::Error &error) {
		KOBRA_LOG_FILE(Log::ERROR) << "Failed to load texture: " << path
			<< " (" << error.what() << ")" << std::endl;

		// Return a 1x1 black texture
		width = 1;
		height = 1;
		channels = 4;

		byte *data = new byte[4] { 0, 0, 0, 255 };

		return data;
	}

	// Get the image dimensions
	width = image.columns();
	height = image.rows();
	channels = image.channels();

	// Print format information
	std::cout << "Loading image with magick: " << path << std::endl;
	std::cout << "Image format: " << image.magick() << std::endl;
	std::cout << "Image dimensions: " << width << "x" << height << std::endl;
	std::cout << "Image channels: " << channels << std::endl;

	// Get the image data
	byte *data = new byte[width * height * 4];
	image.write(0, 0, width, height, "RGBA", Magick::CharPixel, data);

	// Return the image data
	return data;
} */

// Load an image
// TODO: return an optional...
RawImage load_texture(const std::filesystem::path &path)
{
	int width;
	int height;
	int channels;
	
	std::string ext = path.extension().string();

	// Special case extensions
	if (ext == ".exr") {
		// TinyEXR image data
		float *data = nullptr;

		const char *error = nullptr;
		LoadEXR(&data, &width, &height, path.string().c_str(), &error);

		if (error) {
			KOBRA_LOG_FUNC(Log::ERROR) << "Failed to load texture: " << path
				<< " (" << error << ")" << std::endl;
			return RawImage {};
		}

		return RawImage {
			std::vector <uint8_t> {
				reinterpret_cast <uint8_t *> (data),
				reinterpret_cast <uint8_t *> (data)
					+ width * height * 4 * sizeof(float)
			},
			static_cast <uint32_t> (width),
			static_cast <uint32_t> (height),
			4, RawImage::RGBA_32_F
		};
	}

	// Otherwise load with STB
	stbi_set_flip_vertically_on_load(true);
	
	byte *data = stbi_load(
		path.string().c_str(),
		&width, &height, &channels, 4
	);

	// TODO: throw warning, then try to load with ImageMagick
	if (!data) {
		KOBRA_LOG_FUNC(Log::WARN) << "Failed to load texture: " << path
			<< " (" << stbi_failure_reason() << ")" << std::endl;
		return RawImage {};
	}

	printf("Loaded image: %s, %d x %d x %d\n", path.string().c_str(), width, height, channels);
	return RawImage {
		std::vector <uint8_t> (data, data + width * height * 4),
		static_cast <uint32_t> (width),
		static_cast <uint32_t> (height),
		static_cast <uint32_t> (channels),
		RawImage::RGBA_8_UI
	};
}

}

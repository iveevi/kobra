// STBI image implementation
#define STB_IMAGE_IMPLEMENTATION

// Engine headers
#include "../include/image.hpp"
#include "../include/logger.hpp"

namespace kobra {

// Miscellaneous file formats that STB cannot load
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
}

// Load an image
byte *load_texture(const std::filesystem::path &path,
		int &width, int &height, int &channels)
{
	// Special extensions
	std::string ext = path.extension().string();

	std::cout << "Loading texture: " << path << std::endl;
	std::cout << "Extension: " << ext << std::endl;

	if (ext == ".dds")
		return load_texture_magick(path, width, height, channels);

	// Otherwise load with STB
	stbi_set_flip_vertically_on_load(true);
	
	byte *data = stbi_load(
		path.string().c_str(),
		&width, &height, &channels, 4
	);

	// TODO: throw warning, then try to load with ImageMagick
	if (!data)
		assert(false);

	return data;
}

}

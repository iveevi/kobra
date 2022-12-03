#include <cstdint>
#include <vector>

#include "../include/common.hpp"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb/stb_image_write.h>

#include <freetype/freetype.h>

#include FT_FREETYPE_H
#include FT_BBOX_H
#include FT_OUTLINE_H

struct Image {
	std::vector <uint8_t> data;

	uint32_t width;
	uint32_t height;
	uint32_t channels;

	void write(const char *filename) {
		stbi_write_png(filename,
			width, height, channels,
			data.data(), width * channels
		);
	}
};

#define CHECK_FT_ERROR(error) \
	if (error) { \
		std::cerr << "Freetype error: " << error << std::endl; \
		exit(1); \
	}

constexpr char FONT_FILE[] = KOBRA_DIR "/resources/fonts/noto_sans.ttf";
constexpr int FONT_PIXEL_SIZE = 96;

int main()
{
	// Load font
	FT_Library ft;
	FT_Face face;

	CHECK_FT_ERROR(FT_Init_FreeType(&ft));
	CHECK_FT_ERROR(FT_New_Face(ft, FONT_FILE, 0, &face));
	CHECK_FT_ERROR(FT_Set_Pixel_Sizes(face, 0, FONT_PIXEL_SIZE));

	std::cout << "Loaded font: " << face->family_name << std::endl;

	// Check the first characters
	unsigned char c = 'A';

	// Load the glyph
	CHECK_FT_ERROR(FT_Load_Char(face, c, FT_LOAD_RENDER));

	// Get the glyph image data
	Image img;
	img.width = face->glyph->bitmap.width;
	img.height = face->glyph->bitmap.rows;
	img.channels = 1;
	img.data.resize(img.width * img.height);
	memcpy(img.data.data(), face->glyph->bitmap.buffer, img.data.size());

	// Write the image
	img.write("glyph.png");

	// Load the outlines
	CHECK_FT_ERROR(FT_Load_Char(face, c, FT_LOAD_NO_BITMAP));

	// Get the glyph outline
	FT_Outline outline = face->glyph->outline;

	std::cout << "Number of contours: " << outline.n_contours << std::endl;
	std::cout << "Number of points: " << outline.n_points << std::endl;

	for (int i = 0; i < outline.n_contours; i++) {
		std::cout << "Contour " << i << std::endl;

		int start = i == 0 ? 0 : outline.contours[i - 1] + 1;
		int end = outline.contours[i];

		for (int j = start; j <= end; j++) {
			FT_Vector point = outline.points[j];
			char tag = outline.tags[j];
			std::cout << "\tPoint " << j << ": " << point.x/72
				<< ", " << point.y/72 << " (tag[0] = "
				<< (tag & 1) << ", tag[1] = " << ((tag >> 1) & 1)
				<< ")" << std::endl;
		}
	}

	// Generate an SDF image with the outline
	Image sdf;
	sdf.width = img.width;
	sdf.height = img.height;
	sdf.channels = 1;
	sdf.data.resize(sdf.width * sdf.height);
}

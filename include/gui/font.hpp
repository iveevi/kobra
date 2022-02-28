#ifndef FONT_H_
#define FONT_H_

// Standard headers
#include <string>

// FreeType headers
#include <ft2build.h>
#include FT_FREETYPE_H
#include FT_BBOX_H
#include FT_OUTLINE_H

// Engine headers
#include "../common.hpp"
#include "../logger.hpp"

namespace mercury {

namespace gui {

// Font class holds information
//	about a single font
class Font {
	// Check FreeType error
	void check_error(FT_Error error) const {
		if (error) {
			Logger::error() << "FreeType error: "
				<< error << std::endl;
			throw -1;
		}
	}

	// Load FreeType library
	void load_font(const std::string &file) {
		// Load library
		FT_Library library;
		check_error(FT_Init_FreeType(&library));

		// Load font
		FT_Face face;
		check_error(FT_New_Face(library, file.c_str(), 0, &face));

		// Set font size
		check_error(FT_Set_Char_Size(face, 0, 1000 * 64, 96, 96));

		// Process
		size_t total_points = 0;
		size_t total_cells = 0;

		for (size_t i = 0; i < 96; i++) {
			char c = i + ' ';
			std::cout << "Processing character '" << c << "'" << std::endl;
		}
	}
public:
	Font(const std::string &file) {
		// Check that the file exists
		if (!file_exists(file)) {
			Logger::error("Font file not found: " + file);
			throw -1;
		}

		// Load font
		load_font(file);
	}
};

}

}

#endif
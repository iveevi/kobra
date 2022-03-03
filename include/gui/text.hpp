#ifndef TEXT_H_
#define TEXT_H_

// Standard headers
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

// Engine headers
#include "../backend.hpp"
#include "font.hpp"

namespace mercury {

namespace gui {

// Text class
// 	contains glyphs
class Text {
	std::string		_str;
	std::vector <Glyph>	_glyphs;
public:
};

// TextRender class
// 	holds Vulkan structures
// 	and renders for a single font
class TextRender {
	// Reference to glyph (in a text class)
	//	so that updating text is not a pain
	struct Ref {
		int	index;
		Text	*text;

		// For sets
		bool operator==(const Ref &r) const {
			return index == r.index && text == r.text;
		}
	};

	using RefSet = std::set <Ref>;

	// Map of each character to the set of glyphs
	// 	prevents the need to keep rebinding
	// 	the bitmaps
	std::unordered_map <char, RefSet>	_chars;

	// Font to use
	Font					_font;

	// Vulkan structures
	VkGraphicsPipeline			_pipeline;
	VkShaderModule				_vertex;
	VkShaderModule				_fragment;
public:
	// Constructor from paht to font file
	TextRender(const std::string &path) {
	}

};

}

}

#endif

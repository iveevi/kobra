#ifndef CAPTURE_H_
#define CAPTURE_H_

// Standard headers
#include <fstream>

// FFMPEG headers
extern "C" {
	#include <libavutil/imgutils.h>
	#include <libavcodec/avcodec.h>
}

// STB image writer
#include <stb/stb_image_write.h>

// Engine headers
#include "backend.hpp"
#include "core.hpp"

namespace kobra {

namespace capture {

void snapshot(const BufferData &, const vk::Extent3D &, const std::string &);

}

}

#endif

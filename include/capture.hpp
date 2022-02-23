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
#include "buffer_manager.hpp"
#include "core.hpp"

namespace mercury {

// Image structure
// TODO: eventually move to another header
struct Image {
	size_t width;
	size_t height;

	// TODO: needs to hold image format

	bytes data;

	// Write image to file
	void write(const std::string &filename) {
		// TODO: later support other formats
		stbi_write_png(
			filename.c_str(),
			width,
			height,
			4,	// TODO: use format later
			data.data(),
			width * 4
		);
	}
};

// Capture class
class Capture {
public:
	// Capture format
	struct Format {
		size_t bitrate;
		size_t width;
		size_t height;
		size_t framerate;
		size_t gop;
	};
private:
	// FFMPEG contexts and frames
	AVCodecContext	*codec_ctx;
	AVCodec 	*codec;
	AVFrame		*frame;

	// Current file
	std::ofstream	file;

	// Stream buffer
	byte		*buffer		= nullptr;
	size_t		buffer_size	= 0;

	// Capture image
	Image		image;

	// Current frame
	size_t		frame_count	= 0;
public:
	// Constructor
	Capture() {
		// Find codec
		codec = avcodec_find_encoder(AV_CODEC_ID_PNG);
		if (!codec) {
			Logger::error("[Capture] Failed to find codec");
			return;
		}

		// Create codec context
		codec_ctx = avcodec_alloc_context3(codec);
		if (!codec_ctx) {
			Logger::error("[Capture] Failed to allocate codec context");
			return;
		}

		// Allocate frame
		frame = av_frame_alloc();
		if (!frame) {
			Logger::error("[Capture] Failed to allocate frame");
			return;
		}
	}

	// Destructor
	~Capture() {
		// Free frame
		av_frame_free(&frame);

		// Free codec context
		avcodec_free_context(&codec_ctx);

		// Release buffer
		if (buffer)
			delete[] buffer;
	}

	// Starting capture
	void start(const std::string &filename, const Format &fmt) {
		// Set codec parameters
		codec_ctx->bit_rate = fmt.bitrate;
		codec_ctx->width = fmt.width;
		codec_ctx->height = fmt.height;
		codec_ctx->time_base = { 1, (int) fmt.framerate };
		codec_ctx->gop_size = fmt.gop;
		codec_ctx->max_b_frames = 1;
		codec_ctx->pix_fmt = AV_PIX_FMT_RGBA;

		// Open codec
		if (avcodec_open2(codec_ctx, codec, NULL) < 0) {
			Logger::error("[Capture] Failed to open codec");
			return;
		}

		// Create file
		file.open(filename, std::ios::binary);

		// Allocate buffer
		if (buffer)
			delete[] buffer;
		
		buffer_size = fmt.width * fmt.height * 4;
		buffer = new byte[buffer_size];

		// Allocate frame
		av_image_alloc(
			frame->data,
			frame->linesize,
			fmt.width,
			fmt.height,
			AV_PIX_FMT_RGBA,
			1
		);

		frame->width = fmt.width;
		frame->height = fmt.height;
		frame->format = AV_PIX_FMT_RGBA;

		// Set image properties
		image.width = fmt.width;
		image.height = fmt.height;
	}

	// Write frame
	void write(const BufferManager <uint> &pbuf) {
		// Get image data
		snapshot(pbuf, image);

		// Copy each channel to frame
		for (size_t i = 0; i < image.height; i++) {
			for (size_t j = 0; j < image.width; j++) {
				size_t index = i * image.width * 4 + j * 4;
				frame->data[0][index + 0] = image.data[index + 0];
				frame->data[0][index + 1] = image.data[index + 1];
				frame->data[0][index + 2] = image.data[index + 2];
				frame->data[0][index + 3] = image.data[index + 3];
			}
		}

		// Encode frame
		int got_packet = 0;
		AVPacket pkt;
		av_init_packet(&pkt);
		pkt.data = buffer;
		pkt.size = buffer_size;
		avcodec_encode_video2(codec_ctx, &pkt, frame, &got_packet);

		// Write to file
		file.write((char *) buffer, pkt.size);
		frame_count++;
	}

	// Get time in seconds
	double time() const {
		// TODO: this is not correct
		return frame_count / (double) codec_ctx->time_base.den;
	}
	
	// Flush to file
	void flush() {
		// Flush file
		file.flush();
		frame_count = 0;
	}

	// TODO: method to play the video to the renderer

	// Get a snapshot
	// TODO: pass image format as well --> bundle in struct
	static void snapshot(const BufferManager <uint> &, Image &);
};

}

#endif
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

// Write image to file
void snapshot(const BufferData &, const vk::Extent3D &, const std::string &);

// Write images to video
struct Video {
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

	// Current frame
	size_t		frame_count	= 0;
	Format		frame_info;
public:
	// Constructor
	Video() {
		// Find codec
		codec = const_cast <AVCodec *> (avcodec_find_encoder(AV_CODEC_ID_PNG));
		if (!codec) {
			logger("Capture", Log::ERROR) << "Failed to find codec\n";
			return;
		}

		// Create codec context
		codec_ctx = avcodec_alloc_context3(codec);
		if (!codec_ctx) {
			logger("Capture", Log::ERROR) << "Failed to create codec context\n";
			return;
		}

		// Allocate frame
		frame = av_frame_alloc();
		if (!frame) {
			logger("Capture", Log::ERROR) << "Failed to allocate frame\n";
			return;
		}
	}

	// Destructor
	~Video() {
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
			logger("Capture", Log::ERROR) << "Failed to open codec\n";
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

		frame_info = fmt;
	}

	// Write frame
	void write(const std::vector <byte> &data) {
		// Make frame writable
		av_frame_make_writable(frame);

		// Copy each channel to frame
		for (size_t i = 0; i < frame_info.height; i++) {
			for (size_t j = 0; j < frame_info.width; j++) {
				size_t index = i * frame_info.width * 4 + j * 4;
				frame->data[0][index + 0] = data[index + 0];
				frame->data[0][index + 1] = data[index + 1];
				frame->data[0][index + 2] = data[index + 2];
				frame->data[0][index + 3] = data[index + 3];
			}
		}

		// Encode frame
		int ret = avcodec_send_frame(codec_ctx, frame);
		if (ret < 0) {
			logger("Capture", Log::ERROR) << "Failed to encode frame\n";
			return;
		}

		AVPacket *packet = av_packet_alloc();
		while (ret >= 0) {
			// Get packet
			ret = avcodec_receive_packet(codec_ctx, packet);
			if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF)
				break;
			else if (ret < 0) {
				logger("Capture", Log::ERROR) << "Failed to receive packet\n";
				return;
			}

			// Write packet to file
			file.write((char *) packet->data, packet->size);
			av_packet_unref(packet);

			std::cout << "Wrote packet of size " << packet->size << std::endl;
		}

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
};

}

}

#endif

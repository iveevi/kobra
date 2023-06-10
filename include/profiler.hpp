#pragma once

// Standard headers
#include <cmath>
#include <iostream>
#include <mutex>
#include <queue>
#include <stack>
#include <string>
#include <vector>

namespace kobra {

// Forward declarations
struct Profiler;

// Timer event types
enum class EventType {
	eCPU,
	eCUDA
};

// Profiler class
struct Profiler {
	using clk = std::chrono::high_resolution_clock;
	using time_point = clk::time_point;

	// Frame structure (as a tree)
	struct Frame {
		double time = 0.0;
		std::string name = "";
		std::vector <Frame> children = {};

		// Default constructor
		Frame() = default;
	};

	std::queue <Frame> frames;
	std::stack <Frame> stack;

	// Default constructor
	Profiler() = default;

	// Number of recorded frames
	size_t size() const {
		return frames.size();
	}

	Frame *new_frame() {
		stack.push(Frame {});
		return &stack.top();
	}

	// End current frame
	void end() {
		Frame frame = stack.top();
		stack.pop();

		// If frame has no parent, add to queue
		if (stack.empty())
			frames.push(frame);
		else
			stack.top().children.push_back(frame);
	}

	// Return front of queue
	Frame pop() {
		Frame frame = frames.front();
		frames.pop();
		return frame;
	}

	// Pretty print frame
	static std::string pretty(const Frame &frame, double ptime = -1.0f, size_t indent = 0) {
		static std::string indent_str = "  ";

		std::string str;

		// Print indent
		std::string istr;
		for (size_t i = 0; i < indent; i++)
			istr += indent_str;

		// Print name
		str += "[" + frame.name + "] ";

		// Print time
		std::string time_str;

		float time = frame.time;
		std::string units = " us";

		// Milliseconds
		if (time > 1000.0f) {
			float us = time - std::floor(time / 1000.0f) * 1000.0f;
			time = time / 1000.0f;
			time_str = time_str + std::to_string((long int) us) + " us";
			units = " ms";
		}

		// Seconds
		if (time > 1000.0f) {
			float ms = time - std::floor(time / 1000.0f) * 1000.0f;
			time = time / 1000.0f;
			time_str = std::to_string((long int) ms)
				+ " ms, " + time_str;
			units = " s";
		}

		str += std::to_string((long int) time) + units;
		if (!time_str.empty())
			str += ", " + time_str;

		if (ptime >= 0.0f)
			str += " (" + std::to_string(frame.time / ptime * 100.0) + "%)";
		str += "\n";

		// Print children
		for (const Frame &child : frame.children)
			str += istr + "\u2514\u2500 " + pretty(child, frame.time, indent + 1);

		return str;
	}

	// Singleton
	static Profiler &one() {
		static Profiler profiler;
		return profiler;
	}
};

// General scoped event
template <EventType>
struct ScopedEvent;

// Specialization for CPU events
template <>
struct ScopedEvent <EventType::eCPU> {
	Profiler::Frame *frame = nullptr;
	Profiler::time_point start;

	// Constructor
	ScopedEvent(const std::string &name) {
		start = Profiler::clk::now();
		// Profiler::one().frame(name);
		frame = Profiler::one().new_frame();
		frame->name = name;
	}

	// Destructor
	~ScopedEvent() {
		Profiler::time_point end = Profiler::clk::now();
		double time = std::chrono::duration_cast
			<std::chrono::microseconds> (end - start).count();
		frame->time = time;
		Profiler::one().end();
	}
};

#ifdef __CUDACC__

// Specialization for CUDA events
template <>
struct ScopedEvent <EventType::eCUDA> {
	cudaEvent_t start;
	cudaEvent_t end;

	Profiler::Frame *frame = nullptr;

	// Constructor
	ScopedEvent(const std::string &name) {
		cudaEventCreate(&start);
		cudaEventCreate(&end);
		cudaEventRecord(start);

		frame = Profiler::one().new_frame();
		frame->name = name;
	}

	// Destructor
	~ScopedEvent() {
		cudaEventRecord(end);
		cudaEventSynchronize(end);
		float time;
		cudaEventElapsedTime(&time, start, end);
		cudaEventDestroy(start);
		cudaEventDestroy(end);

		frame->time = time * 1000.0f;
		Profiler::one().end();
	}
};

#endif

}

#define KOBRA_PROFILING

// Profiling macros
#ifdef KOBRA_PROFILING

#define CONCAT(a, b) CONCAT_INNER(a, b)
#define CONCAT_INNER(a, b) a ## b

#define KOBRA_PROFILE_TASK(name)				\
	auto CONCAT(sf_, CONCAT(__LINE__, __COUNTER__))		\
	= kobra::ScopedEvent <kobra::EventType::eCPU> (name)

#define KOBRA_PROFILE_CUDA_TASK(name)				\
	auto CONCAT(sf_, CONCAT(__LINE__, __COUNTER__))		\
	= kobra::ScopedEvent <kobra::EventType::eCUDA> (name);

#define KOBRA_PROFILE_PRINT() \
	while (kobra::Profiler::one().size()) { \
		kobra::Profiler::Frame frame = kobra::Profiler::one().pop(); \
		std::cout << kobra::Profiler::pretty(frame); \
	}

#define KOBRA_PROFILE_RESET() \
	while (kobra::Profiler::one().size()) \
		kobra::Profiler::one().pop();

#else

#warning "Profiling disabled"

#define KOBRA_PROFILE_TASK(name)
#define KOBRA_PROFILE_CUDA_TASK(name)
#define KOBRA_PROFILE_PRINT()
#define KOBRA_PROFILE_RESET()

#endif

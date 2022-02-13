#ifndef PROFILER_H_
#define PROFILER_H_

// Standard headers
#include <queue>
#include <stack>
#include <string>
#include <vector>

// TODO: remove
#include <iostream>

// Engine headers
#include "timer.hpp"

namespace mercury {

// Profiler class
class Profiler {
public:
	// Frame structure (as a tree)
	struct Frame {
		std::string		name;
		Timer::time_point	start;
		double			time;
		std::vector <Frame>	children;

		// Constructor
		Frame(const std::string &n, const Timer::time_point &s)
			: name(n), start(s), time(0.0) {}
	};
private:
	// Frame queue
	std::queue <Frame>	_frames;

	// Timer
	Timer			_timer;

	// Frame stack
	std::stack <Frame>	_stack;

	Timer::time_point	_start;
public:
	// Constructor
	Profiler() {
		_start = _timer.now();
	}

	// Create a new frame
	void frame(const std::string &name) {
		Frame frame {name, _timer.now()};

		// Add frame to stack
		_stack.push(frame);
	}

	// End current frame
	void end() {
		// Get current frame
		Frame frame = _stack.top();

		// Get time
		frame.time = _timer.elapsed(frame.start);

		// Pop frame from stack
		_stack.pop();

		// If frame has no parent, add to queue
		if (_stack.empty())
			_frames.push(frame);
		else
			_stack.top().children.push_back(frame);
	}

	// Return front of queue
	Frame pop() {
		Frame frame = _frames.front();
		_frames.pop();

		return frame;
	}

	// Pretty print frame
	static std::string pretty(const Frame &frame, size_t indent = 0) {
		static std::string indent_str = "  ";

		std::string str;

		// Print indent
		std::string istr;
		for (size_t i = 0; i < indent; i++)
			istr += indent_str;

		// Print name
		str += "[" + frame.name + "]: ";

		// Print time
		str += std::to_string(frame.time) + " us\n";

		// Print children
		for (const Frame &child : frame.children)
			str += istr + "\u2514\u2500 " + pretty(child, indent + 1);

		return str;
	}
};

}

#endif

#ifndef TIMER_H_
#define TIMER_H_

// Standard headers
#include <chrono>

namespace kobra {

// Timer structure
class Timer {
public:
	using clk = std::chrono::high_resolution_clock;
	using time_point = clk::time_point;
private:
	time_point _start;
	time_point _end;

	clk _clock;
public:
	// Constructor
	Timer() : _start(_clock.now()), _end(_clock.now()) {}

	// Start timer
	void start() {
		_start = _clock.now();
	}

	// Stop timer
	void stop() {
		_end = _clock.now();
	}

	// Current timepoint
	time_point now() {
		return _clock.now();
	}

	// Get elapsed time (in milliseconds)
	double elapsed() {
		return std::chrono::duration_cast
			<std::chrono::microseconds>
			(_end - _start).count();
	}

	double elapsed_start() {
		return std::chrono::duration_cast
			<std::chrono::microseconds>
			(now() - _start).count();
	}

	double elapsed(time_point t) {
		return std::chrono::duration_cast
			<std::chrono::microseconds>
			(now() - t).count();
	}

	// Elapsed time and reset timer
	double lap() {
		double elapsed = this->elapsed(_start);
		this->start();
		return elapsed;
	}
};

}

#endif

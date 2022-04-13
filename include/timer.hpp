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
	Timer();

	// Methods
	void start();
	void stop();
	
	time_point now();

	double elapsed();
	double elapsed_start();
	double elapsed(time_point t);
	double lap();
};

}

#endif

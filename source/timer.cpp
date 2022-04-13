#include "../include/timer.hpp"

namespace kobra {

// Constructor
Timer::Timer() : _start(_clock.now()), _end(_clock.now()) {}

// Start timer
void Timer::start()
{
	_start = _clock.now();
}

// Stop timer
void Timer::stop()
{
	_end = _clock.now();
}

// Current timepoint
Timer::time_point Timer::now()
{
	return _clock.now();
}

// Get elapsed time (in milliseconds)
double Timer::elapsed()
{
	return std::chrono::duration_cast
		<std::chrono::microseconds>
		(_end - _start).count();
}

double Timer::elapsed_start()
{
	return std::chrono::duration_cast
		<std::chrono::microseconds>
		(now() - _start).count();
}

double Timer::elapsed(time_point t)
{
	return std::chrono::duration_cast
		<std::chrono::microseconds>
		(now() - t).count();
}

// Elapsed time and reset timer
double Timer::lap()
{
	double elapsed = this->elapsed(_start);
	this->start();
	return elapsed;
}

}

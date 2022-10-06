#ifndef KOBRA_CORE_INTERPOLATION_H_
#define KOBRA_CORE_INTERPOLATION_H_

// Standard headers
#include <vector>
#include <algorithm>

namespace kobra {

namespace core {

template <class T>
struct Sequence {
	std::vector <T> values;
	std::vector <float> times;
};

template <class T>
T piecewise_linear(const Sequence <T> &seq, float t)
{
	// Find the first time greater than t
	auto it = std::upper_bound(seq.times.begin(), seq.times.end(), t);

	// If t is greater than the last time, return the last value
	if (it == seq.times.end())
		return seq.values.back();

	// If t is less than the first time, return the first value
	if (it == seq.times.begin())
		return seq.values.front();

	// Find the index of the first time greater than t
	auto index = it - seq.times.begin();

	// Find the time and value before t
	auto t0 = seq.times[index - 1];
	auto v0 = seq.values[index - 1];

	// Find the time and value after t
	auto t1 = seq.times[index];
	auto v1 = seq.values[index];

	// Interpolate between the two values
	return v0 + (v1 - v0) * (t - t0) / (t1 - t0);
}

}

}

#endif

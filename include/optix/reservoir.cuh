#ifndef KOBRA_OPTIX_RESERVOIR_H_
#define KOBRA_OPTIX_RESERVOIR_H_

// Engine headers
#include "../cuda/math.cuh"
#include "../cuda/random.cuh"

namespace kobra {

namespace optix {

// Reservoir sampling
template <class Sample>
struct Reservoir {
	// Single sample reservoir
	Sample sample = Sample();

	// Cumulative weight
	float weight;
	float3 random;
	int count;

	// Constructor
	__forceinline__ __host__ __device__
	Reservoir() : weight(0.0f), count(0) {}

	// Reset
	__forceinline__ __device__
	void reset() {
		weight = 0.0f;
		count = 0;
	}

	// Update the reservoir
	__forceinline__ __device__
	void update(const Sample &sample, const float weight) {
		// Update the cumulative weight
		this->weight += weight;
		this->count++;

		// Randomly select the sample
		if (fract(random3(random).x) < weight / (this->weight + 1e-6f)) {
			this->sample = sample;
			// this->weight = weight;
		}
	}

	// Merge two reservoirs
	__forceinline__ __device__
	void merge(const Reservoir &reservoir) {
		float weight = reservoir.weight * reservoir.count;
		update(reservoir.sample, weight);
	}
};

}

}

#endif

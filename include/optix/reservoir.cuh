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
	float	W;
	float	weight;
	float3	random;
	int	count;
	int	max_count;

	// Constructor
	KCUDA_INLINE KCUDA_HOST_DEVICE
	Reservoir(int max) : W(0.0f), weight(0.0f),
			count(0), max_count(max) {}

	// Reset
	KCUDA_INLINE __device__
	void reset() {
		sample = Sample();
		weight = 0.0f;
		count = 0;
	}

	// Update the reservoir
	KCUDA_INLINE __device__
	void update(const Sample &sample, const float weight) {
		static const float eps = 1e-4f;

		// Update the cumulative weight
		this->weight += weight;
		this->count = min(this->count + 1, max_count);

		// Randomly select the sample
		float r = fract(random3(random).x);
		float w = weight/(this->weight);
	
		bool selected = (r < w + eps);
		if (selected || this->count == 1)
			this->sample = sample;
	}

	// Merge two reservoirs
	KCUDA_INLINE __device__
	void merge(const Reservoir &reservoir, float target) {
		int current = count;
		update(reservoir.sample, target * reservoir.weight * reservoir.count);
		count = min(current + reservoir.count, max_count);
	}
};

}

}

#endif

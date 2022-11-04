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
			count(0), max_count(max) {
		// Generate random number
		float x = *reinterpret_cast <float *> (this);
		random = make_float3(x, 1 - fract(x), fract(x));
	}

	// Reset
	KCUDA_INLINE __device__
	void reset() {
		sample = Sample();
		weight = 0.0f;
		count = 0;
	}

	// Update the reservoir
	KCUDA_INLINE __device__
	bool update(const Sample &sample, const float weight) {
		static const float eps = 1e-4f;

		// Update the cumulative weight
		this->weight += weight;
		this->count = min(this->count + 1, max_count);
		// this->count++;

		// Randomly select the sample
		float r = cuda::rand_uniform(random);
		float w = weight/(this->weight);
	
		bool selected = (r < w + eps);
		if (selected || this->count == 1)
			this->sample = sample;

		return selected;
	}

	// Merge two reservoirs
	KCUDA_INLINE __device__
	void merge(const Reservoir &reservoir, float target) {
		int current = count;
		update(reservoir.sample, target * reservoir.weight * reservoir.count);
		count = min(current + reservoir.count, max_count);
	}
};

// Reservoir with multiple samples
template <class Sample, unsigned int N>
struct MultiReservoir {
	static constexpr unsigned int size = N;

	Sample samples[N] = { Sample() };

	// Cumulative weight
	float	W;
	float	weight;
	float3	random;
	int	count;

	// Constructor
	KCUDA_INLINE KCUDA_HOST_DEVICE
	MultiReservoir(int) : W(0.0f), weight(0.0f), count(0) {
		// Generate random number
		float x = *reinterpret_cast <float *> (this);
		random = make_float3(x, 1 - fract(x), fract(x));
	}

	// Reset
	KCUDA_INLINE __device__
	void reset() {
		for (int i = 0; i < N; i++)
			samples[i] = Sample();
		weight = 0.0f;
		count = 0;
	}

	// Update the reservoir
	KCUDA_INLINE __device__
	bool update(const Sample &sample, const float weight) {
		static const float eps = 1e-4f;

		// Update the cumulative weight
		this->weight += weight;
		this->count++;

		// Randomly select the sample
		bool selected = false;
		for (int i = 0; i < size; i++) {
			float r = cuda::rand_uniform(random);
			float w = weight/(this->weight);
		
			bool s = (r < w + eps);
			if (s || this->count == 1)
				samples[i] = sample;

			selected |= s;
		}

		return selected;
	}

	// Merge two reservoirs
	KCUDA_INLINE __device__
	void merge(const MultiReservoir &reservoir, float target) {
		float wsum = weight + reservoir.weight;

		float r = cuda::rand_uniform(random);
		if (r < reservoir.weight/wsum) {
			for (int i = 0; i < N; i++)
				samples[i] = reservoir.samples[i];
		}

		weight = wsum;
	}
};

}

}

#endif

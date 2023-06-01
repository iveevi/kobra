#pragma once

// Engine headers
#include "../cuda/math.cuh"
#include "../cuda/random.cuh"

namespace kobra {

namespace optix {

// using ReSTIR_Reservoir = Reservoir <PathSample>;
template <class T>
struct WeightedReservoir {
	T sample;

	unsigned int count;

	float weight;
	float mis;
};


// Updating a reservoir
template <class T>
__device__ __forceinline__
bool reservoir_update(WeightedReservoir <T> *reservoir,
		const T &sample, float weight,
		cuda::Seed seed)
{
	reservoir->weight += weight;

	float eta = cuda::rand_uniform(seed);
	bool selected = (eta * reservoir->weight) < weight;

	if (selected)
		reservoir->sample = sample;

	reservoir->count++;

	return selected;
}

}

}
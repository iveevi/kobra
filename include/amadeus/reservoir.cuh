#ifndef KOBRA_AMADEUS_RESERVOIR_H_
#define KOBRA_AMADEUS_RESERVOIR_H_

// Standard headers
#include <random>

// GLM headers
#include <glm/glm.hpp>

// Engine headers
#include "../cuda/core.cuh"
#include "../cuda/random.cuh"

namespace kobra {

namespace amadeus {

// Reservoir sample and structure
template <class T>
struct Reservoir {
	T data;

	float w;
	float W;
	int M;

	float3 seed;

	Reservoir() : data(), w(0.0f), W(0.0f), M(0) {
		static std::random_device rd;
		static std::mt19937 gen(rd());
		static std::uniform_real_distribution <float> dis(0.0f, 1.0f);

		// Initialize the seed
		seed = {dis(gen), dis(gen), dis(gen)};
	}

	KCUDA_INLINE KCUDA_DEVICE
	Reservoir(const glm::vec4 &_seed)
			: data(), w(0.0f), W(0.0f), M(0),
			seed {_seed.x, _seed.y, _seed.z} {}

	KCUDA_INLINE KCUDA_DEVICE
	bool update(const T &sample, float weight) {
		w += weight;

		float eta = cuda::rand_uniform(seed);
		bool selected = (eta * w) < weight;

		if (selected)
			data = sample;

		M++;

		return selected;
	}

	KCUDA_INLINE KCUDA_DEVICE
	void resample(float target) {
		float d = target * M;
		W = (d > 0) ? w/d : 0.0f;
	}

	KCUDA_INLINE KCUDA_DEVICE
	void reset() {
		w = 0.0f;
		M = 0;
		W = 0.0f;
	}

	KCUDA_INLINE KCUDA_DEVICE
	int size() const {
		return M;
	}
};

// TODO: refactor to direct sample
struct DirectLightingSample {
	float3 Le;
	float3 normal;
	float3 point;
	int type;
};

}

}

#endif

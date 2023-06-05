#pragma once

// Engine headers
#include "include/cuda/random.cuh"
#include "include/cuda/brdf.cuh"

// Local headers
#include "editor/path_tracer.cuh"

#define SPARSITY_STRIDE 4

// Utility methods
struct Sky {
        cudaTextureObject_t texture;
        bool enabled;
};

__forceinline__ __device__
float4 sky_at(Sky sky, float3 direction)
{
        if (!sky.enabled)
                return make_float4(0.0f);

        float theta = acosf(direction.y);
        float phi = atan2f(direction.z, direction.x);

        float u = phi/(2.0f * PI);
        float v = theta/PI;

        return tex2D <float4> (sky.texture, u, 1 - v);
}

__forceinline__ __device__
void convert_material(const cuda::_material &src, cuda::Material &dst, float2 uv)
{
        dst.diffuse = src.diffuse;
        dst.specular = src.specular;
        dst.emission = src.emission;
        dst.roughness = src.roughness;
        dst.refraction = src.refraction;
        dst.type = src.type;

        if (src.textures.has_diffuse) {
                float4 diffuse = tex2D <float4> (src.textures.diffuse, uv.x, uv.y);
                dst.diffuse = make_float3(diffuse);
        }
}

// Reservoir sample and structure
// TODO: struct of arrays instead
// TODO: remove for sparse gi...
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
		seed = { dis(gen), dis(gen), dis(gen) };
	}

        __forceinline__ __device__
        Reservoir(const float3 &_seed)
                        : data(), w(0.0f), W(0.0f), M(0),
                        seed { _seed.x, _seed.y, _seed.z } {}

	__forceinline__ __device__
	bool update(const T &sample, float weight) {
		w += weight;

		float eta = cuda::rand_uniform(seed);
		bool selected = (eta * w) < weight;

		if (selected)
			data = sample;

		M++;

		return selected;
	}

	__forceinline__ __device__
	void resample(float target) {
		float d = target * M;
		W = (d > 0) ? w/d : 0.0f;
	}

	__forceinline__ __device__
	void reset() {
		w = 0.0f;
		M = 0;
		W = 0.0f;
	}

	__forceinline__ __device__
	int size() const {
		return M;
	}
};

// Orthogonal camera axis
struct CameraAxis {
        float3 U;
        float3 V;
        float3 W;
        float3 origin;
        uint2 resolution;
};


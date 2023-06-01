#pragma once

// Standard headers
#include <random>

// OptiX headers
#include <optix.h>

// Engine headers
#include "include/vertex.hpp"
#include "include/cuda/material.cuh"
#include "include/cuda/random.cuh"

#define SPARSITY_SITRDE 4

// Local headers
#include "../path_tracer.cuh"

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

	__forceinline__ __device__
	Reservoir(const glm::vec4 &_seed)
			: data(), w(0.0f), W(0.0f), M(0),
			seed {_seed.x, _seed.y, _seed.z} {}

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

// TODO: refactor to direct sample
struct DirectLightingSample {
	float3 Le;
	float3 normal;
	float3 point;
	int type;
};

// Orthogonal camera axis
struct CameraAxis {
        float3 U;
        float3 V;
        float3 W;
        float3 origin;
        uint2 resolution;
};

// Launch info for the G-buffer raytracer
struct SparseGIParameters {
        // Acceleration structure
        OptixTraversableHandle handle;

        // Global parameters
        float time;
        bool reset;
        bool dirty;
        int samples;

        // Previous camera matrices
        glm::mat4 previous_view;
        glm::mat4 previous_projection;
        float3 previous_origin;
        float4 *previous_position;

        // Camera parameters
        CameraAxis camera;

        // G-buffer source surfaces
        cudaSurfaceObject_t position_surface;
        cudaSurfaceObject_t normal_surface;
        cudaSurfaceObject_t uv_surface;
        cudaSurfaceObject_t index_surface;

        // List of all materials
        kobra::cuda::_material *materials;

        // Lighting information
        struct {
                AreaLight *lights;
                uint count;
                uint triangle_count;
        } area;

        Sky sky;

        // Direct lighting (ReSTIR)
        Reservoir <DirectLightingSample> *direct_lighting;

        // Indirect lighting caches
        struct {
                float4 *screen_irradiance; // (R, G, B, # samples)
                float4 *final_irradiance;

                float4 *irradiance_directions;
                float *direction_samples;

                // Screen space irradiance probes
                // (R, G, B, depth) arranged in Bw x Bh grid
                // (each probe in NxN grid layed out with octahedral projection)
                float4 *screen_probes;
                int N;
        } indirect;

        // Additional options during rendering
        struct {
                bool direct;
                bool indirect;
        } options;
        
        // IO interface
        OptixIO io;
};

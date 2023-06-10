#pragma once

// Engine headers
#include "include/cuda/math.cuh"
#include "include/cuda/error.cuh"

// TODO: probe.cuh
// Irradiance probe
struct IrradianceProbe {
        constexpr static int size = 6;

        // Layed out using octahedral projection
        float3 values[size * size];
        float pdfs[size * size];
        float depth[size * size];

        float3 normal;
	float3 position;

	constexpr static float sqrt_2 = 1.41421356237f;
	constexpr static float inv_sqrt_2 = 1.0f/sqrt_2;

	// NOTE: Direction is expected to be in local space

	// Octahedral projection
	__forceinline__ __device__
	float2 to_oct(float3 v) const {
		float3 r = v / (abs(v.x) + abs(v.y) + abs(v.z));
		// float2 s = make_float2(r.x + r.y, r.x - r.y);
		float2 s = make_float2(r);
		return (s + 1.0f) / 2.0f;
	}

	// Disk projection
	__forceinline__ __device__
	float2 to_disk(float3 v) const {
		float2 s = make_float2(v.x, v.y);
		float theta = atan2f(s.y, s.x);
		if (theta < 0.0f)
			theta += 2.0f * M_PI;
		float rp = pow(s.x * s.x + s.y * s.y, 0.3f);
		return make_float2(theta/(2.0f * M_PI), rp);
	}
};

// A single level-node of the irradiance probe LUT
struct IrradianceProbeLUT {
       constexpr static int MAX_REFS = (1 << 15) - 1;

	// Cell properties
        float resolution;
	float size;
	float3 center;

        int32_t level = 0;
	int32_t counter = 0;

	int32_t refs[MAX_REFS];
	uint32_t hashes[MAX_REFS];
	float3 positions[MAX_REFS];

	static void alloc(IrradianceProbeLUT *lut) {
		lut->level = 0;
		lut->size = 10.0f;
		lut->resolution = lut->size/25.0f;
		lut->center = make_float3(0.0f, 0.0f, 0.0f);

		for (int i = 0; i < MAX_REFS; i++) {
			lut->refs[i] = -1;
			lut->hashes[i] = 0xFFFFFFFF;
		}
	}

	__forceinline__ __device__
	static void alloc(IrradianceProbeLUT *lut, int level, float size, float3 center) {
		lut->level = level;
		lut->size = size;
		lut->resolution = size/powf(MAX_REFS, 1.0f/3.0f);
		lut->center = center;
	}

	__forceinline__ __device__
	bool contains(float3 x, bool neighbor = false) {
		float3 min = center - make_float3(size/2.0f + neighbor * resolution);
		float3 max = center + make_float3(size/2.0f + neighbor * resolution);
		return (x.x >= min.x && x.y >= min.y && x.z >= min.z)
			&& (x.x <= max.x && x.y <= max.y && x.z <= max.z);
	}

        __forceinline__ __device__
        uint32_t hash(float3 x, int32_t dx = 0, int32_t dy = 0, int32_t dz = 0) {
		float cdx = x.x - center.x + size/2.0f;
		float cdy = x.y - center.y + size/2.0f;
		float cdz = x.z - center.z + size/2.0f;

		int32_t ix = (int32_t) ((cdx / resolution) + dx);
		int32_t iy = (int32_t) ((cdy / resolution) + dy);
		int32_t iz = (int32_t) ((cdz / resolution) + dz);

		int32_t h = (ix & 0x7FF) | ((iy & 0x7FF) << 11) | ((iz & 0x7FF) << 22);

		// Shuffle bits
		h ^= (h >> 11) ^ (h >> 22);
		h ^= (h << 7) & 0x9D2C5680;
		h ^= (h << 15) & 0xEFC60000;

		return *((uint32_t *) &h);
        }

	// TODO: radius (e.g. multiplication factor)
	__forceinline__ __device__
	uint32_t neighboring_hash(float3 x, int8_t ni) {
		// Options are ni from 0 to 26, e.g. sides of a cube
		int32_t dx = ni % 3 - 1;
		int32_t dy = (ni / 3) % 3 - 1;
		int32_t dz = ni / 9 - 1;

		return hash(x, dx, dy, dz);
	}

        // TODO: try double hashing instead?
        // NOTE: linear probing would be more cache friendly, but double hashing
        // leads to better distribution
        static constexpr int32_t MAX_TRIES = (1 << 2);

	// TODO: analyze how well it is able to cover surfaces with higher number of tries..
	__forceinline__ __device__
	uint32_t double_hash(uint32_t h, int32_t i) {
		// Double hashing (shuffle again)
		// int32_t oh = h;
		//
		// h = (h ^ 61) ^ (h >> 16);
		// h = h + (h << 3);
		// h = h ^ (h >> 4);
		// h = h * 0x27d4eb2d;
		// h = h ^ (h >> 15);
		//
		// return oh + (i + 1) * h;
		return h + (i * i);
	}

	// Allocating new reference
	// TODO: cuckoo hashing; if there is an infinite loop, then kick out the oldest (LRU)
	__forceinline__ __device__
	int32_t request(float3 x) {
		// TODO: return index if already allocated

		uint32_t h = hash(x);
		int32_t success = -1;

		int32_t i = 0;
		int32_t old = INT_MAX;

		while (i < MAX_TRIES) {
			int32_t j = double_hash(h, i) % MAX_REFS;
			old = atomicCAS(&refs[j], -1, h);
			if (old == -1) {
				success = j;
				break;
			}

			i++;
		}

		if (old == -1) {
			hashes[success] = h;
			positions[success] = x;
			atomicAdd(&counter, 1);
		}

		return success;
	}

	// Find the nearest reference to the given position
	// TODO: return the distance to the nearest reference
	__forceinline__ __device__
	int32_t lookup(float3 x, int32_t ni) {
		if (!contains(x, ni != -1))
			return -1;

		uint32_t h = (ni == -1) ? hash(x) : neighboring_hash(x, ni);

		int32_t i = 0;
		int32_t index = -1;

		float closest = FLT_MAX;
		while (i < MAX_TRIES) {
			int32_t j = double_hash(h, i) % MAX_REFS;
			int32_t ref = refs[j];
			if (ref == -1)
				break;

			if (hashes[j] == h) {
				float dist = length(positions[j] - x);
				if (dist < closest) {
					closest = dist;
					index = j;
				}
			}

			i++;
		}

		return index;
	}
};

// Table of all irradiance probes and LUTs
struct IrradianceProbeTable {
        constexpr static int MAX_PROBES = 1 << 20;
	constexpr static int MAX_LUTS = 1 << 8;

        IrradianceProbe *probes = nullptr;
        int32_t counter = 0;

	IrradianceProbeLUT *luts = nullptr;
	int32_t lut_counter = 0;

	IrradianceProbeTable() {
		probes = new IrradianceProbe[MAX_PROBES];

		// Allocate top level LUT upon front
		luts = new IrradianceProbeLUT[MAX_LUTS];
		IrradianceProbeLUT::alloc(&luts[0]);
		lut_counter++;
	}

	IrradianceProbeTable device_copy() const {
		IrradianceProbeTable table;

		CUDA_CHECK(cudaMalloc(&table.probes, sizeof(IrradianceProbe) * MAX_PROBES));
		CUDA_CHECK(cudaMemcpy(table.probes, probes, sizeof(IrradianceProbe) * MAX_PROBES, cudaMemcpyHostToDevice));
		table.counter = counter;

		CUDA_CHECK(cudaMalloc(&table.luts, sizeof(IrradianceProbeLUT) * MAX_LUTS));
		CUDA_CHECK(cudaMemcpy(table.luts, luts, sizeof(IrradianceProbeLUT) * MAX_LUTS, cudaMemcpyHostToDevice));
		table.lut_counter = lut_counter;

		return table;
	}

	__forceinline__ __device__
	int32_t next() {
		int32_t index = atomicAdd(&counter, 1);
		if (index >= MAX_PROBES)
			return -1;
		return index;
	}

	__forceinline__ __device__
	int32_t lut_next() {
		int32_t index = atomicAdd(&lut_counter, 1);
		if (index >= MAX_LUTS)
			return -1;
		return index;
	}

	__forceinline__ __device__
	void clear() {
		counter = 0;
		lut_counter = 0;
	}
};

inline IrradianceProbeTable *alloc_table()
{
	IrradianceProbeTable table;
	IrradianceProbeTable proxy_table = table.device_copy();
	IrradianceProbeTable *device_table;

	CUDA_CHECK(cudaMalloc(&device_table, sizeof(IrradianceProbeTable)));
	CUDA_CHECK(cudaMemcpy(device_table, &proxy_table, sizeof(IrradianceProbeTable), cudaMemcpyHostToDevice));

	return device_table;
}

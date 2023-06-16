#pragma once

// Engine headers
#include "include/cuda/math.cuh"
#include "include/cuda/error.cuh"
#include "include/cuda/alloc.cuh"

// TODO: probe.cuh
// Irradiance probe
struct IrradianceProbe {
        constexpr static int size = 5;

        float3 Le[size * size];
        float pdfs[size * size];
        float depth[size * size];

        float3 normal;
	float3 position;

	constexpr static float sqrt_2 = 1.41421356237f;
	constexpr static float inv_sqrt_2 = 1.0f/sqrt_2;

	// NOTE: Direction is expected to be in local space

	// Octahedral projection
	// __forceinline__ __device__
	// float2 to_oct(float3 v) const {
	// 	float3 r = v / (abs(v.x) + abs(v.y) + abs(v.z));
	// 	// float2 s = make_float2(r.x + r.y, r.x - r.y);
	// 	float2 s = make_float2(r);
	// 	return (s + 1.0f) / 2.0f;
	// }

	// Disk projection
	__forceinline__ __device__
	float2 to_disk(float3 v) const {
		float2 s = make_float2(v.x, v.y);
		float theta = atan2f(s.y, s.x);
		if (theta < 0.0f)
			theta += 2.0f * M_PI;
		// float rp = pow(s.x * s.x + s.y * s.y, 0.3f);
		float rp = sqrt(s.x * s.x + s.y * s.y);
		return make_float2(theta/(2.0f * M_PI), rp);
	}

	__forceinline__ __device__
	int32_t to_index(float2 uv) {
		int32_t x = uv.x * size;
		int32_t y = uv.y * size;
		// values[0] = make_float3(x, y, 0.0f);
		return clamp(y * size + x, 0, size * size - 1);
	}

	// Reconstruct local direction from index
	__forceinline__ __device__
	float3 from_index(int32_t index) {
		int32_t x = index % size;
		int32_t y = index / size;
		float2 uv = clamp(make_float2(x, y) / size, 0.0f, 1.0f);
		return from_uv(uv);
	}

	// Reconstruct local direction from uv
	__forceinline__ __device__
	float3 from_uv(float2 uv) {
		// TODO: center middle of cell
		float theta = 2 * uv.x * M_PI;
		float x = uv.y * cos(theta);
		float y = uv.y * sin(theta);
		float z = sqrt(1.0f - x * x - y * y);
		return make_float3(x, y, z);
	}

	// TODO: method to convert to spherical harmonics
};

// A single level-node of the irradiance probe LUT
static constexpr int32_t MAX_REFS = (1 << 15) - 1;

struct IrradianceProbeLUT {

	// Cell properties
        float resolution;
	float size;
	float3 center;

        int32_t level = 0;
	int32_t counter = 0;
	bool initialized = false;

	// int32_t refs[MAX_REFS];
	// uint32_t hashes[MAX_REFS];
	// float3 positions[MAX_REFS];
	// int32_t signal_ready[MAX_REFS];

	int32_t *refs = nullptr;
	uint32_t *hashes = nullptr;
	float3 *positions = nullptr;
	int32_t *signal_ready = nullptr;

	static constexpr float RESOLUTION_DIVISION = 25.0f;
	static constexpr float L1 = 10.0f;
	static constexpr float L2 = L1 * RESOLUTION_DIVISION;

	// TODO: leave these methods outside...
	static void alloc(IrradianceProbeLUT *lut) {
		lut->level = 2;
		lut->size = L2;
		lut->resolution = L2/RESOLUTION_DIVISION;
		lut->center = make_float3(0.0f, 0.0f, 0.0f);
		// for (int i = 0; i < MAX_REFS; i++) {
		// 	lut->refs[i] = -1;
		// 	lut->hashes[i] = 0xFFFFFFFF;
		// 	lut->signal_ready[i] = 0;
		// }
		lut->initialized = true;
	}

	__forceinline__ __device__
	static void alloc(IrradianceProbeLUT *lut, int level, float size, float3 center) {
		lut->level = level;
		lut->size = size;
		lut->resolution = size/RESOLUTION_DIVISION;
		lut->center = center;

		int32_t minus_counter = 0;
		for (int i = 0; i < MAX_REFS; i++) {
			// lut->refs[i] = -1;
			// lut->hashes[i] = 0xFFFFFFFF;
			// lut->signal_ready[i] = 0;

			minus_counter += (lut->refs[i] == -1);
		}

		// TODO: memset initialize in the host...

		printf("Verifying alloc: %d\n", minus_counter);
		lut->initialized = true;
	}

	__forceinline__ __device__
	bool contains(float3 x, bool neighbor = false) {
		float3 min = center - make_float3(size/2.0f + neighbor * resolution);
		float3 max = center + make_float3(size/2.0f + neighbor * resolution);
		return (x.x >= min.x && x.y >= min.y && x.z >= min.z)
			&& (x.x <= max.x && x.y <= max.y && x.z <= max.z);
	}

	__forceinline__ __device__
	float3 local_center(float3 x) const {
		// Find index, then give the center of the corresponding cell
		float cdx = x.x - center.x + size/2.0f;
		float cdy = x.y - center.y + size/2.0f;
		float cdz = x.z - center.z + size/2.0f;

		int32_t ix = (int32_t) (cdx / resolution);
		int32_t iy = (int32_t) (cdy / resolution);
		int32_t iz = (int32_t) (cdz / resolution);

		float cx = center.x + (ix + 0.5f) * resolution - size/2.0f;
		float cy = center.y + (iy + 0.5f) * resolution - size/2.0f;
		float cz = center.z + (iz + 0.5f) * resolution - size/2.0f;

		return make_float3(cx, cy, cz);
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
        // static constexpr int32_t MAX_TRIES = (1 << 2);
        static constexpr int32_t MAX_TRIES = (1 << 1);

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
	// __forceinline__ __device__
	// int32_t request(float3 x) {
	// 	// TODO: return index if already allocated
	//
	// 	uint32_t h = hash(x);
	// 	int32_t success = -1;
	//
	// 	int32_t i = 0;
	// 	int32_t old = INT_MAX;
	//
	// 	while (i < MAX_TRIES) {
	// 		int32_t j = double_hash(h, i) % MAX_REFS;
	// 		old = atomicCAS(&refs[j], -1, h);
	// 		if (old == -1) {
	// 			success = j;
	// 			break;
	// 		}
	//
	// 		i++;
	// 	}
	//
	// 	if (old == -1) {
	// 		hashes[success] = h;
	// 		// TODO: store center of the cell instead of the position
	// 		positions[success] = x;
	// 		atomicAdd(&counter, 1);
	// 	}
	//
	// 	return success;
	// }

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

	__forceinline__ __device__
	int32_t full_lookup(float3 x) {
		// Check all 27 neighboring cells as well
		int32_t result = lookup(x, -1);
		if (result == -1) {
			for (int32_t ni = 0; ni < 27; ni++) {
				result = lookup(x, ni);
				if (result != -1)
					break;
			}
		}

		return result;
	}

	// Get the closest (N = 4) probes
	// NOTE: Indices are assumed to be null before calling this function (e.g. -1)
	template<int32_t N = 4>
	__forceinline__ __device__
	void closest(float3 x, int32_t indices[N], float distances[N]) {
		// Make sure x is inside the grid (or near the surface)
		if (!contains(x, true))
			return;

		// First is the closest, etc.
		// float closest[N] = { FLT_MAX };

		for (int32_t i = 0; i < MAX_TRIES; i++) {
			int32_t j = double_hash(hash(x), i) % MAX_REFS;
			int32_t ref = refs[j];
			if (ref == -1)
				break;

			float dist = length(positions[j] - x);

			// Insertion sort
			for (int32_t k = 0; k < N; k++) {
				if (dist < distances[k]) {
					for (int32_t l = N - 1; l > k; l--) {
						distances[l] = distances[l - 1];
						indices[l] = indices[l - 1];
					}

					distances[k] = dist;
					indices[k] = j;
					break;
				}
			}
		}

		// Also check all 27 neighboring cells
		for (int32_t ni = 0; ni < 27; ni++) {
			for (int32_t i = 0; i < MAX_TRIES; i++) {
				int32_t j = double_hash(neighboring_hash(x, ni), i) % MAX_REFS;
				int32_t ref = refs[j];
				if (ref == -1)
					break;

				float dist = length(positions[j] - x);

				// Insertion sort
				for (int32_t k = 0; k < N; k++) {
					if (dist < distances[k]) {
						for (int32_t l = N - 1; l > k; l--) {
							distances[l] = distances[l - 1];
							indices[l] = indices[l - 1];
						}

						distances[k] = dist;
						indices[k] = j;
						break;
					}
				}
			}
		}
	}
};

// Table of all irradiance probes and LUTs
struct IrradianceProbeTable {
        static constexpr int MAX_PROBES = 1 << 20;
	static constexpr int MAX_LUTS = 1 << 10;

        IrradianceProbe *probes = nullptr;
        int32_t counter = 0;

	IrradianceProbeLUT *luts = nullptr;
	int32_t lut_counter = 0;

        // float3 *values = nullptr;
        // float *pdfs = nullptr;
        // float *depth = nullptr;

	// Linearlized array of LUT data
	static constexpr int32_t LUT_DATA_SIZE = MAX_REFS * MAX_LUTS;

	float3 *positions;
	int32_t *refs;
	int32_t *signal_ready;
	uint32_t *hashes;

	static constexpr int32_t STRIDE = IrradianceProbe::size * IrradianceProbe::size;

	IrradianceProbeTable() {
		// TODO: make obselete...
		probes = new IrradianceProbe[MAX_PROBES];

		// Allocate LUT data
		positions = new float3[LUT_DATA_SIZE];
		refs = new int32_t[LUT_DATA_SIZE];
		signal_ready = new int32_t[LUT_DATA_SIZE];
		hashes = new uint32_t[LUT_DATA_SIZE];

		std::fill(refs, refs + LUT_DATA_SIZE, -1);
		std::fill(signal_ready, signal_ready + LUT_DATA_SIZE, 0);
		std::fill(hashes, hashes + LUT_DATA_SIZE, 0xFFFFFFFF);

		// Allocate top level LUT upon front
		luts = new IrradianceProbeLUT[MAX_LUTS];
		for (int i = 0; i < MAX_LUTS; i++) {
			luts[i].refs = refs + i * MAX_REFS;
			luts[i].positions = positions + i * MAX_REFS;
			luts[i].signal_ready = signal_ready + i * MAX_REFS;
			luts[i].hashes = hashes + i * MAX_REFS;
		}

		// IrradianceProbeLUT::alloc(&luts[0]);
		lut_counter++;

		// Allocate memory for all probes
		int32_t size = MAX_PROBES * IrradianceProbe::size * IrradianceProbe::size;
		// values = new float3[size];
		// pdfs = new float[size];
		// depth = new float[size];

		printf("Size of IrradianceProbeTable: %d\n", sizeof(IrradianceProbeTable));
		printf("Size of probes: %d/%d\n", sizeof(IrradianceProbe) * MAX_PROBES, sizeof(IrradianceProbe));
		printf("Size of luts: %d/%d\n", sizeof(IrradianceProbeLUT) * MAX_LUTS, sizeof(IrradianceProbeLUT));

		for (int i = 0; i < 10; i++) {
			int32_t *addr = luts[i].refs;
			int32_t offset = addr - (int32_t *) refs;
			int32_t re_index = offset/MAX_REFS;
			printf("LUT%d := Offset: %d, re_index: %d (%d)\n", i, offset, re_index, sizeof(int32_t) * MAX_REFS);
		}
	}

	IrradianceProbeTable device_copy() const {
		IrradianceProbeTable table;

		CUDA_CHECK(cudaMalloc(&table.probes, sizeof(IrradianceProbe) * MAX_PROBES));
		CUDA_CHECK(cudaMemcpy(table.probes, probes, sizeof(IrradianceProbe) * MAX_PROBES, cudaMemcpyHostToDevice));
		table.counter = counter;

		// Copy LUT data
		CUDA_CHECK(cudaMalloc(&table.positions, sizeof(float3) * LUT_DATA_SIZE));
		CUDA_CHECK(cudaMemcpy(table.positions, positions, sizeof(float3) * LUT_DATA_SIZE, cudaMemcpyHostToDevice));

		CUDA_CHECK(cudaMalloc(&table.refs, sizeof(int32_t) * LUT_DATA_SIZE));
		CUDA_CHECK(cudaMemcpy(table.refs, refs, sizeof(int32_t) * LUT_DATA_SIZE, cudaMemcpyHostToDevice));

		CUDA_CHECK(cudaMalloc(&table.signal_ready, sizeof(int32_t) * LUT_DATA_SIZE));
		CUDA_CHECK(cudaMemcpy(table.signal_ready, signal_ready, sizeof(int32_t) * LUT_DATA_SIZE, cudaMemcpyHostToDevice));

		CUDA_CHECK(cudaMalloc(&table.hashes, sizeof(uint32_t) * LUT_DATA_SIZE));
		CUDA_CHECK(cudaMemcpy(table.hashes, hashes, sizeof(uint32_t) * LUT_DATA_SIZE, cudaMemcpyHostToDevice));

		printf("Allocated probes at %p\n", table.probes);
		printf("Allocated LUT positions at %p\n", table.positions);
		printf("Allocated LUT refs at %p\n", table.refs);
		printf("Allocated LUT signal_ready at %p\n", table.signal_ready);
		printf("Allocated LUT hashes at %p\n", table.hashes);

		// Allocate memory for all LUTs

		// Proxy structures
		std::vector <IrradianceProbeLUT> dev_luts { MAX_LUTS };
		for (int i = 0; i < MAX_LUTS; i++) {
			dev_luts[i].refs = table.refs + i * MAX_REFS;
			dev_luts[i].signal_ready = table.signal_ready + i * MAX_REFS;
			dev_luts[i].positions = table.positions + i * MAX_REFS;
			dev_luts[i].hashes = table.hashes + i * MAX_REFS;
		}

		// Allocate top level up front
		IrradianceProbeLUT::alloc(&dev_luts[0]);

		CUDA_CHECK(cudaMalloc(&table.luts, sizeof(IrradianceProbeLUT) * MAX_LUTS));
		CUDA_CHECK(cudaMemcpy(table.luts, dev_luts.data(), sizeof(IrradianceProbeLUT) * MAX_LUTS, cudaMemcpyHostToDevice));

		// CUDA_CHECK(cudaMalloc(&table.luts, sizeof(IrradianceProbeLUT) * MAX_LUTS));
		// CUDA_CHECK(cudaMemcpy(table.luts, luts, sizeof(IrradianceProbeLUT) * MAX_LUTS, cudaMemcpyHostToDevice));
		// table.luts = kobra::cuda::make_buffer(dev_luts);

		table.lut_counter = 1;

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

// Table lookup methods
__forceinline__ __device__
int32_t request_L1(IrradianceProbeLUT *lut, float3 x)
{
	// TODO: asser that lut level is 1
        constexpr int32_t MAX_TRIES = (1 << 1);

	uint32_t h = lut->hash(x);
	int32_t success = -1;

	int32_t i = 0;
	int32_t old = INT_MAX;

	// TODO: methods
	while (i < MAX_TRIES) {
		int32_t j = lut->double_hash(h, i) % MAX_REFS;
		old = atomicCAS(&lut->refs[j], -1, h);
		if (old == -1) {
			success = j;
			break;
		}

		i++;
	}

	if (old == -1) {
		lut->hashes[success] = h;
		lut->positions[success] = x;
		atomicAdd(&lut->counter, 1);
	}

	return success;
}

struct LookupL2 {
	int32_t index;
	int32_t success;
};

__forceinline__ __device__
LookupL2 request_L2(IrradianceProbeLUT *lut, float3 x)
{
	// TODO: asser that lut level >= 2
	// NOTE: For second level, we linearly search for an empty slot (dense)
	uint32_t h = lut->hash(x);
	LookupL2 result = { -1, -1 };

	// Prune if not contained
	if (!lut->contains(x))
		return result;

	int32_t i = 0;
	int32_t old = INT_MAX;

	// TODO: precompute neighbor information (hash and offset position)

	// TODO: methods
	while (i < MAX_REFS) {
		int32_t j = (h + i) % MAX_REFS;
		old = atomicCAS(&lut->refs[j], -1, h);

		// If we found an empty slot, we are done
		if (old == -1) {
			printf("Found empty slot at %d\n", j);
			result.success = j;
			result.index = j;

			// Compute center of the corresponding cell
			// TODO: method
			float cdx = x.x - lut->center.x + lut->size/2.0f;
			float cdy = x.y - lut->center.y + lut->size/2.0f;
			float cdz = x.z - lut->center.z + lut->size/2.0f;

			int32_t ix = cdx / lut->resolution;
			int32_t iy = cdy / lut->resolution;
			int32_t iz = cdz / lut->resolution;

			float3 center = make_float3(
				lut->center.x + (ix + 0.5f) * lut->resolution - lut->size/2.0f,
				lut->center.y + (iy + 0.5f) * lut->resolution - lut->size/2.0f,
				lut->center.z + (iz + 0.5f) * lut->resolution - lut->size/2.0f
			);

			lut->hashes[j] = h;
			lut->positions[j] = center;
			atomicAdd(&lut->counter, 1);
			atomicExch(&lut->signal_ready[j], 1);
			printf("Set ready signal at %d\n", j);
			break;
		}

		// Warp level synchronization
		__syncwarp();

		// Otherwise, check if the position belongs to the current cell
		// TODO: needs to wait if the currently hashed cell is being updated...

		// Wait until the ready signal is ON for the current cell
		// int32_t counter = 0;
		// while (lut->signal_ready[j] == 0) {
		// 	// printf("Waiting for ready signal at %d\n", j);
		// 	// if (counter++ % 1000 == 0)
		// 	// 	printf("Waiting for ready signal at %d\n", j);
		// 	uint64_t t = clock64() + 0;
		// }

		// Kernel level synchronization
		volatile int32_t *ready = &lut->signal_ready[j];
		while (*ready == 0) {}

		float3 center = lut->positions[j];

		float3 min = center - make_float3(0.5f * lut->resolution);
		float3 max = center + make_float3(0.5f * lut->resolution);

		bool bounded_max = (x.x <= max.x) && (x.y <= max.y) && (x.z <= max.z);
		bool bounded_min = (x.x >= min.x) && (x.y >= min.y) && (x.z >= min.z);

		if (bounded_max && bounded_min) {
			result.index = j;
			break;
		}

		// Move on
		i++;
	}

	// if (old == -1) {
	// 	// Compute center of the corresponding cell
	// 	float cdx = x.x - lut->center.x + lut->size/2.0f;
	// 	float cdy = x.y - lut->center.y + lut->size/2.0f;
	// 	float cdz = x.z - lut->center.z + lut->size/2.0f;
	//
	// 	int32_t ix = cdx / lut->resolution;
	// 	int32_t iy = cdy / lut->resolution;
	// 	int32_t iz = cdz / lut->resolution;
	//
	// 	float3 center = make_float3(
	// 		lut->center.x + (ix + 0.5f) * lut->resolution - lut->size/2.0f,
	// 		lut->center.y + (iy + 0.5f) * lut->resolution - lut->size/2.0f,
	// 		lut->center.z + (iz + 0.5f) * lut->resolution - lut->size/2.0f
	// 	);
	//
	// 	lut->hashes[result.success] = h;
	// 	lut->positions[result.success] = center;
	// 	atomicAdd(&lut->counter, 1);
	// 	atomicExch(&lut->signal_ready[result.success], 1);
	// 	printf("Set ready signal at %d\n", result.success);
	// }

	// printf("Returning %d, %d\n", result.success, result.index);

	// TODO: record the maximum number of iterations and then double hashing...

	return result;
}

__forceinline__ __device__
int32_t lookup_L2(IrradianceProbeLUT *lut, float3 x)
{
	if (!lut->contains(x))
		return -1;

	uint32_t h = lut->hash(x);

	int32_t i = 0;
	int32_t index = -1;

	while (i < MAX_REFS) {
		int32_t j = (h + i) % MAX_REFS;

		// If the cell is empty, return -1
		if (lut->refs[j] == -1)
			break;

		// Otherwise, check if the position belongs to the current cell
		float3 center = lut->positions[j];

		float3 min = center - make_float3(0.5f * lut->resolution);
		float3 max = center + make_float3(0.5f * lut->resolution);

		bool bounded_max = (x.x <= max.x) && (x.y <= max.y) && (x.z <= max.z);
		bool bounded_min = (x.x >= min.x) && (x.y >= min.y) && (x.z >= min.z);

		if (bounded_max && bounded_min) {
			index = j;
			break;
		}

		// Move on
		i++;
	}

	return index;
}

struct ProbeAllocationInfo {
        cudaSurfaceObject_t index_surface;
        cudaSurfaceObject_t position_surface;
	cudaSurfaceObject_t normal_surface;
        float *sobel;
	float *raster;

	IrradianceProbeTable *table;

        vk::Extent2D extent;
};

#include "include/backend.hpp"
#include "optix/probe.cuh"
#include <cassert>

__global__
void probe_allocation(ProbeAllocationInfo info)
{
        // TODO: use block parallelization
        // i.e. sync threads in block, then use atomicAdd

        int index = threadIdx.x + blockIdx.x * blockDim.x;

        // Each thread goes through 16x16 block to allocate probes
        // int x = (index % info.extent.width) / 16;
        // int y = (index / info.extent.width) / 16;
        int x = index % (info.extent.width/16);
        int y = index / (info.extent.width/16);

        // if (x >= (info.extent.width / 16 + 1) || y >= (info.extent.height / 16 + 1))
        //         return;
        if (x >= (info.extent.width/16) || y >= (info.extent.height/16))
                return;

        int x2 = x * 16;
        int y2 = y * 16;

	int32_t best_index = -1;
	int32_t misses = 0;

	int32_t best_distance = 0;

	int32_t x_best = x2 + 8;
	int32_t y_best = y2 + 8;

        // TODO: shared memory: project existing probes onto 16x16 block
        // TODO: need to check if block has invalid regions
	// TODO: floodfill parallelized...
	for (int i = 0; i < 16; i++) {
		for (int j = 0; j < 16; j++) {
			int x3 = x2 + i;
			int y3 = y2 + j;

			if (x3 < 0 || x3 >= info.extent.width || y3 < 0 || y3 >= info.extent.height)
				continue;

			int32_t n_index = y3 * info.extent.width + x3;
			if (info.raster[n_index] != 0)
				continue;

			// Search for the closest
			// int N = 16;
			int N = 4;

			int32_t distance = 0;
			for (int k = -N; k <= N; k++) {
				for (int l = -N; l <= N; l++) {
					int x4 = x3 + k;
					int y4 = y3 + l;

					if (x4 < 0 || x4 >= info.extent.width || y4 < 0 || y4 >= info.extent.height)
						continue;

					int32_t n_index2 = y4 * info.extent.width + x4;
					if (info.raster[n_index2] == 0)
						continue;

					distance = max(abs(k) + abs(l), distance);
				}
			}

			if (distance > best_distance) {
				best_distance = distance;
				best_index = n_index;
				x_best = x3;
				y_best = y3;
			}

			misses++;
		}
	}

	if (misses < 0.9 * 16 * 16)
		return;

	// If all are misses, then use middle
	// TODO: use a special heuristic for allocating in the first frame ish
	if (misses == 16 * 16) {
		x_best = x2 + 8;
		y_best = y2 + 8;
	}

        // For now lazily allocate at the central position of the block
        int x3 = x_best;
        int y3 = y_best;

	if (x3 < 0 || x3 >= info.extent.width || y3 < 0 || y3 >= info.extent.height)
		return;

        y3 = info.extent.height - (y3 + 1);

        int32_t raw_index;
        surf2Dread(&raw_index, info.index_surface, x3 * sizeof(int32_t), y3);
        if (raw_index == -1)
                return;

        float4 raw_position;
        surf2Dread(&raw_position, info.position_surface, x3 * sizeof(float4), y3);
        float3 position = make_float3(raw_position);

	float4 raw_normal;
	surf2Dread(&raw_normal, info.normal_surface, x3 * sizeof(float4), y3);
	float3 normal = make_float3(raw_normal);

        float radius = 0.01f;

	// Top level (TODO: eventually do multilevel...)
	IrradianceProbeLUT *lut = &info.table->luts[0];
	if (!lut->contains(position))
		return;

	// uint32_t hash = lut->hash(position);
	LookupL2 result = request_L2(lut, position);
	if (result.index == -1)
		return;

	// Allocate lower level LUT if needed
	if (result.success != -1) {
		// Grab next LUT
		int32_t lut_index = info.table->lut_next();
		if (lut_index == -1)
			return;

		assert(lut_index >= 0 && lut_index < info.table->lut_counter);

		IrradianceProbeLUT *nlut = &info.table->luts[lut_index];
		float3 center = lut->local_center(position);
		float3 recorded_center = lut->positions[result.index];
		IrradianceProbeLUT::alloc(nlut, 1, IrradianceProbeLUT::L1, center);
		printf("Allocated new LUT @%d (index = %d), center at <%f, %f, %f> vs recorded center <%f, %f, %f>\n",
			lut_index, result.index, center.x, center.y, center.z, recorded_center.x, recorded_center.y, recorded_center.z);
		lut->refs[result.index] = lut_index;
	}

	// Wait for the LUT to be allocated
	assert(info.table->lut_counter <= IrradianceProbeTable::MAX_LUTS);
	int32_t current_index;
	int32_t iterations = 0;
	do {
		// TODO: atomic read?
		current_index = lut->refs[result.index];
		if (iterations++ > 10)
			printf("Waiting for LUT %d (currently %d) to be allocated\n", result.index, current_index);
	} while (current_index < 0 || current_index >= info.table->lut_counter);

	// TODO: fix this bug, this is a fail case safety check
	if (!(current_index >= 0 && current_index < IrradianceProbeTable::MAX_LUTS)) {
		// printf("[!] BAD current_index = %d, counter is %d\n", current_index, info.table->lut_counter);
		return;
	}

	assert(current_index >= 0 && current_index < IrradianceProbeTable::MAX_LUTS);
	IrradianceProbeLUT *nlut = &info.table->luts[current_index];

	// Allocate probe if possible
	int32_t success = request_L1(nlut, position);
	if (success == -1)
		return;

	// Allocate the probe
	int32_t probe_index = info.table->next();
	if (probe_index == -1)
		return;

	IrradianceProbe *probe = &info.table->probes[probe_index];
	nlut->refs[success] = probe_index;

	*probe = {};
	probe->position = position;
	probe->normal = normal;

	printf("Allocated probe @%d\n", probe_index);
}

#ifndef KOBRA_OPTIX_GRID_BASED_RESERVOIRS_PARAMETERS_H_
#define KOBRA_OPTIX_GRID_BASED_RESERVOIRS_PARAMETERS_H_

#include "../optix/lighting.cuh"
#include "../optix/reservoir.cuh"

namespace kobra {

namespace optix {

constexpr float GBR_SIZE = 50.0f;

constexpr int GBR_CELL_LIMIT = 100;
constexpr int GBR_RESERVOIR_COUNT = 1;
constexpr int GRID_RESOLUTION = 100;
constexpr int TOTAL_CELLS = GRID_RESOLUTION * GRID_RESOLUTION * GRID_RESOLUTION;
constexpr int TOTAL_RESERVOIRS = GBR_RESERVOIR_COUNT * TOTAL_CELLS;

__host__ __device__
constexpr uint3 GBR_RESOLUTION = {
	GRID_RESOLUTION,
	GRID_RESOLUTION,
	GRID_RESOLUTION
};

struct GRBSample {
	float3 source; // position
	float3 value;
	float3 point;
	float3 normal;
	float target;
	int type; // 0 - quad, 1 - triangle, 2 - envmap
	int index;

	// TODO: constructor...
};

using Reservoir = WeightedReservoir <GRBSample>;

struct GridBasedReservoirsParameters {
	// Image resolution
	uint2 resolution;

	// Camera position
	// TODO: struct
	float3 camera;

	float3 cam_u;
	float3 cam_v;
	float3 cam_w;

	// Time
	// TODO: settings struct
	float time;

	// Accumulation status
	bool accumulate;
	int samples;

	// Scene information
	OptixTraversableHandle traversable;

	int instances;

	// Textures
	cudaTextureObject_t envmap;
	bool has_envmap;

	// Lights
	struct {
		optix::QuadLight *quads;
		optix::TriangleLight *triangles;

		uint quad_count;
		uint triangle_count;
	} lights;

	// Output buffers (color + AOV)
	float4 *color_buffer;
	float4 *normal_buffer;
	float4 *albedo_buffer;
	float4 *position_buffer;

	// Grid-based reservoirs
	struct {
		Reservoir *new_samples;
		Reservoir *light_reservoirs;
		Reservoir *light_reservoirs_old;
		int *sample_indices;
		int *cell_sizes;
		uint3 resolution;
		bool reproject;
	} gb_ris;
};

}

}

#endif
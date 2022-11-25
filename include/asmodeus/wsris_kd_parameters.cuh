#include "../optix/parameters.cuh"

namespace kobra {

namespace optix {

struct WorldSpaceKdReservoirsParameters {
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

	// World space kd-tree reservoirs
	optix::WorldNode *kd_tree;
	optix::LightReservoir *kd_reservoirs;
	optix::LightReservoir *kd_reservoirs_prev;
	int **kd_locks;
	int kd_nodes;
	int kd_leaves;
};

}

}

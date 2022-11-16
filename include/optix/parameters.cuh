#ifndef KOBRA_OPTIX_PARAMETERS_H_
#define KOBRA_OPTIX_PARAMETERS_H_

// Engine headers
#include "../bbox.hpp"
#include "../core/kd.cuh"
#include "../cuda/material.cuh"
#include "../cuda/math.cuh"
#include "../cuda/random.cuh"
#include "lighting.cuh"
#include "reservoir.cuh"

namespace kobra {

namespace optix {
	
// Constants
enum : unsigned int {
	eRegular = 0,
	eReSTIR,
	eVoxel,
	eCount
};

// Reservoir samples for Resampling techniques
struct LightSample {
	float3 value;
	float3 point;
	float3 normal;
	float target;
	int type; // 0 - quad, 1 - triangle
	int index;
};

struct PathSample {
	float3 value;
	float3 position;
	float3 source;
	float3 normal;
	Shading shading;
	float3 direction;
	float pdf;
	float target;
	bool missed;
};

struct VoxelSample {
	float3 value;
	float3 position;
	float3 direction;
};

struct TMRIS_Sample {
	float3 value;
	float3 point;
	float3 normal;
	float target;
};

// using ReSTIR_Reservoir = Reservoir <PathSample>;
template <class T>
struct WeightedReservoir {
	T sample;

	unsigned int count;

	float weight;
	float mis;
};

using LightReservoir = WeightedReservoir <LightSample>;
using ReSTIR_Reservoir = WeightedReservoir <PathSample>;
using Voxel_Reservoir = MultiReservoir <VoxelSample, 10>;
using TMRIS_Reservoir = WeightedReservoir <TMRIS_Sample>;
using WorldNode = core::KdNode <int>;

// Hit data record
struct Hit {
	// Mesh data
	float2			*texcoords;
	float3			*vertices;
	uint3			*triangles;

	float3			*normals;
	float3			*tangents;
	float3			*bitangents;

	// Auto UV mapping parameters
	BoundingBox		bbox;

	// Material and textures
	cuda::Material		material;

	struct {
		cudaTextureObject_t	diffuse;
		cudaTextureObject_t	normal;
		cudaTextureObject_t	roughness;

		bool			has_diffuse = false;
		bool			has_normal = false;
		bool			has_roughness = false;
	} textures;
	
	// Texture mapped reservoir sampling
	static constexpr int TMRIS_RESOLUTION = 32;

	struct {
		TMRIS_Reservoir	*f_res; //facing forward
		TMRIS_Reservoir	*b_res; //facing backward

		TMRIS_Reservoir	*f_res_prev;
		TMRIS_Reservoir	*b_res_prev;

		int		**f_locks;
		int		**b_locks;

		int		resolution;
	} tmris;
};

// Kernel-common parameters for hybrid tracer
struct HT_Parameters {
	// Image resolution
	uint2 resolution;

	// Camera position
	float3 camera;

	float3 cam_u;
	float3 cam_v;
	float3 cam_w;

	// Time
	float time;

	// Accumulation status
	bool accumulate;
	int samples;

	// Scene information
	OptixTraversableHandle traversable;

	int instances;

	// G-buffer textures
	cudaTextureObject_t positions;
	cudaTextureObject_t normals;
	cudaTextureObject_t ids;

	cudaTextureObject_t albedo;
	cudaTextureObject_t specular;
	cudaTextureObject_t extra;

	cudaTextureObject_t envmap;

	// Lights
	struct {
		QuadLight *quads;
		TriangleLight *triangles;

		uint quad_count;
		uint triangle_count;
	} lights;

	// Reservoirs and advanced sampling strategies
	struct {
		ReSTIR_Reservoir *r_temporal;
		ReSTIR_Reservoir *r_temporal_prev;
		
		ReSTIR_Reservoir *r_spatial;
		ReSTIR_Reservoir *r_spatial_prev;
	} advanced;

	// Output buffers
	float4 *color_buffer;
};

// Kernel-common parameters for Basilisk path tracer
struct BasiliskOptions {
	bool disable_accumulation;
	bool indirect_only;
	bool recursive_wsris;
	bool wsris_visualize;
	bool reprojected_reuse;
};

// Enable hashing with World Space RIS
// #define WSRIS_HASH_RESOLUION 10

struct BasiliskParameters {
	// Mode, indicates various flags...
	// TODO: create an abstractoin for integrators
	uint mode; // TODO: move to options

	// Additional options
	BasiliskOptions options;

	// Image resolution
	uint2 resolution;

	// Camera position
	float3 camera;

	float3 cam_u;
	float3 cam_v;
	float3 cam_w;

	// Time
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
		QuadLight *quads;
		TriangleLight *triangles;

		uint quad_count;
		uint triangle_count;
	} lights;

	// Reservoirs and advanced sampling strategies
	struct {
		// ReSTIR
		LightReservoir *r_lights;
		LightReservoir *r_lights_prev;

		LightReservoir *r_spatial;
		LightReservoir *r_spatial_prev;
	} advanced;

	// Voxel reservoirs sampling
	struct {
		Voxel_Reservoir *reservoirs;

		int **locks;
		int resolution;

		float3 min;
		float3 max;
	} voxel;

	// Output buffers (color + AOV)
	float4 *color_buffer;
	float4 *normal_buffer;
	float4 *albedo_buffer;
	
	float4 *position_buffer;

	WorldNode *kd_tree;
	LightReservoir *kd_reservoirs;
	LightReservoir *kd_reservoirs_prev;
	int **kd_locks;
	int kd_nodes;
	int kd_leaves;
};

}

}

#endif

// OptiX headers
#include <optix.h>

// Engine headers
#include "../../include/cuda/brdf.cuh"
#include "../../include/cuda/material.cuh"
#include "../../include/cuda/math.cuh"
#include "../../include/optix/core.cuh"
#include "../../include/optix/parameters.cuh"
#include "../../include/optix/lighting.cuh"
#include "../../include/cuda/matrix.cuh"

using namespace kobra::cuda;
using namespace kobra::optix;

extern "C"
{
	__constant__ kobra::optix::WadjetParameters parameters;
}

// Local constants
static const float eps = 1e-3f;

// Check shadow visibility
KCUDA_INLINE __device__
bool is_occluded(float3 origin, float3 dir, float R)
{
	static float eps = 0.05f;

	bool vis = true;

	unsigned int j0, j1;
	pack_pointer <bool> (&vis, j0, j1);

	optixTrace(parameters.traversable,
		origin, dir,
		0, R - eps, 0,
		OptixVisibilityMask(0b1),
		OPTIX_RAY_FLAG_DISABLE_CLOSESTHIT
			| OPTIX_RAY_FLAG_DISABLE_ANYHIT
			| OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT,
		parameters.instances, 0, 1,
		j0, j1
	);

	return vis;
}

// Trace ray into scene and get relevant information
__device__ float3 Ld(float3 x, float3 wo, float3 n,
		Material mat, bool entering, float3 &seed)
{
	int quad_count = parameters.lights.quad_count;
	int tri_count = parameters.lights.triangle_count;

// #define GROUND_TRUTH

#ifdef GROUND_TRUTH
	
	float3 contr = {0.0f};

	for (int i = 0; i < quad_count; i++) {
		QuadLight light = parameters.lights.quads[i];
		contr += Ld_light(light, x, wo, n, mat, entering, seed);
	}

	for (int i = 0; i < tri_count; i++) {
		TriangleLight light = parameters.lights.triangles[i];
		contr += Ld_light(light, x, wo, n, mat, entering, seed);
	}

	return contr;

#else

	if (quad_count == 0 && tri_count == 0)
		return make_float3(0.0f);

	int total_count = quad_count + tri_count;

	random3(seed);
	unsigned int i = fract(seed.x) * (quad_count + tri_count);
	i = min(i, quad_count + tri_count - 1);

	if (i < quad_count) {
		QuadLight light = parameters.lights.quads[i];
		return total_count * Ld_light(light, x, wo, n, mat, entering, seed);
	}

	TriangleLight light = parameters.lights.triangles[i - quad_count];
	return total_count * Ld_light(light, x, wo, n, mat, entering, seed);

#endif

}

// Ray packet data
struct RayPacket {
	float3	value;
	float3	position;
	float3	normal;

	float	ior;
	
	int	depth;
	uint	index;
	
	float3	seed;
};

static KCUDA_INLINE KCUDA_HOST_DEVICE
void make_ray(uint3 idx,
		 float3 &origin,
		 float3 &direction,
		 float3 &seed)
{
	const float3 U = parameters.cam_u;
	const float3 V = parameters.cam_v;
	const float3 W = parameters.cam_w;
	
	/* Jittered halton
	int xoff = rand(parameters.image_width, seed);
	int yoff = rand(parameters.image_height, seed);

	// Compute ray origin and direction
	float xoffset = parameters.xoffset[xoff];
	float yoffset = parameters.yoffset[yoff];
	radius = sqrt(xoffset * xoffset + yoffset * yoffset)/sqrt(0.5f); */

	random3(seed);
	
	float xoffset = fract(seed.x) - 0.5f;
	float yoffset = fract(seed.y) - 0.5f;

	float2 d = 2.0f * make_float2(
		float(idx.x + xoffset)/parameters.resolution.x,
		float(idx.y + yoffset)/parameters.resolution.y
	) - 1.0f;

	origin = parameters.camera;
	direction = normalize(d.x * U + d.y * V + W);
}

// Ray generation kernel
extern "C" __global__ void __raygen__rg()
{
	// Get the launch index
	const uint3 idx = optixGetLaunchIndex();

	// Index to store and read the pixel
	const uint index = idx.x + idx.y * parameters.resolution.x;

	// Prepare the ray packet
	RayPacket rp {
		.value = make_float3(0.0f),
		.ior = 1.0f,
		.depth = 0,
		.index = index,
		.seed = make_float3(idx.x, idx.y, parameters.time)
	};
	
	// Trace ray and generate contribution
	unsigned int i0, i1;
	pack_pointer(&rp, i0, i1);

	float3 origin;
	float3 direction;

	make_ray(idx, origin, direction, rp.seed);

	optixTrace(parameters.traversable,
		origin, direction,
		0.0f, 1e16f, 0.0f,
		OptixVisibilityMask(0b11),
		OPTIX_RAY_FLAG_DISABLE_ANYHIT,
		0, 0, 0,
		i0, i1
	);
		
	// Finally, store the result
	float4 sample = make_float4(rp.value, 1.0f);
	if (parameters.accumulate && false) {
		float4 prev = parameters.color_buffer[index];
		parameters.color_buffer[index] = (prev * parameters.samples + sample)
			/(parameters.samples + 1);
	} else {
		parameters.color_buffer[index] = sample;
	}
}

// Interpolate triangle values
template <class T>
KCUDA_INLINE __device__
T interpolate(T *arr, uint3 triagle, float2 bary)
{
	T a = arr[triagle.x];
	T b = arr[triagle.y];
	T c = arr[triagle.z];

	return (1.0f - bary.x - bary.y) * a + bary.x * b + bary.y * c;
}

// Calculate hit normal
static __device__ float3 calculate_normal
		(Hit *hit_data, uint3 triangle,
		 float2 bary, float2 uv, bool &entering)
{
	float3 e1 = hit_data->vertices[triangle.y] - hit_data->vertices[triangle.x];
	float3 e2 = hit_data->vertices[triangle.z] - hit_data->vertices[triangle.x];
	float3 ng = cross(e1, e2);

	if (dot(ng, optixGetWorldRayDirection()) > 0.0f) {
		ng = -ng;
		entering = false;
	} else {
		entering = true;
	}

	ng = normalize(ng);

	float3 normal = interpolate(hit_data->normals, triangle, bary);
	if (dot(normal, ng) < 0.0f)
		normal = -normal;

	normal = normalize(normal);

	if (hit_data->textures.has_normal) {
		float4 n4 = tex2D <float4> (hit_data->textures.normal, uv.x, uv.y);
		float3 n = 2 * make_float3(n4.x, n4.y, n4.z) - 1;

		// Tangent and bitangent
		float3 tangent = interpolate(hit_data->tangents, triangle, bary);
		float3 bitangent = interpolate(hit_data->bitangents, triangle, bary);

		mat3 tbn = mat3(
			normalize(tangent),
			normalize(bitangent),
			normalize(normal)
		);

		normal = normalize(tbn * n);
	}

	return normal;
}

// Calculate relevant material data for a hit
KCUDA_INLINE __device__
void calculate_material(Hit *hit_data, Material &mat, uint3 triangle, float2 uv)
{
	if (hit_data->textures.has_diffuse) {
		float4 d4 = tex2D <float4> (hit_data->textures.diffuse, uv.x, uv.y);
		mat.diffuse = make_float3(d4);
	}

	if (hit_data->textures.has_roughness) {
		float4 r4 = tex2D <float4> (hit_data->textures.roughness, uv.x, uv.y);
		mat.roughness = r4.x;
	}
}

// TODO: launch parameter for ray depth
// #define MAX_DEPTH 10
#define MAX_DEPTH 3

// Weighting function
KCUDA_INLINE static __device__
float weight_kernel(const PathSample &sample)
{
	return length(sample.value);
}

// Temporal resampling
KCUDA_INLINE __device__
float3 temporal_reuse(RayPacket *rp, const PathSample &sample, float weight)
{
	// Get reservoir
	auto &r_temporal = parameters.advanced.r_temporal[rp->index];

	// Proceed to add the current sample to the reservoir
	r_temporal.update(sample, weight);
	r_temporal.W = r_temporal.weight/(
		r_temporal.count * weight_kernel(r_temporal.sample)
		+ 1e-6f
	);

	// Get resampled value
	return r_temporal.sample.value;
}

// Spatiotemporal resampling
KCUDA_INLINE __device__
float3 spatiotemporal_reuse(RayPacket *rp, float3 x, float3 n)
{
	// X and Y of the pixel
	int ix = rp->index % parameters.resolution.x;
	int iy = rp->index / parameters.resolution.x;

	// Then use spatial resampling
	auto &r_spatial = parameters.advanced.r_spatial[rp->index];
	auto &r_temporal = parameters.advanced.r_temporal[rp->index];
	auto &s_radius = parameters.advanced.sampling_radii[rp->index];

	int Z = 0;
	int success = 0;

	r_spatial.merge(r_temporal, weight_kernel(r_temporal.sample));
	Z += r_temporal.count;
	
	const int SPATIAL_SAMPLES = (r_spatial.count < 250) ? 9 : 3;
	for (int i = 0; i < SPATIAL_SAMPLES; i++) {
		// Generate random neighboring pixel
		random3(rp->seed);

		float radius = s_radius * fract(rp->seed.x);
		float angle = 2 * M_PI * fract(rp->seed.y);

		int ny = iy + radius * sin(angle);
		int nx = ix + radius * cos(angle);
		
		if ((nx < 0 || nx >= parameters.resolution.x)
				|| (ny < 0 || ny >= parameters.resolution.y))
			continue;

		int nindex = ny * parameters.resolution.x + nx;

		// Get the appropriate reservoir
		auto *reservoir = &parameters.advanced.r_spatial_prev[nindex];
		if (reservoir->count > 50)
			reservoir = &parameters.advanced.r_temporal_prev[nindex];

		if (reservoir->count == 0)
			continue;

		// Get information relative to sample
		auto &sample = reservoir->sample;

		// Check geometry similarity
		float depth_x = length(x - parameters.camera);
		float depth_s = length(sample.p_pos - parameters.camera);

		float theta = 180 * acos(dot(n, sample.s_normal))/M_PI;
		float ndepth = abs(depth_x - depth_s)/max(depth_x, depth_s);

		if (angle > 25 || ndepth > 0.1)
			continue;

		// Check if the sample is visible
		float3 direction = normalize(sample.p_pos - x);
		float distance = length(sample.p_pos - x);

		bool occluded;
		if (sample.missed)
			occluded = is_occluded(x + sample.dir * eps, sample.dir, 1e6);
		else
			occluded = is_occluded(x + direction * eps, direction, distance);

		if (occluded)
			continue;

		// Compute Jacobian
		float3 xq_1 = sample.p_pos;
		float3 xq_2 = sample.s_pos;
		float3 xr_1 = x;

		float3 v_r = xr_1 - xq_2;
		float3 v_q = xq_1 - xq_2;

		float d_r = length(v_r);
		float d_q = length(v_q);

		v_r /= d_r;
		v_q /= d_q;

		float phi_r = acos(dot(sample.s_normal, v_r));
		float phi_q = acos(dot(sample.s_normal, v_q));

		float J = abs(phi_r/phi_q) * (d_q * d_q)/(d_r * d_r);

		// If conditions are sufficient, merge reservoir
		r_spatial.merge(
			*reservoir,
			weight_kernel(reservoir->sample)/J
		);

		Z += reservoir->count;
		success++;
	}

	// Compute final weight
	r_spatial.W = r_spatial.weight/(
		Z * weight_kernel(r_spatial.sample)
		+ 1e-6f
	);

	// Reduce radius if no samples were found
	if (success == 0)
		s_radius = max(s_radius * 0.5f, 3.0f);

	// Get resampled value
	return r_spatial.sample.value;
}

// Closest hit kernel
extern "C" __global__ void __closesthit__ch()
{
	// Get payload
	RayPacket *rp;
	unsigned int i0 = optixGetPayload_0();
	unsigned int i1 = optixGetPayload_1();
	rp = unpack_pointer <RayPacket> (i0, i1);
	
	if (rp->depth > MAX_DEPTH)
		return;

	// Check if primary ray
	bool primary = (rp->depth == 0);
	
	// Get data from the SBT
	Hit *hit = reinterpret_cast <Hit *> (optixGetSbtDataPointer());

	// Calculate relevant data for the hit
	float2 bary = optixGetTriangleBarycentrics();
	int primitive_index = optixGetPrimitiveIndex();
	uint3 triangle = hit->triangles[primitive_index];

	// Get UV coordinates
	float2 uv = interpolate(hit->texcoords, triangle, bary);
	uv.y = 1 - uv.y;

	// Calculate the material
	Material material = hit->material;

	// TODO: check for light, not just emissive material
	if (hit->material.type == Shading::eEmissive) {
		rp->value = material.emission;
		return;
	}
	
	calculate_material(hit, material, triangle, uv);

	bool entering;
	float3 wo = -optixGetWorldRayDirection();
	float3 n = calculate_normal(hit, triangle, bary, uv, entering);
	float3 x = interpolate(hit->vertices, triangle, bary);

	// Offset by normal
	// TODO: use more complex shadow bias functions

	// TODO: an easier check for transmissive objects
	x += (material.type == Shading::eTransmission ? -1 : 1) * n * eps;

	float3 direct = Ld(x, wo, n, material, entering, rp->seed);

	// Generate new ray
	Shading out;
	float3 wi;
	float pdf;

	float3 f = eval(material, n, wo, entering, wi, pdf, out, rp->seed);
	if (length(f) < 1e-6f)
		return;

	// Get threshold value for current ray
	float3 T = f * abs(dot(wi, n))/pdf;

	// Recursive raytrace
	float3 offset = 1e-3f * n;
	if (out & Shading::eTransmission)
		offset = 1e-3f * wi;

	// Update ior
	rp->ior = material.refraction;
	rp->depth++;

#define VOXEL_SAMPLE_REUSE

#if defined(RESTIR_SAMPLING)

	// Recurse
	optixTrace(parameters.traversable,
		x + offset, wi,
		0.0f, 1e16f, 0.0f,
		OptixVisibilityMask(0b1),
		OPTIX_RAY_FLAG_DISABLE_ANYHIT,
		0, 0, 0,
		i0, i1
	);

	// Post: advanced sampling techniques if any
	float3 indirect = rp->value;

	// ReSTIR GI
	float max_radius = min(
		parameters.resolution.x,
		parameters.resolution.y
	)/10.0f;

	if (parameters.samples == 0) {
		auto &r_temporal = parameters.advanced.r_temporal[rp->index];
		auto &r_spatial = parameters.advanced.r_spatial[rp->index];
		auto &s_radius = parameters.advanced.sampling_radii[rp->index];

		// Reset for motion
		r_temporal.reset();
		r_spatial.reset();
		s_radius = max_radius;
	}

	if (primary && parameters.samples > 0) {
		// TODO: The ray misses if its depth is 1
		//	but not if it hits a light (check value)
		//	this fixes lights being black with ReSTIR
		bool missed = (rp->depth == 1);

		// Generate sample and weight
		PathSample sample {
			.value = rp->value,
			.dir = wi,
			.p_pos = x,
			.p_normal = n,
			.s_pos = rp->position,
			.s_normal = rp->normal,
			.missed = missed
		};

		float weight = weight_kernel(sample)/pdf;

		// First actually update the temporal reservoir
		temporal_reuse(rp, sample, weight);

		if (parameters.samples > 0) {
			// Then use spatiotemporal resampling
			indirect = spatiotemporal_reuse(rp, x, n);

			auto &r_spatial = parameters.advanced.r_temporal[rp->index];
			rp->value = direct + f * r_spatial.sample.value *
				r_spatial.W * abs(dot(wi, n))/pdf;
			// rp->value = direct + T * indirect * r_spatial.W;
			// rp->value = direct + T * indirect;
		}
	} else {
		rp->value = direct + T * indirect;
	}

	// float radius = parameters.advanced.sampling_radii[rp->index];
	// rp->value = make_float3(radius/max_radius);

	// TODO: write different programs for these, then create mutliple
	// pipelines
#elif defined(VOXEL_SAMPLE_REUSE)

	// Get voxel coordinates
	float3 v_min = parameters.voxel.min;
	float3 v_max = parameters.voxel.max;
	int res = parameters.voxel.resolution;

	int3 c = make_int3(
		(x.x - v_min.x)/(v_max.x - v_min.x) * res,
		(x.y - v_min.y)/(v_max.y - v_min.y) * res,
		(x.z - v_min.z)/(v_max.z - v_min.z) * res
	);

	c = min(c, make_int3(res - 1));

	// rp->value = make_float3((c.x + c.y + c.z) % 2);
	// return;

	// Issue with this approach using the same sample in a voxel creates
	// extreme aliasing (you can distinguish the voxels by color...)

	// TODO: screen shot of the naive approach (no spatial reuse, only
	// temporal) for reference in a writeup

	// Get reservoir at the voxel
	uint index = c.x + c.y * res + c.z * res * res;

	auto &r_voxel = parameters.voxel.reservoirs[index];
	int *lock = parameters.voxel.locks[index];

	/* bool occluded = false;
	float3 cached_sample = make_float3(0.0f);
	float3 cached_position = make_float3(0.0f);
	float3 cached_direction = make_float3(0.0f);
	float cached_W = 0; */

	/* if (parameters.samples == 0) {
		while (atomicCAS(lock, 0, 1) == 0);
		if (r_voxel.count > 0) {
			r_voxel.reset();
			r_voxel.sample.value = make_float3(0.0f);
			r_voxel.sample.position = make_float3(0.0f);
		}
		atomicExch(lock, 0);
	} */

	/* if (r_voxel.count > 0) {
		while (atomicCAS(lock, 0, 1) == 0);
		cached_position = r_voxel.sample.position;
		cached_sample = r_voxel.sample.value;
		cached_direction = r_voxel.sample.direction;
		cached_W = r_voxel.weight/(r_voxel.count * length(cached_sample) + 1e-6);
		atomicExch(lock, 0);

		// Check if the sample is occluded
		float3 L = cached_position - x;
		float3 L_n = normalize(L);
		// occluded = is_occluded(x + n * 0.01, L_n, length(L));
	} */

	float e = fract(random3(rp->seed).x);
	int count = r_voxel.count;

	
	// TODO: use a different threshold than 0.5 for spatial reuse
	// TODO: threshold should decrease over time
	float threshold = (1.0f - tanh(count/10))/2.0f;

	// TODO: analyze speedup when recursively updating voxels
	// primary = rp->depth < (MAX_DEPTH - 1);
	if (primary && count > 0 && e > threshold) {
		/* float3 brdf = kobra::cuda::brdf(material, n,
			cached_direction, wo,
			entering, material.type
		);

		float pdf = kobra::cuda::pdf(material, n,
			cached_direction, wo,
			entering, material.type
		);

		if (pdf > 0) {
			rp->value = direct + brdf * cached_sample *
				abs(dot(cached_direction, n))/pdf;
		} else {
			rp->value = make_float3(1, 0, 1);
		} */

		float3 total_indirect = make_float3(0.0f);

		// TODO: how to determine number of samples to take?
		//	probably shouldnt be too low
		const int samples = 9;
		const float radius = float(res)/100.0f;

		int success = 0;
		int n_occluded = 0;
		for (int i = 0; i < samples; i++) {
			// Generate random 3D offset index
			int3 offset = make_int3(
				int(random3(rp->seed).x * radius),
				int(random3(rp->seed).y * radius),
				int(random3(rp->seed).z * radius)
			);

			int3 nindex = c + offset;

			// Check if the offset is in bounds
			if (nindex.x < 0 || nindex.x >= res ||
				nindex.y < 0 || nindex.y >= res ||
				nindex.z < 0 || nindex.z >= res)
				continue;

			// Get the reservoir at the offset
			uint nindex_1d = nindex.x + nindex.y * res + nindex.z * res * res;

			auto &r_voxel = parameters.voxel.reservoirs[nindex_1d];

			// Lock and extract the sample
			while (atomicCAS(lock, 0, 1) == 0);
			float3 sample = r_voxel.sample.value;
			float3 position = r_voxel.sample.position;
			float3 direction = r_voxel.sample.direction;
			float W = r_voxel.W;
			int count = r_voxel.count;
			atomicExch(lock, 0);

			// Skip if the reservoir is empty
			if (count == 0)
				continue;

			// Check for occulsion
			float3 L = position - x;
			float3 L_n = normalize(L);

			bool occluded = is_occluded(x, L_n, length(L));
			if (occluded) {
				n_occluded++;
				continue;
			}

			// Add the contribution
			float3 brdf = kobra::cuda::brdf(material, n,
				direction, wo,
				entering, material.type
			);

			float pdf = kobra::cuda::pdf(material, n,
				direction, wo,
				entering, material.type
			);

			// total_indirect += sample * brdf * abs(dot(direction, n))/pdf;
			total_indirect += sample * brdf * abs(dot(direction, n)) * W;
			success++;
		}

		if (success == 0) {
			// NOTE: keep this viualization of occlusions density
			// TODO: also show sample density (i.e. the threshold value)
			rp->value = make_float3(float(n_occluded)/float(samples));
		} else
			rp->value = direct + total_indirect/success;
	} else if (primary) {
		// Recurse
		optixTrace(parameters.traversable,
			x + offset, wi,
			0.0f, 1e16f, 0.0f,
			OptixVisibilityMask(0b1),
			OPTIX_RAY_FLAG_DISABLE_ANYHIT,
			0, 0, 0,
			i0, i1
		);

		float weight = length(rp->value)/pdf;

		// Update reservoir, locking
		VoxelSample vs {
			.value = rp->value,
			.position = rp->position,
			.direction = wi,
		};

		while (atomicCAS(lock, 0, 1) == 0);
		bool selected = r_voxel.update(vs, weight);
		float3 value = r_voxel.sample.value;
		float3 position = r_voxel.sample.position;
		float3 direction = r_voxel.sample.direction;
		float W = r_voxel.W  = r_voxel.weight/(r_voxel.count * length(value) + 1e-6);
		atomicExch(lock, 0);

		// rp->value = direct + T * rp->value;
		rp->value = make_float3(0, 1, 0);
	} else {
		optixTrace(parameters.traversable,
			x + offset, wi,
			0.0f, 1e16f, 0.0f,
			OptixVisibilityMask(0b1),
			OPTIX_RAY_FLAG_DISABLE_ANYHIT,
			0, 0, 0,
			i0, i1
		);

		// rp->value = direct + T * rp->value;
		rp->value = make_float3(0, 0, 0);
	}

#else
		
	optixTrace(parameters.traversable,
		x + offset, wi,
		0.0f, 1e16f, 0.0f,
		OptixVisibilityMask(0b1),
		OPTIX_RAY_FLAG_DISABLE_ANYHIT,
		0, 0, 0,
		i0, i1
	);

	rp->value = direct + T * rp->value;

#endif

	rp->position = x;
	rp->normal = n;
}

extern "C" __global__ void __closesthit__shadow() {}

// Miss kernel
extern "C" __global__ void __miss__ms()
{
	// Get direction
	const float3 ray_direction = optixGetWorldRayDirection();

	float u = atan2(ray_direction.x, ray_direction.z)/(2.0f * M_PI) + 0.5f;
	float v = asin(ray_direction.y)/M_PI + 0.5f;

	float4 c = tex2D <float4> (parameters.envmap, u, v);

	// Transfer to payload
	RayPacket *rp;
	unsigned int i0 = optixGetPayload_0();
	unsigned int i1 = optixGetPayload_1();
	rp = unpack_pointer <RayPacket> (i0, i1);

	rp->value = make_float3(c);
}

extern "C" __global__ void __miss__shadow()
{
	unsigned int i0 = optixGetPayload_0();
	unsigned int i1 = optixGetPayload_1();
	bool *vis = unpack_pointer <bool> (i0, i1);
	*vis = false;
}

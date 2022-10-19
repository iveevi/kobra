#include "wadjet_common.cuh"

#define VOXEL_SPATIAL_REUSE
// #define VOXEL_NAIVE_RESERVOIRS

#if defined(VOXEL_SPATIAL_REUSE)

// Closest hit program for Voxel Reservoirs
extern "C" __global__ void __closesthit__voxel()
{
	LOAD_RAYPACKET();
	LOAD_INTERSECTION_DATA();

	// Check if primary ray
	bool primary = (rp->depth == 0);
	
	if (hit->material.type == Shading::eEmissive) {
		rp->value = material.emission;
		return;
	}
	
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

	// Update ior
	rp->ior = material.refraction;
	rp->depth++;
	
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


	// Get reservoir at the voxel
	uint index = c.x + c.y * res + c.z * res * res;

	auto &r_voxel = parameters.voxel.reservoirs[index];
	int *lock = parameters.voxel.locks[index];

	float e = fract(random3(rp->seed).x);
	int count = r_voxel.count;

	
	// TODO: use a different threshold than 0.5 for spatial reuse
	// TODO: threshold should decrease over time
	float threshold = (1.0f - tanh(count/10))/2.0f;

	// TODO: analyze speedup when recursively updating voxels
	if (primary && count > 100) {
		float3 total_indirect = make_float3(0.0f);

		// TODO: how to determine number of samples to take?
		//	probably shouldnt be too low
		const int samples = 9; // 25, 100, etc
		const float radius = float(res)/5.0f;

		int success = 0;
		int n_occluded = 0;
		int n_out = 0;
		int n_void = 0;

		for (int i = 0; i < samples; i++) {
			// Generate random 3D offset index
			// TODO: use spherical mapping instead of rectangular

			float3 r = fract(random3(rp->seed));
			r = r * 2.0f - 1.0f;

			int3 offset = make_int3(
				r.x * radius,
				r.y * radius,
				r.z * radius
			);

			int3 nindex = c + offset;

			// Check if the offset is in bounds
			if (nindex.x < 0 || nindex.x >= res ||
				nindex.y < 0 || nindex.y >= res ||
				nindex.z < 0 || nindex.z >= res) {
				n_out++;
				continue;
			}

			// Get the reservoir at the offset
			int nindex_1d = nindex.x + nindex.y * res + nindex.z * res * res;

			// Get voxel and lock
			auto &r_voxel = parameters.voxel.reservoirs[nindex_1d];
			int *lock = parameters.voxel.locks[nindex_1d];

			// Lock and extract the sample
			// TODO: is the lock necessary?
			// while (atomicCAS(lock, 0, 1) == 0);
			float3 sample = r_voxel.sample.value;
			float3 position = r_voxel.sample.position;
			float3 direction = r_voxel.sample.direction;
			float W = r_voxel.W;
			int count = r_voxel.count;
			// atomicExch(lock, 0);

			// Skip if the reservoir is empty
			if (count == 0) {
				n_void++;
				continue;
			}

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

		/* if (success == 0) {
			// NOTE: keep this viualization of occlusions density
			// TODO: also show sample density (i.e. the threshold value)
			// rp->value = make_float3(float(n_empty)/float(samples));

			// TODO: want to avoid this:
			trace <eVoxel> (x, wi, i0, i1);
			rp->value = direct + T * rp->value;
		} else {
			rp->value = direct + total_indirect/success;
		} */

		rp->value = make_float3(n_void)/float(samples);
	} else if (primary) {
		// Recurse
		trace <eVoxel> (x, wi, i0, i1);
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
		rp->value = make_float3(1, 0, 1);
	} else {
		trace <eVoxel> (x, wi, i0, i1);
		rp->value = direct + T * rp->value;
	}

	rp->position = x;
	rp->normal = n;
}

#elif defined(VOXEL_NAIVE_RESERVOIRS)

extern "C" __global__ void __closesthit__voxel()
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

	// Update ior
	rp->ior = material.refraction;
	rp->depth++;

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

	// Issue with this approach using the same sample in a voxel creates
	// extreme aliasing (you can distinguish the voxels by color...)

	// TODO: screen shot of the naive approach (no spatial reuse, only
	// temporal) for reference in a writeup

	// Get reservoir at the voxel
	uint index = c.x + c.y * res + c.z * res * res;

	auto &r_voxel = parameters.voxel.reservoirs[index];
	int *lock = parameters.voxel.locks[index];

	bool occluded = false;
	float3 cached_sample = make_float3(0.0f);
	float3 cached_position = make_float3(0.0f);
	float3 cached_direction = make_float3(0.0f);
	float cached_W = 0;

	if (r_voxel.count > 0) {
		while (atomicCAS(lock, 0, 1) == 0);
		cached_position = r_voxel.sample.position;
		cached_sample = r_voxel.sample.value;
		cached_direction = r_voxel.sample.direction;
		cached_W = r_voxel.weight/(r_voxel.count * length(cached_sample) + 1e-6);
		atomicExch(lock, 0);

		// Check if the sample is occluded
		float3 L = cached_position - x;
		float3 L_n = normalize(L);
		occluded = is_occluded(x + n * 0.01, L_n, length(L));
	}

	// TODO: analyze speedup when recursively updating voxels
	int count = r_voxel.count;
	
	if (primary && count > 0 && !occluded) {
		float3 brdf = kobra::cuda::brdf(material, n,
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
			rp->value = direct;
		}
	} else if (primary) {
		// Recurse
		trace <eVoxel> (x, wi, i0, i1);
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

		rp->value = direct + T * rp->value;
	} else {
		trace <eVoxel> (x, wi, i0, i1);
		rp->value = direct + T * rp->value;
	}

	rp->position = x;
	rp->normal = n;
}

#endif

#include "basilisk_common.cuh"

// #define VOXEL_SPATIAL_REUSE
// #define VOXEL_NAIVE_RESERVOIRS
// #define TEXTURE_MAPPED_RESERVOIRS
#define KD_TREE_RESERVOIRS
// #define BACKUP_RIS

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

	float e = rand_uniform(rp->seed);
	int count = r_voxel.count;

	
	// TODO: use a different threshold than 0.5 for spatial reuse
	// TODO: threshold should decrease over time
	float threshold = (1.0f - tanh(count/10))/2.0f;

	// TODO: analyze speedup when recursively updating voxels
	if (primary && count > 10 && e > 0.25) {
		float3 total_indirect = make_float3(0.0f);

		// TODO: how to determine number of samples to take?
		//	probably shouldnt be too low
		const int samples = 3; // 25, 100, etc
		const float max_radius = float(res);

		int success = 0;
		int n_occluded = 0;
		int n_out = 0;
		int n_void = 0;

		for (int i = 0; i < samples; i++) {
			// Generate random 3D offset index
			// TODO: use spherical mapping instead of rectangular
			
			float3 r = rand_uniform_3f(rp->seed);

			// NOTE: sqrt of the random variable results in larger
			// radii
			float radius = fract(sqrt(pcg3f(r).x)) * max_radius;

			// TODO: select between these filters by sampling ~5
			// from each in the inital stage
#if 0

			// Cubic sampling
			r = r * 2.0f - 1.0f;

			int3 offset = make_int3(
				r.x * radius,
				r.y * radius,
				r.z * radius
			);

#elif 0

			// Spherical sampling
			float theta = r.x * 2.0f * M_PI;
			float phi = r.y * M_PI;

			float3 pre_offset = make_int3(
				radius * sin(phi) * cos(theta),
				radius * sin(phi) * sin(theta),
				radius * cos(phi)
			);

			// pre_offset += n * 

#else

			// Normal disk sampling
			float theta = r.x * 2.0f * M_PI;

			// Get vectors orthogonal to n
			const float3 up = make_float3(0.0f, 1.0f, 0.0f);
			const float3 right = make_float3(1.0f, 0.0f, 0.0f);

			float3 u = normalize(cross(n, up));
			if (length(u) < 1e-6f)
				u = normalize(cross(n, right));

			float3 v = normalize(cross(n, u));

			float3 pre_offset = make_float3(
				radius * cos(theta) * u.x + radius * sin(theta) * v.x,
				radius * cos(theta) * u.y + radius * sin(theta) * v.y,
				radius * cos(theta) * u.z + radius * sin(theta) * v.z
			);

			pre_offset += 0.5 * n * (2 * r.y - 1);
			int3 offset = make_int3(pre_offset);

#endif

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
			float3 sample = r_voxel.samples[0].value;
			float3 position = r_voxel.samples[0].position;
			float3 direction = r_voxel.samples[0].direction;
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
		
		// TODO: spatial reservoirs as well...

// #define VISUALIZE

		if (success == 0) {
#ifdef VISUALIZE
			if (n_void > n_occluded)
				rp->value = make_float3(0, 0, n_void)/float(samples);
			else
				rp->value = make_float3(n_occluded, 0, 0)/float(samples);
#else
			// TODO: want to avoid this:
			trace <eVoxel> (x, wi, i0, i1);
			rp->value = direct + T * rp->value;
#endif
		} else {
#ifdef VISUALIZE
			rp->value = make_float3(0, 1, 0);
#else
			rp->value = direct + total_indirect/success;
#endif
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
		float3 value = r_voxel.samples[0].value;
		float3 position = r_voxel.samples[0].position;
		float3 direction = r_voxel.samples[0].direction;
		float W = r_voxel.W  = r_voxel.weight/(r_voxel.count * length(value) + 1e-6);
		atomicExch(lock, 0);

#ifdef VISUALIZE
		rp->value = make_float3(1, 0, 1);
#else
		rp->value = direct + T * rp->value;
#endif
	} else {
		// Regular rays
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

	int success = 0;

	float3 total_indirect = make_float3(0);

	// primary = ((MAX_DEPTH - rp->depth) >= MAX_DEPTH);
	primary = (rp->depth < 1);
	for (int i = 0; i < Voxel_Reservoir::size; i++) {
		// while (atomicCAS(lock, 0, 1) == 0);

		auto sample = r_voxel.samples[i];

		float3 value = sample.value;
		float3 position = sample.position;
		float3 direction = sample.direction;

		// atomicExch(lock, 0);

		// Check if the sample is occluded
		float3 L = position - x;
		float3 L_n = normalize(L);

		bool occluded = is_occluded(x + n * 0.01, L_n, length(L));
		if (occluded)
			continue;

		// Add to indirect lighting
		float pdf = kobra::cuda::pdf(material, n,
			direction, wo,
			entering, material.type
		);

		if (isnan(pdf) || pdf < 0.01)
			continue;

		/* if (isnan(pdf) || isnan(1.0/pdf)) {
			printf("pdf: %f\t1/pdf: %f\n", pdf, 1.0/pdf);
			assert(false);
		} */
		
		float3 brdf = kobra::cuda::brdf(material, n,
			direction, wo,
			entering, material.type
		);

		float3 f = brdf * abs(dot(direction, n))/pdf;
		total_indirect += value * f;

		success++;
	}

	// Reuse only if primary
	float r = fract(random3(rp->seed).x);
	if (primary && success > 0) {
		// rp->value = make_float3(success/float(Voxel_Reservoir::size));
		rp->value = direct + total_indirect/float(success);
		return;
	}

	// Regular rays and add to reservoir
	rp->depth++;
	trace <eVoxel> (x, wi, i0, i1);

	// Construct sample if primary ray
	if (primary) {
		VoxelSample sample {
			.value = rp->value,
			.position = x,
			.direction = wi
		};

		float weight = length(sample.value)/pdf;

		// Add to reservoir
		while (atomicCAS(lock, 0, 1) == 0);
		r_voxel.update(sample, weight);
		atomicExch(lock, 0);
		
		// rp->value = make_float3(1, 0, 0);
		// return;
	}

	// rp->value = make_float3(1, 0, 0);
	rp->value = direct + T * rp->value;

	/* bool occluded = false;
	float3 cached_sample = make_float3(0.0f);
	float3 cached_position = make_float3(0.0f);
	float3 cached_direction = make_float3(0.0f);
	float cached_W = 0;

	if (r_voxel.count > 0) {
		while (atomicCAS(lock, 0, 1) == 0);
		cached_position = r_voxel.samples[0].position;
		cached_sample = r_voxel.samples[0].value;
		cached_direction = r_voxel.samples[0].direction;
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
				cached_W * abs(dot(cached_direction, n));
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
		float3 value = r_voxel.samples[0].value;
		float3 position = r_voxel.samples[0].position;
		float3 direction = r_voxel.samples[0].direction;
		float W = r_voxel.W  = r_voxel.weight/(r_voxel.count * length(value) + 1e-6);
		atomicExch(lock, 0);

		rp->value = direct + T * rp->value;
	} else {
		trace <eVoxel> (x, wi, i0, i1);
		rp->value = direct + T * rp->value;
	} */

	rp->position = x;
	rp->normal = n;
}

#elif defined(TEXTURE_MAPPED_RESERVOIRS)

// TMRIS
// TODO: move to separate file and kernel
extern "C" __global__ void __closesthit__voxel()
{
	// TODO: resolution based on mesh size/complexity (mostly size)
	constexpr int res = Hit::TMRIS_RESOLUTION;

	LOAD_RAYPACKET();
	LOAD_INTERSECTION_DATA();

	BoundingBox bbox = hit->bbox;

	glm::vec3 gcentroid = (bbox.min + bbox.max)/2.0f;

	float3 centroid {gcentroid.x, gcentroid.y, gcentroid.z};
	float extent_z = bbox.max.z - bbox.min.z;
	float extent_x = bbox.max.x - bbox.min.x;
	float extent_y = bbox.max.y - bbox.min.y;

	// TODO: projected axis must be computed... assume z for now
	float3 dx = x - centroid;

	float u = (dx.x + extent_x/2.0f)/extent_x;
	float v = (dx.y + extent_y/2.0f)/extent_y;

	bool forward = dx.z > 0;

	int ix = u * res;
	int iy = v * res;

	int mod = (ix + iy) % 2;
	rp->value = make_float3(mod);

	/* Spatial sampling test for "more continuous" mapping
	const int SAMPLES = 10;
	const float SAMPLE_RADIUS = res * 0.0f;

	float sum_u = 0;
	float sum_v = 0;

	int over_count = 0;

	for (int i = 0; i < SAMPLES; i++) {
		float3 eta = rand_uniform_3f(rp->seed);

		float r = SAMPLE_RADIUS * sqrt(eta.x);
		float theta = 2.0 * M_PI * eta.y;

		int offx = r * cos(theta);
		int offy = r * sin(theta);

		int nix = ix + offx;
		int niy = iy + offy;

		bool over_x = nix < 0 || nix >= res;
		bool over_y = niy < 0 || niy >= res;

		if (over_x || over_y) {
			// Flip
			if (over_x)
				nix = ix - offx;
			if (over_y)
				niy = iy - offy;

			over_count++;
		}

		// assert(nix >= 0 && nix < res);
		// assert(niy >= 0 && niy < res);

		sum_u += nix/float(res);
		sum_v += niy/float(res);
	}

	rp->value = make_float3(sum_u, sum_v, 0)/float(SAMPLES); */
}

#elif defined(KD_TREE_RESERVOIRS)

// TODO: move to separate file and kernel
__forceinline__ __device__
float get(float3 a, int axis)
{
	if (axis == 0) return a.x;
	if (axis == 1) return a.y;
	if (axis == 2) return a.z;
}

extern "C" __global__ void __closesthit__voxel()
{
	LOAD_RAYPACKET();
	LOAD_INTERSECTION_DATA();

	// TODO: first pass of rays is proxy for initialization?
	// TODO: extra buffer for direct lighting only, so that we can continue
	// with full lighting and show actual results?

	// Offset by normal
	x += (material.type == Shading::eTransmission ? -1 : 1) * n * eps;
	
	// Construct SurfaceHit instance for lighting calculations
	SurfaceHit surface_hit {
		.mat = material,
		.entering = entering,
		.n = n,
		.wo = wo,
		.x = x,
	};

	// Reservoir for spatial sampling
	LightReservoir spatial {
		.sample = LightSample {},
		.count = 0,
		.weight = 0.0f,
	};

	// TODO: combine with vanilla ReSTIR

	// Obtain direct lighting sample
	// NOTE: decorrelating samples places into local and world space
	// reservoirs by using different samples for each
	// TODO: observe whether this is actually beneficial
	FullLightSample fls = sample_direct(rp->seed);

	// Compute target function (unocculted lighting)
	float3 D = fls.point - surface_hit.x;
	float d = length(D);
	D /= d;

	float3 Li = direct_occluded(surface_hit, fls.Le, fls.normal, fls.type, D, d);
		
	// Contribution and weight
	float target = Li.x + Li.y + Li.z; // Luminance
	float pdf = fls.pdf;
	
	float w = (pdf > 0.0f) ? target/pdf : 0.0f;
		
	// Update reservoir
	// TODO: initialize sample to use
	reservoir_update(&spatial, LightSample {
		.value = Li,
		.target = target,
		.type = fls.type,
		.index = fls.index
	}, w, rp->seed);

	// World space resampling
	float3 direct = make_float3(0);

	if (parameters.kd_tree) {
		FullLightSample fls = sample_direct(rp->seed);

		// Compute target function (unocculted lighting)
		float3 D = fls.point - surface_hit.x;
		float d = length(D);
		D /= d;

		float3 Li = direct_unoccluded(surface_hit, fls.Le, fls.normal, fls.type, D, d);
			
		// Contribution and weight
		float target = Li.x + Li.y + Li.z; // Luminance
		float pdf = fls.pdf;
		
		float w = (pdf > 0.0f) ? target/pdf : 0.0f;

		// TODO: skip traversal if w is zero?

		// Traverse the kd-tree
		WorldNode *kd_node = nullptr;
		int *lock = nullptr;

		int root = 0;
		int depth = 0;

		int lefts = 0;
		int rights = 0;

		float3 pos = surface_hit.x;
		
		while (root >= 0) {
			depth++;
			kd_node = &parameters.kd_tree[root];
			lock = parameters.kd_locks[root];
			
			// If no valid branches, exit
			int left = kd_node->left;
			int right = kd_node->right;

			if (left == -1 && right == -1)
				break;

			// If only one valid branch, traverse it
			if (left == -1) {
				root = right;
				rights++;
				continue;
			}

			if (right == -1) {
				root = left;
				lefts++;
				continue;
			}

			// Otherwise, choose the branch according to the split
			float split = kd_node->split;
			int axis = kd_node->axis;

			if (get(pos, axis) < split) {
				root = left;
				lefts++;
			} else {
				root = right;
				rights++;
			}
		}

		// rp->value = make_float3(length(x - kd_node->point));
		// return;

		// Lock and update the reservoir
		// TODO: similar scoped lock as std::lock_guard, in cuda/sync.h
		while (atomicCAS(lock, 0, 1) == 0);	// Lock

		int res_idx = kd_node->data;
		auto *reservoir = &parameters.kd_reservoirs[res_idx];
		auto *sample = &reservoir->sample;

		reservoir_update(reservoir, LightSample {
			.value = fls.Le,
			.point = fls.point,
			.normal = fls.normal,
			.target = target,
			.type = fls.type,
			.index = fls.index
		}, w, rp->seed);

		LightSample ls = *sample;
		float w_sum = reservoir->weight;
		int count = reservoir->count;

		atomicExch(lock, 0);			// Unlock

		// Compute lighting again
		// TODO: spatial reservoir and sampling...
		
		/* Compute value and target
		D = ls.point - surface_hit.x;
		d = length(D);
		D /= d;

		Li = direct_occluded(surface_hit, ls.value, ls.normal, D, d);
		float denom = count * ls.target;

		float W = (denom > 0.0f) ? w_sum/denom : 0.0f;

		// Insert into spatial reservoir
		target = Li.x + Li.y + Li.z; // Luminance
		w = target * W * count; // TODO: compute without doing repeated work

		reservoir_update(&spatial, LightSample {
			.value = Li,
			.target = target,
			.type = ls.type,
			.index = ls.index
		}, w, rp->seed);

		spatial.count = 1;
		if (target > 0.0f)
			spatial.count += count; */

		// TODO: two strategies
		//	hierarchical: go up a few levels and then traverse down
		//	pick a random node and traverse down
		const int SPATIAL_SAMPLES = 1;

		// Choose a root node a few level up and randomly
		// traverse the tree to obtain a sample
		const int LEVELS = 10;

		// TODO: try selecting random indices in the tree instead?
		int levels = min(depth, LEVELS);
		while (levels--) {
			kd_node = &parameters.kd_tree[root];
			lock = parameters.kd_locks[root];

			if (kd_node->parent == -1)
				break;

			root = kd_node->parent;
		}

		int successes = 0;
		for (int i = 0; i < SPATIAL_SAMPLES; i++) {
			int node = root;

			while (true) {
				kd_node = &parameters.kd_tree[node];
				lock = parameters.kd_locks[node];

				float split = kd_node->split;
				int axis = kd_node->axis;

				// If no valid branches, exit
				int left = kd_node->left;
				int right = kd_node->right;

				if (left == -1 && right == -1)
					break;

				// If only one valid branch, go there
				if (left == -1) {
					node = right;
					continue;
				}

				if (right == -1) {
					node = left;
					continue;
				}

				// Otherwise, choose a random branch
				float eta = rand_uniform(rp->seed);

				if (eta < 0.5f)
					node = left;
				else
					node = right;
			}

			// Get necessary data
			// TODO: maybe lock?
			res_idx = kd_node->data;
			reservoir = &parameters.kd_reservoirs[res_idx];
			sample = &reservoir->sample;

			// Compute value and target
			D = sample->point - surface_hit.x;
			d = length(D);
			D /= d;

			Li = direct_occluded(surface_hit, sample->value, sample->normal, sample->type, D, d);

			float denom = reservoir->count * sample->target;
			float W = (denom > 0.0f) ? reservoir->weight/denom : 0.0f;

			// Insert into spatial reservoir
			target = Li.x + Li.y + Li.z; // Luminance
			w = target * W * reservoir->count; // TODO: compute without doing repeated work

			int pcount = spatial.count;
			reservoir_update(&spatial, LightSample {
				.value = Li,
				.target = target,
				.type = sample->type,
				.index = sample->index
			}, w, rp->seed);

			spatial.count = pcount + (target > 0.0f ? reservoir->count : 0);
			successes += (target > 0.0f);
		}
	}

	// Final direct lighting result
	float denom = spatial.count * spatial.sample.target;
	float W = (denom > 0) ? spatial.weight/denom : 0.0f;

	direct = spatial.sample.value * W;
	
	// Add emission as well
	if (material.type == Shading::eEmissive)
		direct += material.emission;

	// Also compute indirect lighting
	Shading out;
	float3 wi;
	// float pdf;

	float3 f = eval(surface_hit, wi, pdf, out, rp->seed);

	// Get threshold value for current ray
	float3 T = f * abs(dot(wi, n))/pdf;

	// Update for next ray
	rp->ior = material.refraction;
	rp->pdf *= pdf;
	rp->depth++;
	
	// Trace the next ray
	float3 indirect = make_float3(0.0f);
	if (pdf > 0) {
		trace <eRegular> (x, wi, i0, i1);
		indirect = rp->value;
	}

	// Update the value
	rp->value = direct;
	if (pdf > 0)
		rp->value += T * indirect;

	rp->position = make_float4(x, 1);
}

#elif defined(BACKUP_RIS)

extern "C" __global__ void __closesthit__voxel()
{
	LOAD_RAYPACKET();
	LOAD_INTERSECTION_DATA();

	// Offset by normal
	x += (material.type == Shading::eTransmission ? -1 : 1) * n * eps;
	
	// Construct SurfaceHit instance for lighting calculations
	SurfaceHit surface_hit {
		.mat = material,
		.entering = entering,
		.n = n,
		.wo = wo,
		.x = x,
	};

	float3 direct = Ld(surface_hit, rp->seed);
	if (material.type == Shading::eEmissive)
		direct += material.emission;

	// Update ior
	rp->ior = material.refraction;
	rp->depth++;

	// Resampling Importance Sampling
	constexpr int M = 10;

	WeightedReservoir <PathSample> reservoir {
		.sample = {},
		.count = 0,
		.weight = 0.0f,
	};

	int Z = 0;
	for (int i = 0; i < M; i++) {
		// Generate new ray
		Shading out;
		float3 wi;
		float pdf;

		// TODO: cosine hemisphere sampling
		float3 f = eval(surface_hit, wi, pdf, out, rp->seed);

		// Get indirect lighting
		trace <eRegular> (x, wi, i0, i1);

		// Resampling
		float3 indirect = f * abs(dot(wi, n)) * rp->value;
		float target = length(indirect);

		float w = (pdf > 0.0f) ? target/pdf : 0.0f;

		reservoir.weight += w;

		float eta2 = rand_uniform(rp->seed);
		bool selected = (eta2 * reservoir.weight < w);

		PathSample sample {
			.value = indirect,
			.pdf = pdf * rp->pdf,
			.target = target
		};

		if (selected)
			reservoir.sample = sample;

		reservoir.count++;
		Z++;
	}

	PathSample sample = reservoir.sample;
	float denom = M * sample.target;
	float W = (denom > 0.0f) ? reservoir.weight/denom : 0.0f;
	rp->value = direct + W * sample.value;

	// Pass through features
	rp->normal = n;
	rp->albedo = material.diffuse;
}

#endif

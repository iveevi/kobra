#include "basilisk_common.cuh"

// #define VOXEL_SPATIAL_REUSE
// #define VOXEL_NAIVE_RESERVOIRS
#define TEXTURE_MAPPED_RESERVOIRS
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

template <class T>
__forceinline__ __device__
bool update_reservoir(WeightedReservoir <T> *res, const T &sample,
		float pdf_hat, float pdf, float3 &seed)
{
	res->mis += pdf + 1e-4;

	float mis = pdf/res->mis;
	float mis_confidence = 1.0f/float(res->count + 1);
	// float mis_confidence = res->count/float(res->count + 1);
	float weight = mis * mis_confidence * pdf_hat/pdf;
	
	res->weight += weight;
	res->count = min(res->count + 1, 20);
	// res->count++;

	float q = weight/res->weight;
	float eta = rand_uniform(seed);

	bool selected = eta < q;
	if (selected)
		res->sample = sample;

	return selected;
}

template <class T>
__forceinline__ __device__
void merge_reservoir(WeightedReservoir <T> *a, WeightedReservoir <T> *b,
		float pdf_hat, float3 &seed)
{
	a->mis += b->mis;

	float mis = b->mis/a->mis;
	float mis_confidence = b->count/float(a->count + b->count);
	float weight = mis * mis_confidence * pdf_hat/b->mis;

	a->weight += weight;
	a->count = min(a->count + b->count, 20);

	float q = weight/a->weight;
	float eta = rand_uniform(seed);
	if (eta < q)
		a->sample = b->sample;
}

const float isqrt2 = 0.70710676908493042;

__device__
float2 cubify(float3 s)
{
	float xx2 = s.x * s.x * 2.0;
	float yy2 = s.y * s.y * 2.0;

	float2 v {xx2 - yy2, yy2 - xx2};

	float ii = v.y - 3.0;
	ii *= ii;

	float isqrt = -sqrt(ii - 12.0 * xx2) + 3.0;

	v += {isqrt, isqrt};
	v = make_float2(sqrt(v.x), sqrt(v.y));
	v *= isqrt2;

	return v;

	// return sign(s) * make_float3(v, 1.0);
}

__device__
float2 sphere2cube(float3 sphere, int &face_index)
{
	float3 f = abs(sphere);

	bool a = f.y >= f.x && f.y >= f.z;
	bool b = f.x >= f.z;

	/* float3 s = a ? make_float3(sphere.y, sphere.z, sphere.x) :
		(b ? make_float3(sphere.x, sphere.z, sphere.y) :
		make_float3(sphere.x, sphere.y, sphere.z)); */

	// return a ? cubify(sphere.xzy).xzy : b ? cubify(sphere.yzx).zxy : cubify(sphere);
	float3 s;
	if (a) {
		s = make_float3(sphere.x, sphere.z, sphere.y);
		face_index = 0;
	} else {
		if (b) {
			s = make_float3(sphere.y, sphere.z, sphere.x);
			face_index = 1;
		} else{ 
			s = sphere;
			face_index = 2;
		}
	}

	if (s.z < 0.0) {
		s = -s;
		face_index += 3;
	}

	return cubify(s);
}

// TMRIS
// TODO: move to separate file and kernel
extern "C" __global__ void __closesthit__voxel()
{
	LOAD_RAYPACKET();
	LOAD_INTERSECTION_DATA();

	// Compute projection onto optimal plane
	float3 xvec = (x - hit->centroid) + 0.01 * n;
	float3 nvec = hit->opt_normal;

	bool forward = dot(nvec, xvec) > 0;

#define USE_PCA_MAPPING

#if defined(USE_SPHERICAL_MAPPING)

	float3 d = normalize(xvec + 0.5 * n);

	float u = atan2(d.x, d.z)/(2.0f * M_PI) + 0.5f;
	float v = asin(d.y)/M_PI + 0.5f;
	
	// TODO: dual textures?

	u = 0.5 * (u + forward);

#elif defined(USE_PCA_MAPPING)

	float3 xproj = xvec - dot(xvec, nvec) * nvec;

	float3 axis0 = hit->opt_tangent;
	float3 axis1 = hit->opt_bitangent;
	
	float u = dot(xproj, hit->opt_tangent);
	float v = dot(xproj, hit->opt_bitangent);
	
	// Normalize
	float2 u_extent = hit->extent_tangent;
	float2 v_extent = hit->extent_bitangent;

	u = (u - u_extent.x)/(u_extent.y - u_extent.x);
	v = (v - v_extent.x)/(v_extent.y - v_extent.x);

#elif defined(USE_CUBE_SPHERE_MAPPING)

	// Project to sphere, then to cube
	float3 d = normalize(xvec + 2 * n);

	int face_index;
	float2 t_uv = sphere2cube(d, face_index);
	float u = t_uv.x;
	float v = t_uv.y;

	// Split into 6 faces
	u = 0.5 * (u + int(face_index/3));
	v = (v + face_index % 3)/3.0f;

#else

	float u = uv.x;
	float v = uv.y;

#endif
	
	// TODO: reolution based on mesh size/complexity (mostly size)
	constexpr int res = Hit::TMRIS_RESOLUTION;

	int ix = u * res;
	int iy = v * res;

	int index = ix + iy * res;
	index = clamp(index, 0, res * res - 1);

#if 0

	rp->value = make_float3(
		ix/float(res),
		iy/float(res),
		forward
	);
	
	return;

#endif

	// Spatial sampling test for "more continuous" mapping
	const int SAMPLES = 10;
	const float SAMPLE_RADIUS = res * 0.1f;

	float sum_u = 0;
	float sum_v = 0;
	float sum_w = 0;

	int over_count = 0;

	for (int i = 0; i < SAMPLES; i++) {
		float3 eta = rand_uniform_3f(rp->seed);

		float r = SAMPLE_RADIUS * sqrt(eta.x);
		float theta = 2.0 * M_PI * eta.y;

		int offx = r * cos(theta);
		int offy = r * sin(theta);

		int nix = ix + offx;
		int niy = iy + offy;

		bool f = forward;

		bool over_x = nix < 0 || nix >= res;
		bool over_y = niy < 0 || niy >= res;

		if (over_x || over_y) {
			// Flip
			if (over_x)
				nix = ix - offx;
			if (over_y)
				niy = iy - offy;
			f = !f;

			over_count++;
		}

		// assert(nix >= 0 && nix < res);
		// assert(niy >= 0 && niy < res);

		sum_u += nix/float(res);
		sum_v += niy/float(res);
		sum_w += f;
	}

	rp->value = make_float3(sum_u, sum_v, sum_w)/float(SAMPLES);
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

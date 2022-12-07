#include "../../include/optix/sbt.cuh"
#include "../../include/asmodeus/wsris_grid_parameters.cuh"
#include "common.cuh"

extern "C"
{
	__constant__ kobra::optix::GridBasedReservoirsParameters parameters;
}

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

	pcg3f(seed);
	
	float xoffset = (fract(seed.x) - 0.5f);
	float yoffset = (fract(seed.y) - 0.5f);

	float2 d = 2.0f * make_float2(
		float(idx.x + xoffset)/parameters.resolution.x,
		float(idx.y + yoffset)/parameters.resolution.y
	) - 1.0f;

	origin = parameters.camera;
	direction = normalize(d.x * U + d.y * V + W);
}

// Accumulatoin helper
__forceinline__ __device__
void accumulate(float4 &dst, float4 sample)
{
	if (parameters.accumulate) {
		float4 prev = dst;
		int count = parameters.samples;
		dst = (prev * count + sample)/(count + 1);
	} else {
		dst = sample;
	}
}

////////////////////
// Sampling stage //
////////////////////

// Ray generation program
extern "C" __global__ void __raygen__samples()
{
	// Get the launch index
	const uint3 idx = optixGetLaunchIndex();

	// Index to store and read the pixel
	const uint index = idx.x + idx.y * parameters.resolution.x;

	// Prepare the ray packet
	RayPacket rp {
		.value = make_float3(0.0f),
		.position = make_float4(0),
		.pdf = 1.0f,
		.miss_depth = -1,
		.ior = 1.0f,
		.depth = 0,
		.index = index,
	};
	
	// Trace ray and generate contribution
	unsigned int i0, i1;
	pack_pointer(&rp, i0, i1);

	float3 origin;
	float3 direction;

	make_ray(idx, origin, direction, rp.seed);

	// TODO: seed generatoin method
	rp.seed = make_float3(
		sin(idx.x - idx.y),
		parameters.samples,
		parameters.time
	);

	rp.seed.x *= origin.x;
	rp.seed.y *= origin.y - 1.0f;
	rp.seed.z *= direction.z;

	// First reset corresponding reservoir
	auto &gb_ris = parameters.gb_ris;

	gb_ris.new_samples[index] = Reservoir {
		.sample = GRBSample {},
		.count = 0,
		.weight = 0.0f,
		.mis = 0.0f,
	};

	// Switch on the ray type
	trace(parameters.traversable, 1, origin, direction, i0, i1);

	// Store data
	parameters.position_buffer[index] = rp.position;
}

// Target function
__forceinline__ __device__
float target_function(float3 Li)
{
	return Li.x + Li.y + Li.z;
}

// Closest hit kernel
// TODO: disable miss shader
extern "C" __global__ void __closesthit__samples()
{
	// Load all necessary data
	LOAD_RAYPACKET();
	LOAD_INTERSECTION_DATA();

	// Offset by normal
	// TODO: use more complex shadow bias functions
	// TODO: an easier check for transmissive objects
	x += (material.type == Shading::eTransmission ? -1 : 1) * n * eps;
	
	// Construct SurfaceHit instance for lighting calculations
	SurfaceHit surface_hit {
		.mat = material,
		.entering = entering,
		.n = n,
		.wo = wo,
		.x = x,
	};

	LightingContext lc {
		parameters.traversable,
		parameters.lights.quads,
		parameters.lights.triangles,
		parameters.lights.quad_count,
		parameters.lights.triangle_count,
		parameters.has_envmap,
		parameters.envmap,
	};

	auto &gb_ris = parameters.gb_ris;

	// Create a reservoir for the sample
	Reservoir &reservoir = gb_ris.new_samples[rp->index];
	
	FullLightSample fls = sample_direct(lc, surface_hit, rp->seed);
	
	// Compute lighting
	float3 D = fls.point - surface_hit.x;
	float d = length(D);
	D /= d;

	float3 Li = direct_unoccluded(surface_hit, fls.Le, fls.normal, fls.type, D, d);

	// Resampling
	float target = target_function(Li);
	float pdf = fls.pdf;
	float w = (pdf > 0.0f) ? target/pdf : 0.0f;

	reservoir_update(&reservoir, GRBSample {
		.source = surface_hit.x,
		.value = fls.Le,
		.point = fls.point,
		.normal = fls.normal,
		.target = target,
		.type = fls.type,
		.index = fls.index
	}, w, rp->seed);

	// Update the reservoir
	gb_ris.new_samples[rp->index] = reservoir;

	// Of course we need the position
	rp->position = make_float4(surface_hit.x, 1.0f);
}

//////////////////////
// Evaluation stage //
//////////////////////

// TODO: rather than retracing, start from the previous sample
// evaluate the direct lighting and get indirect lighting
extern "C" __global__ void __raygen__eval()
{
	// Get the launch index
	const uint3 idx = optixGetLaunchIndex();

	// Index to store and read the pixel
	const uint index = idx.x + idx.y * parameters.resolution.x;

	// Prepare the ray packet
	RayPacket rp {
		.value = make_float3(0.0f),
		.position = make_float4(0),
		.pdf = 1.0f,
		.miss_depth = -1,
		.ior = 1.0f,
		.depth = 0,
		.index = index,
	};
	
	// Trace ray and generate contribution
	unsigned int i0, i1;
	pack_pointer(&rp, i0, i1);

	float3 origin;
	float3 direction;

	make_ray(idx, origin, direction, rp.seed);

	// TODO: seed generatoin method
	rp.seed = make_float3(
		sin(idx.x - idx.y),
		parameters.samples,
		parameters.time
	);

	rp.seed.x *= origin.x;
	rp.seed.y *= origin.y - 1.0f;
	rp.seed.z *= direction.z;

	// Switch on the ray type
	// TODO: skip the templates, just pass the mode on...
	trace(parameters.traversable, 1, origin, direction, i0, i1);

	// Finally, store the result
	float4 sample = make_float4(rp.value, 1.0f);
	float4 normal = make_float4(rp.normal, 0.0f);
	float4 albedo = make_float4(rp.albedo, 0.0f);

	// Check for NaNs
	if (isnan(sample.x) || isnan(sample.y) || isnan(sample.z))
		sample = make_float4(1, 0, 1, 1);

	// Accumulate necessary data
	accumulate(parameters.color_buffer[index], sample);
	accumulate(parameters.normal_buffer[index], normal);
	accumulate(parameters.albedo_buffer[index], albedo);

	// Store data
	parameters.position_buffer[index] = rp.position;
}

// Ray box intersection
__device__
float ray_x_box(float3 o, float3 d, float3 bmin, float3 bmax)
{
	float3 inv_d = 1.0f / d;
	float3 t0 = (bmin - o) * inv_d;
	float3 t1 = (bmax - o) * inv_d;

	float3 tmin = fminf(t0, t1);
	float3 tmax = fmaxf(t0, t1);

	float tmin_max = fmaxf(fmaxf(tmin.x, tmin.y), tmin.z);
	float tmax_min = fminf(fminf(tmax.x, tmax.y), tmax.z);

	return (tmin_max < tmax_min) ? tmin_max : -1.0f;
}

// Closest hit kernel
extern "C" __global__ void __closesthit__eval()
{
	// Load all necessary data
	LOAD_RAYPACKET();
	LOAD_INTERSECTION_DATA();

	bool primary = (rp->depth == 0);

	// Offset by normal
	// TODO: use more complex shadow bias functions
	// TODO: an easier check for transmissive objects
	x += (material.type == Shading::eTransmission ? -1 : 1) * n * eps;
	
	// Construct SurfaceHit instance for lighting calculations
	SurfaceHit surface_hit {
		.mat = material,
		.entering = entering,
		.n = n,
		.wo = wo,
		.x = x,
	};

	LightingContext lc {
		parameters.traversable,
		parameters.lights.quads,
		parameters.lights.triangles,
		parameters.lights.quad_count,
		parameters.lights.triangle_count,
		parameters.has_envmap,
		parameters.envmap,
	};

	// Get grid index
	float3 delta = (surface_hit.x - parameters.camera) + optix::GBR_SIZE;

	int dim = optix::GRID_RESOLUTION;
	int ix = (int) (dim * delta.x / (2 * optix::GBR_SIZE));
	int iy = (int) (dim * delta.y / (2 * optix::GBR_SIZE));
	int iz = (int) (dim * delta.z / (2 * optix::GBR_SIZE));

	// TODO: M capping
	float3 direct = make_float3(0.0);
	if (ix < 0 || ix >= dim || iy < 0 || iy >= dim || iz < 0 || iz >= dim) {
		auto &gb_ris = parameters.gb_ris;
		if (gb_ris.reproject) {
			// Get closest cell to camera; ray box intersection
			float3 bmin = parameters.camera - optix::GBR_SIZE;
			float3 bmax = parameters.camera + optix::GBR_SIZE;
			float3 d = normalize(parameters.camera - surface_hit.x);

			float t = ray_x_box(surface_hit.x, d, bmin, bmax);
			float3 new_x = surface_hit.x + (t + optix::GBR_SIZE * 0.1f) * d;

			// Get new grid index
			delta = (new_x - parameters.camera) + optix::GBR_SIZE;

			ix = (int) (dim * delta.x / (2 * optix::GBR_SIZE));
			iy = (int) (dim * delta.y / (2 * optix::GBR_SIZE));
			iz = (int) (dim * delta.z / (2 * optix::GBR_SIZE));

			int cell = ix + iy * dim + iz * dim * dim;
			int mod = (ix + iy + iz) % 2;
			
			int cell_size = gb_ris.cell_sizes[cell];

			float min_dist = 1e10f;
			int min_index = -1;

			Reservoir local {
				.sample = {},
				.count = 0,
				.weight = 0.0f,
			};

			int spatial_samples = 3;
			float3 radius = make_float3(10);
			float3 sum = make_float3(0);

			// TODO: multireservoir; then choose closest from each reservoir
			// TODO: remove the duplicate code...

			int Z = 0;
			for (int i = 0; i < spatial_samples; i++) {
				// Generate random cell offset
				float3 offset_f = 2 * rand_uniform_3f(rp->seed) - 1;
				offset_f *= radius;

				// TODO: spherical...
				int3 offset = make_int3(offset_f);

				int nx = ix + offset.x;
				int ny = iy + offset.y;
				int nz = iz + offset.z;

				if (nx < 0 || nx >= dim)
					nx = ix - offset.x;

				if (ny < 0 || ny >= dim)
					ny = iy - offset.y;

				if (nz < 0 || nz >= dim)
					nz = iz - offset.z;

				/* if (nx < 0 || nx >= dim || ny < 0 || ny >= dim || nz < 0 || nz >= dim)
					continue; */

				int ncell = nx + ny * dim + nz * dim * dim;

				auto &gb_ris = parameters.gb_ris;

				int rindex = ncell * GBR_RESERVOIR_COUNT;
				rindex += rand_uniform(GBR_RESERVOIR_COUNT, rp->seed);

				const Reservoir &reservoir = gb_ris.light_reservoirs_old[rindex];

				// Get the sample
				const GRBSample &sample = reservoir.sample; // TODO: refactor to GBR...
				float denom = reservoir.count * sample.target;
				float W = (denom > 0 ? reservoir.weight/denom : 0.0f);

				float3 D = sample.point - surface_hit.x;
				float d = length(D);
				D /= d;

				float3 Le = sample.value;
				float3 Li = direct_occluded(
					lc.handle,
					surface_hit, Le, sample.normal,
					sample.type, D, d
				);

				sum += Li * W;
				
				float target = target_function(Li);
				float w = reservoir.count * target * W;

				int count = local.count;
				reservoir_update(&local, GRBSample {
					.value = Li,
					.target = target,
				}, w, rp->seed);
				local.count = count + reservoir.count;

				if (sample.target > 1e-6f)
					Z += reservoir.count;
			}

			sum /= spatial_samples;

			float denom = local.count * local.sample.target;
			float W = (denom > 0 ? local.weight/denom : 0.0f);
			direct = material.emission + local.sample.value * W;
		} else {
			direct = material.emission + Ld(lc, surface_hit, rp->seed);
		}
	} else {
		// TODO: method
		// Get reservoirs
		int cell = ix + iy * dim + iz * dim * dim;
		int mod = (ix + iy + iz) % 2;
		
		auto &gb_ris = parameters.gb_ris;
		int cell_size = gb_ris.cell_sizes[cell];

		float min_dist = 1e10f;
		int min_index = -1;

		Reservoir local {
			.sample = {},
			.count = 0,
			.weight = 0.0f,
		};

		int spatial_samples = 3;
		float3 radius = make_float3(10);
		float3 sum = make_float3(0);

		int Z = 0;
		for (int i = 0; i < spatial_samples; i++) {
			// Generate random cell offset
			float3 offset_f = 2 * rand_uniform_3f(rp->seed) - 1;
			offset_f *= radius;

			// TODO: spherical...
			int3 offset = make_int3(offset_f);

			int nx = ix + offset.x;
			int ny = iy + offset.y;
			int nz = iz + offset.z;

			if (nx < 0 || nx >= dim)
				nx = ix - offset.x;

			if (ny < 0 || ny >= dim)
				ny = iy - offset.y;

			if (nz < 0 || nz >= dim)
				nz = iz - offset.z;

			/* if (nx < 0 || nx >= dim || ny < 0 || ny >= dim || nz < 0 || nz >= dim)
				continue; */

			int ncell = nx + ny * dim + nz * dim * dim;

			auto &gb_ris = parameters.gb_ris;

			int rindex = ncell * GBR_RESERVOIR_COUNT;
			rindex += rand_uniform(GBR_RESERVOIR_COUNT, rp->seed);

			const Reservoir &reservoir = gb_ris.light_reservoirs_old[rindex];

			// Get the sample
			const GRBSample &sample = reservoir.sample; // TODO: refactor to GBR...
			float denom = reservoir.count * sample.target;
			float W = (denom > 0 ? reservoir.weight/denom : 0.0f);

			float3 D = sample.point - surface_hit.x;
			float d = length(D);
			D /= d;

			float3 Le = sample.value;
			float3 Li = direct_occluded(
				lc.handle,
				surface_hit, Le, sample.normal,
				sample.type, D, d
			);

			sum += Li * W;
			
			float target = target_function(Li);
			float w = reservoir.count * target * W;

			int count = local.count;
			reservoir_update(&local, GRBSample {
				.value = Li,
				.target = target,
			}, w, rp->seed);
			local.count = count + reservoir.count;

			if (sample.target > 1e-6f)
				Z += reservoir.count;
		}

		sum /= spatial_samples;

		float denom = local.count * local.sample.target;
		float W = (denom > 0 ? local.weight/denom : 0.0f);
		direct = local.sample.value * W;
		if (material.type == Shading::eEmissive)
			direct += material.emission;
	}


	// Also compute indirect lighting
	// TODO: method...
	Shading out;
	float3 wi;
	float pdf;

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
		// trace <eRegular> (x, wi, i0, i1);
		trace(
			parameters.traversable, 1,
			x, wi, i0, i1
		);

		indirect = rp->value;
	}

	// Update the value
	rp->value = direct;
	if (pdf > 0)
		rp->value += T * indirect;
	
	rp->position = make_float4(x, 1);
	rp->normal = n;
	rp->albedo = material.diffuse;
	rp->wi = wi;
}

// Other shaders...
extern "C" __global__ void __closesthit__shadow() {}

// Miss kernel
extern "C" __global__ void __miss__ms()
{
	LOAD_RAYPACKET();

	// Get direction
	const float3 ray_direction = optixGetWorldRayDirection();

	float u = atan2(ray_direction.x, ray_direction.z)/(2.0f * M_PI) + 0.5f;
	float v = asin(ray_direction.y)/M_PI + 0.5f;

	float4 c = make_float4(0);
	if (parameters.envmap != 0)
		c = tex2D <float4> (parameters.envmap, u, v);

	rp->value = make_float3(c);
	rp->wi = ray_direction;
	rp->miss_depth = rp->depth;
}

extern "C" __global__ void __miss__shadow()
{
	unsigned int i0 = optixGetPayload_0();
	unsigned int i1 = optixGetPayload_1();
	bool *vis = unpack_pointer <bool> (i0, i1);
	*vis = false;
}

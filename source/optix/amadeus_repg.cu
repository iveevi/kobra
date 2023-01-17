#include "../../include/amadeus/repg.cuh"
#include "amadeus_common.cuh"

#define MAX_DEPTH 2

using kobra::amadeus::Reservoir;
using kobra::amadeus::Sample;

extern "C"
{
	__constant__ kobra::amadeus::RePG_Parameters parameters;
}

// Ray packet data
struct RayPacket {
	float3	value;

	float4	position;
	float3	normal;
	float3	albedo;

	float3	wi;
	float3	seed;

	float	ior;
	
	int	depth;
	int	index;
};

// TODO: smaller packet for indirect ray stage

static KCUDA_INLINE KCUDA_HOST_DEVICE
void make_ray(uint3 idx,
		float3 &origin,
		float3 &direction,
		float3 &seed)
{
	const float3 U = to_f3(parameters.camera.ax_u);
	const float3 V = to_f3(parameters.camera.ax_v);
	const float3 W = to_f3(parameters.camera.ax_w);
	
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

	origin = to_f3(parameters.camera.center);
	direction = normalize(d.x * U + d.y * V + W);
}

// Accumulatoin helper
template <class T>
__forceinline__ __device__
void accumulate(T &dst, T sample)
{
	if (parameters.accumulate) {
		T prev = dst;
		float count = parameters.samples;
		dst = (prev * count + sample)/(count + 1);
	} else {
		dst = sample;
	}
}

// Ray generation kernel
extern "C" __global__ void __raygen__()
{
	// Get the launch index
	const uint3 idx = optixGetLaunchIndex();

	// Index to store and read the pixel
	const int index = idx.x + idx.y * parameters.resolution.x;

	// Prepare the ray packet
	RayPacket rp {
		.value = make_float3(0.0f),
		.position = make_float4(0),
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

	// Trace the ray
	trace(parameters.traversable, 0, 1, origin, direction, i0, i1);

	// Finally, store the result
	glm::vec4 color = {rp.value.x, rp.value.y, rp.value.z, 1.0f};
	glm::vec3 normal = {rp.normal.x, rp.normal.y, rp.normal.z};
	glm::vec3 albedo = {rp.albedo.x, rp.albedo.y, rp.albedo.z};
	glm::vec3 position = {rp.position.x, rp.position.y, rp.position.z};

	// Check for NaNs
	if (isnan(color.x) || isnan(color.y) || isnan(color.z))
		color = {1, 0, 1, 1};

	// Accumulate and store necessary data
	auto &buffers = parameters.buffers;
	accumulate(buffers.color[index], color);
	accumulate(buffers.normal[index], normal);
	accumulate(buffers.albedo[index], albedo);
	buffers.position[index] = position;
}

// Closest hit kernel
union IEEE {
	struct {
		unsigned int mantissa : 23;
		unsigned int exponent : 8;
		unsigned int sign : 1;
	} parts;
	float value;
};

extern "C" __global__ void __closesthit__()
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
	
	auto &lights = parameters.lights;

	LightingContext lc {
		parameters.traversable,
		lights.quad_lights,
		lights.tri_lights,
		lights.quad_count,
		lights.tri_count,
		parameters.has_environment_map,
		parameters.environment_map,
	};

	float3 direct = make_float3(0.0f);

	if (primary) {
		// Resample the light sources
		Reservoir <Sample> &reservoir = parameters.reservoirs[rp->index];
		reservoir.reset();

		for (int i = 0; i < 8; i++) {
			// Sample the light sources
			FullLightSample fls = sample_direct(lc, surface_hit, rp->seed);
		
			// Compute lighting
			float3 D = fls.point - surface_hit.x;
			float d = length(D);

			float3 Li = direct_unoccluded(surface_hit, fls, D/d, d);

			// Resampling
			// TODO: common target function...
			float target = Li.x + Li.y + Li.z;
			float pdf = fls.pdf;

			reservoir.update(
				Sample {
					.Le = fls.Le,
					.normal = fls.normal,
					.point = fls.point,
					.type = fls.type,
				},
				(pdf > 0) ? target/pdf : 0
			);
		}

		// Compute direct lighting
		Sample sample = reservoir.data;

		float3 D = sample.point - surface_hit.x;
		float d = length(D);

		float3 Li = direct_occluded(
			parameters.traversable, surface_hit,
			sample.Le,
			sample.normal,
			sample.type,
			D/d, d
		);

		float target = Li.x + Li.y + Li.z;
		reservoir.resample(target);

		// TODO: visibility reuse
		bool occluded = is_occluded(lc.handle, surface_hit.x, D/d, d);
		reservoir.W *= 1.0f - occluded;

		// Save material
		parameters.materials[rp->index] = material;

		// Compute direct lighting
		direct = Li * reservoir.W;
	} else {
		direct = material.emission + Ld(lc, surface_hit, rp->seed);
	}

	//--------------------------------------------------------------------------//

	/* Compute spatial hash
	float3 voxel_size = {1.0f, 1.0f, 1.0f};
	float3 voxel_index = floor(x/voxel_size);
	// TODO: cast to uint3 and then back to flaot3
	// to keep deterministic results

	// Take the exponent bits of each index
	IEEE f0 = {.value = voxel_index.x};
	IEEE f1 = {.value = voxel_index.y};
	IEEE f2 = {.value = voxel_index.z};

	// Regular and interleaved hash
	uint16_t exp1 = f0.parts.exponent;
	uint16_t exp2 = f1.parts.exponent;
	uint16_t exp3 = f2.parts.exponent;

	uint16_t hash = exp1 | (exp2 << 5) | (exp3 << 10);

	// Interleaved only for visualisation
	uint16_t interleaved = (exp1 & 0x1) | ((exp2 & 0x1) << 1) | ((exp3 & 0x1) << 2)
		| ((exp1 & 0x2) << 2) | ((exp2 & 0x2) << 3) | ((exp3 & 0x2) << 4)
		| ((exp1 & 0x4) << 4) | ((exp2 & 0x4) << 5) | ((exp3 & 0x4) << 6)
		| ((exp1 & 0x8) << 6) | ((exp2 & 0x8) << 7) | ((exp3 & 0x8) << 8)
		| ((exp1 & 0x10) << 8) | ((exp2 & 0x10) << 9) | ((exp3 & 0x10) << 10);

	// TODO: plus one sign bit? on y-axis?

	// Color table for false color visualization
	float3 colors[] = {
		{0.5, 0.61, 0.61},
		{0.19, 0.29, 0.65},
		{0.48, 0.22, 0.97},
		{0.95, 0.38, 0.92},
		{0.94, 0.3, 0.43},
		{0.12, 1, 0.51},
		{0.93, 0.92, 0.23},
		{0.93, 0.92, 0.23}
	};

	int3 voxel_index3 = make_int3(abs(voxel_index));
	float mod = (voxel_index3.x + voxel_index3.y + voxel_index3.z) % 2;
	rp->value = {mod, mod, mod};

	rp->value = colors[interleaved % 8]; */
	
	// Initial resampling in this kernel, then accumulate in another
	
	// Update values
	rp->value = direct;
	rp->position = make_float4(x, 1);
	rp->normal = n;
	rp->albedo = material.diffuse;
}

// Miss kernels
extern "C" __global__ void __miss__()
{
	LOAD_RAYPACKET();

	// Get direction
	const float3 ray_direction = optixGetWorldRayDirection();

	float u = atan2(ray_direction.x, ray_direction.z)/(2.0f * M_PI) + 0.5f;
	float v = asin(ray_direction.y)/M_PI + 0.5f;

	float4 c = make_float4(0);
	if (parameters.has_environment_map)
		c = tex2D <float4> (parameters.environment_map, u, v);

	rp->value = make_float3(c);
	rp->wi = ray_direction;
}

extern "C" __global__ void __miss__shadow()
{
	unsigned int i0 = optixGetPayload_0();
	unsigned int i1 = optixGetPayload_1();
	bool *vis = unpack_pointer <bool> (i0, i1);
	*vis = false;
}

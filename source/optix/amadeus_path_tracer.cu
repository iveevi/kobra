#include "../../include/amadeus/path_tracer.cuh"
#include "amadeus_common.cuh"

extern "C"
{
	__constant__ kobra::amadeus::PathTracerParameters parameters;
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
};

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
	const uint index = idx.x + idx.y * parameters.resolution.x;

	// Prepare the ray packet
	RayPacket rp {
		.value = make_float3(0.0f),
		.position = make_float4(0),
		.ior = 1.0f,
		.depth = 0,
	};

	// Trace ray and generate contribution
	unsigned int i0, i1;
	pack_pointer(&rp, i0, i1);

	float3 origin;
	float3 direction;

	make_ray(idx, origin, direction, rp.seed);

	// TODO: seed generatoin method
	rp.seed = make_float3(
		sinf(idx.x - idx.y),
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

        printf("Color: %f %f %f\n", color.x, color.y, color.z);

	// Accumulate and store necessary data
	auto &buffers = parameters.buffers;
	accumulate(buffers.color[index], color);
	accumulate(buffers.normal[index], {normal, 0.0f});
	accumulate(buffers.albedo[index], {albedo, 0.0f});
	buffers.position[index] = {position, 0.0f};
}

// Closest hit kernel
extern "C" __global__ void __closesthit__()
{
	// Load all necessary data
	LOAD_RAYPACKET(parameters);
	LOAD_INTERSECTION_DATA(parameters);

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

	float3 direct = material.emission + Ld(lc, surface_hit, rp->seed);

	// Generate new ray
	Shading out;
	float3 wi;
	float pdf;

	float3 f = eval(surface_hit, wi, pdf, out, rp->seed);


	// Get threshold value for current ray
	float3 T = f * abs(dot(wi, n))/pdf;

	// Update for next ray
	// TODO: boolean member for toggling russian roulette
	rp->ior = material.refraction;
	rp->depth++;

	// Trace the next ray
	float3 indirect = make_float3(0.0f);
	if (pdf > 0) {
		trace(parameters.traversable, 0, 1, x, wi, i0, i1);
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

// Miss kernels
extern "C" __global__ void __miss__()
{
	LOAD_RAYPACKET(parameters);

	// Get direction
	const float3 ray_direction = optixGetWorldRayDirection();

	float u = atan2f(ray_direction.x, ray_direction.z)/(2.0f * M_PI) + 0.5f;
	float v = asinf(ray_direction.y)/M_PI + 0.5f;

	float4 c = make_float4(0);
	if (parameters.has_environment_map)
		c = tex2D <float4> (parameters.environment_map, u, 1 - v);

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

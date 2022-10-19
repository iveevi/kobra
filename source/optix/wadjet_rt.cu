#include "wadjet_common.cuh"

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

	// Switch on the ray type
	switch (parameters.mode) {
	case eRegular:
		trace <eRegular> (origin, direction, i0, i1);
		break;
	case eReSTIR:
		trace <eReSTIR> (origin, direction, i0, i1);
		break;
	case eVoxel:
		trace <eVoxel> (origin, direction, i0, i1);
		break;
	default:
		break;
	}
		
	// Finally, store the result
	float4 sample = make_float4(rp.value, 1.0f);
	if (parameters.accumulate) {
		float4 prev = parameters.color_buffer[index];
		parameters.color_buffer[index] = (prev * parameters.samples + sample)
			/(parameters.samples + 1);
	} else {
		parameters.color_buffer[index] = sample;
	}
}

// Closest hit kernel
extern "C" __global__ void __closesthit__ch()
{
	// Load all necessary data
	LOAD_RAYPACKET();
	LOAD_INTERSECTION_DATA();

	// TODO: check for light, not just emissive material
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

	// Update for next ray
	// TODO: boolean member for toggling russian roulette
	rp->ior = material.refraction;
	rp->depth++;
	
	// Trace the next ray
	trace <eRegular> (x, wi, i0, i1);

	// Update the value
	rp->value = direct + T * rp->value;
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

	float4 c = make_float4(0);
	if (parameters.envmap != 0)
		c = tex2D <float4> (parameters.envmap, u, v);

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

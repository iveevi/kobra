#include "../../include/amadeus/restir.cuh"
#include "amadeus_common.cuh"

#define MAX_DEPTH 2

using kobra::amadeus::Reservoir;
using kobra::amadeus::Sample;

extern "C"
{
	__constant__ kobra::amadeus::ReSTIR_Parameters parameters;
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

// TODO: Move to common
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
	
	// float xoffset = (fract(seed.x) - 0.5f);
	// float yoffset = (fract(seed.y) - 0.5f);
	
	// TODO: store offsets for each pixel
	float xoffset = -0.5f;
	float yoffset = -0.5f;

	float2 d = 2.0f * make_float2(
		float(idx.x + xoffset)/parameters.resolution.x,
		float(idx.y + yoffset)/parameters.resolution.y
	) - 1.0f;

	origin = to_f3(parameters.camera.center);
	direction = normalize(d.x * U + d.y * V + W);
}

// Accumulatoin helper
template <class T>
KCUDA_INLINE KCUDA_DEVICE
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

// Ray generation kernel for initial tracing pass
extern "C" __global__ void __raygen__initial()
{
	// Get the launch index
	const uint3 idx = optixGetLaunchIndex();

	// Index to store and read the pixel
	const int index = idx.x + idx.y * parameters.resolution.x;

	// Reset current reservoir
	parameters.current[index].reset();

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
	
	// Store color into intermediate buffer
	parameters.intermediate[index] = color;
	parameters.auxiliary[index] = {rp.seed.x, rp.seed.y, rp.seed.z, 1.0f};

	// Accumulate and store necessary data
	auto &buffers = parameters.buffers;
	accumulate(buffers.normal[index], normal);
	accumulate(buffers.albedo[index], albedo);
	buffers.position[index] = position;
}

// Calculate target function for reservoir
KCUDA_INLINE KCUDA_DEVICE
float target(const Reservoir <Sample> &reservoir, const SurfaceHit &hit)
{
	const Sample &sample = reservoir.data;

	float3 D = sample.point - hit.x;
	float d = length(D);

	float3 Li = direct_unoccluded(
		hit, sample.Le, sample.normal,
		sample.type, D/d, d
	);

	return Li.x + Li.y + Li.z;
}

KCUDA_INLINE KCUDA_DEVICE
float occluded_target(const Reservoir <Sample> &reservoir, const SurfaceHit &hit)
{
	const Sample &sample = reservoir.data;

	float3 D = sample.point - hit.x;
	float d = length(D);

	float3 Li = direct_occluded(
		parameters.traversable,
		hit, sample.Le, sample.normal,
		sample.type, D/d, d
	);

	return Li.x + Li.y + Li.z;
}

// Temporal reprojection index
KCUDA_INLINE KCUDA_DEVICE
int reproject(int index, glm::vec3 position)
{
	int pindex = -1;
	if (parameters.samples > 0)
		return index;

	// Project position
	glm::vec4 p = parameters.previous_camera.projection
		* parameters.previous_camera.view
		* glm::vec4(position, 1.0f);

	float u = p.x/p.w;
	float v = p.y/p.w;

	bool in_u_bounds = (u >= -1.0f && u <= 1.0f);
	bool in_v_bounds = (v >= -1.0f && v <= 1.0f);

	if (in_u_bounds && in_v_bounds) {
		u = (u + 1.0f) * 0.5f;
		v = (v + 1.0f) * 0.5f;

		int ix = u * parameters.resolution.x + 0.5;
		int iy = v * parameters.resolution.y + 0.5;

		pindex = iy * parameters.resolution.x + ix;
	}

	return pindex;
}

// Ray generation kernel for temporal reuse pass
extern "C" __global__ void __raygen__temporal()
{
	// Get the launch index
	const uint3 idx = optixGetLaunchIndex();

	// Index to store and read the pixel
	const int index = idx.x + idx.y * parameters.resolution.x;

	// Get current and previous frame reservoirs
	Reservoir <Sample> &current = parameters.current[index];

	// If current reservoir is empty, skip resampling
	float3 direct = make_float3(0.0f);
	
	glm::vec3 position = parameters.buffers.position[index];

	if (current.size() > 0) {
		// Reconstruct surface intersection information
		Material material = parameters.materials[index];
		glm::vec3 normal = parameters.buffers.normal[index];
		glm::vec3 wo = glm::normalize(parameters.camera.center - position);

		SurfaceHit hit {
			.mat = material,
			.n = {normal.x, normal.y, normal.z},
			.wo = {wo.x, wo.y, wo.z},
			.x = {position.x, position.y, position.z},
		};

		// Find reprojected index
		int pindex = reproject(index, position);
		Reservoir <Sample> *previous = (pindex >= 0) ? &parameters.previous[pindex] : nullptr;

		// Merge current and previous frame reservoirs
		Reservoir <Sample> merged(parameters.auxiliary[index]);

		float t_current = target(current, hit);
		merged.update(current.data, t_current * current.W * current.M);

		int N = current.M;
		if (previous && previous->size() > 0) {
			// M-capping
			previous->M = min(previous->M, 200);

			float t_previous = target(*previous, hit);
			merged.update(previous->data, t_previous * previous->W * previous->M);
			N += (t_previous > 0) * previous->M;
		}
		
		// Resample merged reservoir
		merged.M = N;
		float t_merged = target(merged, hit);
		merged.resample(t_merged);
		current = merged;
		
		/* Shading
		Sample sample = current.data;
		float3 point = sample.point;
		float3 D = point - hit.x;
		float d = length(D);

		float3 lighting = direct_occluded(
			parameters.traversable, hit,
			sample.Le,
			sample.normal,
			sample.type,
			D/d, d
		);

		direct = material.emission + lighting * current.W; */
	}

	/* Final shading using current frame reservoir
	glm::vec4 color = {direct.x, direct.y, direct.z, 1.0f};
	color += parameters.intermediate[index];

	auto &buffers = parameters.buffers;
	accumulate(buffers.color[index], color); */
}

// Sampling spatial neighborhood
KCUDA_INLINE KCUDA_DEVICE
int sample_spatial_neighborhood(int index, Seed seed, int radius = 20)
{
	int width = parameters.resolution.x;
	int height = parameters.resolution.y;

	int x = index % width;
	int y = index / width;

	float3 offset = rand_uniform_3f(seed);
	float theta = 2.0f * M_PI * offset.x;
	float r = radius * sqrt(offset.y);

	int x0 = x + r * cos(theta);
	int y0 = y + r * sin(theta);

	x0 = clamp(x0, 0, width - 1);
	y0 = clamp(y0, 0, height - 1);

	return x0 + y0 * width;
}

// Ray generation kernel for final spatial reuse pass
extern "C" __global__ void __raygen__spatial()
{
	// Get the launch index
	const uint3 idx = optixGetLaunchIndex();

	// Index to store and read the pixel
	const int index = idx.x + idx.y * parameters.resolution.x;

	// Reconstruct surface intersection information
	Material material = parameters.materials[index];
	glm::vec3 position = parameters.buffers.position[index];
	glm::vec3 normal = parameters.buffers.normal[index];
	glm::vec3 wo = glm::normalize(parameters.camera.center - position);

	SurfaceHit hit {
		.mat = material,
		.n = {normal.x, normal.y, normal.z},
		.wo = {wo.x, wo.y, wo.z},
		.x = {position.x, position.y, position.z},
	};

	// Get current reservoir
	Reservoir <Sample> current = parameters.current[index];

	// If current reservoir is empty, skip resampling
	float3 direct = make_float3(0.0f);
	if (current.size() > 0) {
		// Merge current and previous frame reservoirs
		Reservoir <Sample> merged(parameters.auxiliary[index]);

		float t_current = target(current, hit);
		merged.update(current.data, t_current * current.W * current.M);

		// Sample spatial neighborhood
		constexpr int SAMPLES = 5;

		SurfaceHit hits[SAMPLES];
		int sizes[SAMPLES];

		int count = 0;
		for (int i = 0; i < SAMPLES; i++) {
			int index0 = sample_spatial_neighborhood(index, merged.seed, 30);
			Reservoir <Sample> neighbor = parameters.current[index0];

			float t_neighbor = target(neighbor, hit);

			if (neighbor.size() > 0)
				merged.update(neighbor.data, t_neighbor * neighbor.W * neighbor.M);

			// Reconstruct neighbor surface intersection information
			Material material0 = parameters.materials[index0];
			glm::vec3 position0 = parameters.buffers.position[index0];
			glm::vec3 normal0 = parameters.buffers.normal[index0];

			hits[i] = {
				.mat = material0,
				.entering = neighbor.size() > 0, // Entering now dictates sample validity
				.n = {normal0.x, normal0.y, normal0.z},
				.wo = {wo.x, wo.y, wo.z},
				.x = {position0.x, position0.y, position0.z},
			};

			sizes[i] = neighbor.size();

			// Add to count
			count += neighbor.M;
		}

		merged.M = current.M + count;

		// Fix bias (Original ReSTIR paper)
		int Z = current.M;
		for (int i = 0; i < SAMPLES; i++) {
			float t_neighbor = (hits[i].entering) ? target(merged, hits[i]) : 0.0f;
			if (t_neighbor > 0.0f)
				Z += sizes[i];
		}

		// Resample merged reservoir
		// TODO: option for bias merged instead
		float t_merged = target(merged, hit);
		float denominator = t_merged * Z;
		merged.W = (denominator > 0.0f) ? merged.w/denominator : 0.0f;
		current = merged;

		// Shading
		Sample sample = current.data;
		float3 point = sample.point;
		float3 D = point - hit.x;
		float d = length(D);

		float3 lighting = direct_occluded(
			parameters.traversable, hit,
			sample.Le,
			sample.normal,
			sample.type,
			D/d, d
		);

		direct = material.emission + lighting * current.W;

		// Save current reservoir to previous frame reservoir
		parameters.previous[index] = current;
	} else {
		// Reset previous frame reservoir
		parameters.previous[index].reset();
	}

	// Final shading using current frame reservoir
	glm::vec4 color = {direct.x, direct.y, direct.z, 1.0f};
	color += parameters.intermediate[index];

	if (glm::any(glm::isnan(color)))
		color = {1, 0, 1, 1};

	auto &buffers = parameters.buffers;
	accumulate(buffers.color[index], color);
}

// Closest hit kernel
extern "C" __global__ void __closesthit__initial()
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
		Reservoir <Sample> &reservoir = parameters.current[rp->index];
		reservoir.reset();

		for (int i = 0; i < 32; i++) {
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
		// bool occluded = is_occluded(lc.handle, surface_hit.x, D/d, d);
		// reservoir.W *= 1.0f - occluded;

		// Save material
		parameters.materials[rp->index] = material;
	} else {
		direct = material.emission + Ld(lc, surface_hit, rp->seed);
	}

	// Generate new ray
	Shading out;
	float3 wi;
	float pdf;

	float3 f = eval(surface_hit, wi, pdf, out, rp->seed);

	// Get threshold value for current ray
	float3 T = (pdf > 0) ? f * abs(dot(wi, n))/pdf : make_float3(0.0f);

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

	// Update values
	rp->value = direct + T * indirect;
	rp->position = make_float4(x, 1);
	rp->normal = n;
	rp->albedo = material.diffuse;
	rp->wi = wi;
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

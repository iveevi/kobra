#include "../../include/asmodeus/wsris_kd_parameters.cuh"
#include "common.cuh"

extern "C"
{
	__constant__ kobra::optix::WorldSpaceKdReservoirsParameters parameters;
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
	if (parameters.accumulate && false) {
		float4 prev = dst;
		int count = parameters.samples;
		dst = (prev * count + sample)/(count + 1);
	} else {
		dst = sample;
	}
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
	trace <eRegular> (
		parameters.traversable, eCount,
		origin, direction, i0, i1
	);

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

	// Stor data
	parameters.position_buffer[index] = rp.position;
}

// Helpers
__forceinline__ __device__
float get(float3 a, int axis)
{
	if (axis == 0) return a.x;
	if (axis == 1) return a.y;
	if (axis == 2) return a.z;
}

// Closest hit kernel
extern "C" __global__ void __closesthit__ch()
{
	LOAD_RAYPACKET();
	LOAD_INTERSECTION_DATA();

	// Check if primary ray
	bool primary = (rp->depth == 0);

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
	
	LightingContext lc {
		parameters.traversable,
		parameters.lights.quads,
		parameters.lights.triangles,
		parameters.lights.quad_count,
		parameters.lights.triangle_count,
		parameters.has_envmap,
		parameters.envmap,
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
	FullLightSample fls = sample_direct(lc, surface_hit, rp->seed);

	// Compute target function (unocculted lighting)
	float3 D = fls.point - surface_hit.x;
	float d = length(D);
	D /= d;

	float3 Li = direct_occluded(
		parameters.traversable,
		surface_hit, fls.Le, fls.normal, fls.type, D, d
	);
		
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
		FullLightSample fls = sample_direct(lc, surface_hit, rp->seed);

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

		int root = 0;
		int depth = 0;

		int lefts = 0;
		int rights = 0;

		float3 pos = surface_hit.x;
		
		while (root >= 0) {
			depth++;
			kd_node = &parameters.kd_tree[root];
			
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

		// Lock and update the reservoir
		int res_idx = kd_node->data;
		
		int *lock = parameters.kd_locks[res_idx];

		// while (atomicCAS(lock, 0, 1) == 0); // Lock

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

		// atomicExch(lock, 0);			// Unlock

		// TODO: two strategies
		//	hierarchical: go up a few levels and then traverse down
		//	pick a random node and traverse down
		const int SPATIAL_SAMPLES = 3;

		// Choose a root node a few level up and randomly
		// traverse the tree to obtain a sample
		const int LEVELS = 10;

		// TODO: try selecting random indices in the tree instead?
		int levels = min(depth, LEVELS);
		while (levels--) {
			kd_node = &parameters.kd_tree[root];

			if (kd_node->parent == -1)
				break;

			root = kd_node->parent;
		}

		int successes = 0;
		for (int i = 0; i < SPATIAL_SAMPLES; i++) {
			int node = root;

			while (true) {
				kd_node = &parameters.kd_tree[node];

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

			// TODO: syncronized pipeline, this one is copied
			// because no lock is used
			LightReservoir rsampled = parameters.kd_reservoirs_prev[res_idx];
			LightSample sample = rsampled.sample;

			// Compute value and target
			D = sample.point - surface_hit.x;
			d = length(D);
			D /= d;

			Li = direct_occluded(parameters.traversable, surface_hit, sample.value,
					sample.normal, sample.type, D, d);

			float denom = rsampled.count * sample.target;
			float W = (denom > 0.0f) ? rsampled.weight/denom : 0.0f;

			// Insert into spatial reservoir
			target = Li.x + Li.y + Li.z; // Luminance
			w = target * W * rsampled.count; // TODO: compute without doing repeated work

			int pcount = spatial.count;
			reservoir_update(&spatial, LightSample {
				.value = Li,
				.target = target,
				.type = sample.type,
				.index = sample.index
			}, w, rp->seed);

			// spatial.count = pcount + (target > 0.0f ? reservoir->count : 0);
			spatial.count = pcount + rsampled.count;
			successes += (target > 0.0f);

			/* Also insert into temporal reservoir
			Li = direct_unoccluded(surface_hit, sample.value, sample.normal, sample.type, D, d);

			target = Li.x + Li.y + Li.z; // Luminance
			denom = rsampled.count * target;
			W = (denom > 0.0f) ? rsampled.weight/denom : 0.0f;
			// assert(!isnan(W));

			w = target * W * rsampled.count;
			// assert(!isnan(w));

			reservoir->weight += w;
			// assert(!isnan(reservoir->weight));

			float p = w/reservoir->weight;
			float eta = rand_uniform(rp->seed);

			if (eta < p) {
				reservoir->sample = LightSample {
					.value = Li,
					.point = sample.point,
					.normal = sample.normal,
					.target = target,
					.type = sample.type,
					.index = sample.index
				};
			}

			reservoir->count += rsampled.count; */
		}
	}

	// Final direct lighting result
	float denom = spatial.count * spatial.sample.target;
	float W = (denom > 0) ? spatial.weight/denom : 0.0f;
	// assert(!isnan(W));

	direct = spatial.sample.value * W;
	
	// Add emission as well
	if (material.type == Shading::eEmissive)
		direct += material.emission;

	// Also compute indirect lighting
	Shading out;
	float3 wi;

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
		trace(parameters.traversable, eCount, x, wi, i0, i1);
		indirect = rp->value;
	}

	// Update the value
	rp->value = direct;
	if (pdf > 0)
		rp->value += T * indirect;

	rp->position = make_float4(x, 1);
}

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

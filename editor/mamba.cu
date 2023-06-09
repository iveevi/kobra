// Engine headers
#include "include/cuda/alloc.cuh"

// Local headers
#include "editor/common.hpp"
#include "editor/editor_viewport.cuh"
#include "include/cuda/brdf.cuh"
#include "mamba.cuh"

using namespace kobra;

// TODO: probe.cuh
// Irradiance probe
struct IrradianceProbe {
        constexpr static int size = 6;

        // Layed out using octahedral projection
        float3 values[size * size];
        float pdfs[size * size];
        float depth[size * size];

        float3 normal;
	float3 position;

	constexpr static float sqrt_2 = 1.41421356237f;
	constexpr static float inv_sqrt_2 = 1.0f/sqrt_2;

	// NOTE: Direction is expected to be in local space

	// Octahedral projection
	__forceinline__ __device__
	float2 to_oct(float3 v) const {
		float3 r = v / (abs(v.x) + abs(v.y) + abs(v.z));
		// float2 s = make_float2(r.x + r.y, r.x - r.y);
		float2 s = make_float2(r);
		return (s + 1.0f) / 2.0f;
	}

	// Disk projection
	__forceinline__ __device__
	float2 to_disk(float3 v) const {
		float2 s = make_float2(v.x, v.y);
		float theta = atan2f(s.y, s.x);
		if (theta < 0.0f)
			theta += 2.0f * M_PI;
		float rp = pow(s.x * s.x + s.y * s.y, 0.3f);
		return make_float2(theta/(2.0f * M_PI), rp);
	}
};

// A single level-node of the irradiance probe LUT
struct IrradianceProbeLUT {
       constexpr static int MAX_REFS = (1 << 15) - 1;

	// Cell properties
        float resolution;
	float size;
	float3 center;

        int32_t level = 0;
	int32_t counter = 0;

	int32_t refs[MAX_REFS];
	uint32_t hashes[MAX_REFS];
	float3 positions[MAX_REFS];

	static void alloc(IrradianceProbeLUT *lut) {
		lut->level = 0;
		lut->size = 10.0f;
		lut->resolution = lut->size/25.0f;
		lut->center = make_float3(0.0f, 0.0f, 0.0f);

		for (int i = 0; i < MAX_REFS; i++) {
			lut->refs[i] = -1;
			lut->hashes[i] = 0xFFFFFFFF;
		}
	}

	__forceinline__ __device__
	static void alloc(IrradianceProbeLUT *lut, int level, float size, float3 center) {
		lut->level = level;
		lut->size = size;
		lut->resolution = size/powf(MAX_REFS, 1.0f/3.0f);
		lut->center = center;
	}

	__forceinline__ __device__
	bool contains(float3 x, bool neighbor = false) {
		float3 min = center - make_float3(size/2.0f + neighbor * resolution);
		float3 max = center + make_float3(size/2.0f + neighbor * resolution);
		return (x.x >= min.x && x.y >= min.y && x.z >= min.z)
			&& (x.x <= max.x && x.y <= max.y && x.z <= max.z);
	}

        __forceinline__ __device__
        uint32_t hash(float3 x, int32_t dx = 0, int32_t dy = 0, int32_t dz = 0) {
		float cdx = x.x - center.x + size/2.0f;
		float cdy = x.y - center.y + size/2.0f;
		float cdz = x.z - center.z + size/2.0f;

		int32_t ix = (int32_t) ((cdx / resolution) + dx);
		int32_t iy = (int32_t) ((cdy / resolution) + dy);
		int32_t iz = (int32_t) ((cdz / resolution) + dz);

		int32_t h = (ix & 0x7FF) | ((iy & 0x7FF) << 11) | ((iz & 0x7FF) << 22);

		// Shuffle bits
		h ^= (h >> 11) ^ (h >> 22);
		h ^= (h << 7) & 0x9D2C5680;
		h ^= (h << 15) & 0xEFC60000;

		return *((uint32_t *) &h);
        }

	// TODO: radius (e.g. multiplication factor)
	__forceinline__ __device__
	uint32_t neighboring_hash(float3 x, int8_t ni) {
		// Options are ni from 0 to 26, e.g. sides of a cube
		int32_t dx = ni % 3 - 1;
		int32_t dy = (ni / 3) % 3 - 1;
		int32_t dz = ni / 9 - 1;

		return hash(x, dx, dy, dz);
	}

        // TODO: try double hashing instead?
        // NOTE: linear probing would be more cache friendly, but double hashing
        // leads to better distribution
        static constexpr int32_t MAX_TRIES = (1 << 2);

	// TODO: analyze how well it is able to cover surfaces with higher number of tries..
	__forceinline__ __device__
	uint32_t double_hash(uint32_t h, int32_t i) {
		// Double hashing (shuffle again)
		// int32_t oh = h;
		//
		// h = (h ^ 61) ^ (h >> 16);
		// h = h + (h << 3);
		// h = h ^ (h >> 4);
		// h = h * 0x27d4eb2d;
		// h = h ^ (h >> 15);
		//
		// return oh + (i + 1) * h;
		return h + (i * i);
	}

	// Allocating new reference
	// TODO: cuckoo hashing; if there is an infinite loop, then kick out the oldest (LRU)
	__forceinline__ __device__
	int32_t request(float3 x) {
		// TODO: return index if already allocated

		uint32_t h = hash(x);
		int32_t success = -1;

		int32_t i = 0;
		int32_t old = INT_MAX;

		while (i < MAX_TRIES) {
			int32_t j = double_hash(h, i) % MAX_REFS;
			old = atomicCAS(&refs[j], -1, h);
			if (old == -1) {
				success = j;
				break;
			}

			i++;
		}

		if (old == -1) {
			hashes[success] = h;
			positions[success] = x;
			atomicAdd(&counter, 1);
		}

		return success;
	}

	// Find the nearest reference to the given position
	// TODO: return the distance to the nearest reference
	__forceinline__ __device__
	int32_t lookup(float3 x, int32_t ni) {
		if (!contains(x, ni != -1))
			return -1;

		uint32_t h = (ni == -1) ? hash(x) : neighboring_hash(x, ni);

		int32_t i = 0;
		int32_t index = -1;

		float closest = FLT_MAX;
		while (i < MAX_TRIES) {
			int32_t j = double_hash(h, i) % MAX_REFS;
			int32_t ref = refs[j];
			if (ref == -1)
				break;

			if (hashes[j] == h) {
				float dist = length(positions[j] - x);
				if (dist < closest) {
					closest = dist;
					index = j;
				}
			}

			i++;
		}

		return index;
	}
};

// Table of all irradiance probes and LUTs
struct IrradianceProbeTable {
        constexpr static int MAX_PROBES = 1 << 20;
	constexpr static int MAX_LUTS = 1 << 8;

        IrradianceProbe *probes = nullptr;
        int32_t counter = 0;

	IrradianceProbeLUT *luts = nullptr;
	int32_t lut_counter = 0;

	IrradianceProbeTable() {
		probes = new IrradianceProbe[MAX_PROBES];

		// Allocate top level LUT upon front
		luts = new IrradianceProbeLUT[MAX_LUTS];
		IrradianceProbeLUT::alloc(&luts[0]);
		lut_counter++;
	}

	IrradianceProbeTable device_copy() const {
		IrradianceProbeTable table;

		CUDA_CHECK(cudaMalloc(&table.probes, sizeof(IrradianceProbe) * MAX_PROBES));
		CUDA_CHECK(cudaMemcpy(table.probes, probes, sizeof(IrradianceProbe) * MAX_PROBES, cudaMemcpyHostToDevice));
		table.counter = counter;

		CUDA_CHECK(cudaMalloc(&table.luts, sizeof(IrradianceProbeLUT) * MAX_LUTS));
		CUDA_CHECK(cudaMemcpy(table.luts, luts, sizeof(IrradianceProbeLUT) * MAX_LUTS, cudaMemcpyHostToDevice));
		table.lut_counter = lut_counter;

		return table;
	}

	__forceinline__ __device__
	int32_t next() {
		int32_t index = atomicAdd(&counter, 1);
		if (index >= MAX_PROBES)
			return -1;
		return index;
	}

	__forceinline__ __device__
	int32_t lut_next() {
		int32_t index = atomicAdd(&lut_counter, 1);
		if (index >= MAX_LUTS)
			return -1;
		return index;
	}

	__forceinline__ __device__
	void clear() {
		counter = 0;
		lut_counter = 0;
	}
};

inline IrradianceProbeTable *alloc_table()
{
	IrradianceProbeTable table;
	IrradianceProbeTable proxy_table = table.device_copy();
	IrradianceProbeTable *device_table;

	CUDA_CHECK(cudaMalloc(&device_table, sizeof(IrradianceProbeTable)));
	CUDA_CHECK(cudaMemcpy(device_table, &proxy_table, sizeof(IrradianceProbeTable), cudaMemcpyHostToDevice));

	return device_table;
}

// OptiX compilation options
static constexpr OptixPipelineCompileOptions pipeline_compile_options = {
	.usesMotionBlur = false,
	.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_LEVEL_INSTANCING,
	.numPayloadValues = 2,
	.numAttributeValues = 0,
	.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE,
	// .exceptionFlags = OPTIX_EXCEPTION_FLAG_DEBUG,
	.pipelineLaunchParamsVariableName = "info",
	.usesPrimitiveTypeFlags = (unsigned int) OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE,
};

static constexpr OptixModuleCompileOptions module_compile_options = {
	.optLevel = OPTIX_COMPILE_OPTIMIZATION_LEVEL_3,
	.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_NONE,
	// .optLevel = OPTIX_COMPILE_OPTIMIZATION_LEVEL_0,
	// .debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_FULL,
};

static constexpr OptixPipelineLinkOptions pipeline_link_options = {
	.maxTraceDepth = 1,
};

// Constructor
Mamba::Mamba(const OptixDeviceContext &context)
{
        static constexpr const char OPTIX_PTX_FILE[] = "bin/ptx/mamba_shader.o";

        // Load module
        module = optix::load_optix_module(context, OPTIX_PTX_FILE, pipeline_compile_options, module_compile_options);

        // Load programs
        OptixProgramGroupOptions program_options = {};

        // Descriptions of all the programs
        std::vector <OptixProgramGroupDesc> program_descs = {
                OPTIX_DESC_RAYGEN (module, "__raygen__direct_primary"),
                OPTIX_DESC_RAYGEN (module, "__raygen__temporal_reuse"),
                OPTIX_DESC_RAYGEN (module, "__raygen__spatial_reuse"),
                OPTIX_DESC_HIT    (module, "__closesthit__"),
                OPTIX_DESC_MISS   (module, "__miss__"),
        };

        // Corresponding program groups
        std::vector <OptixProgramGroup *> program_groups = {
                &raygen_direct_primary,
                &raygen_direct_temporal,
                &raygen_direct_spatial,
                &closest_hit,
                &miss,
        };

        optix::load_program_groups(
                context,
                program_descs,
                program_options,
                program_groups
        );

        direct_ppl = optix::link_optix_pipeline(context, {
                raygen_direct_primary,
                raygen_direct_temporal,
                raygen_direct_spatial,
                closest_hit,
                miss,
        }, pipeline_compile_options, pipeline_link_options);

        // Create shader binding table
        direct_initial_sbt = {};

        // Ray generation
        CUdeviceptr dev_direct_initial_raygen_record;

        optix::Record <void> direct_initial_raygen_record;
        OPTIX_CHECK(optixSbtRecordPackHeader(raygen_direct_primary, &direct_initial_raygen_record));
        CUDA_CHECK(cudaMalloc((void **) &dev_direct_initial_raygen_record, sizeof(optix::Record <void>)));
        CUDA_CHECK(cudaMemcpy((void *) dev_direct_initial_raygen_record, &direct_initial_raygen_record, sizeof(optix::Record <void>), cudaMemcpyHostToDevice));

        optix::Record <void> direct_temporal_raygen_record;
        OPTIX_CHECK(optixSbtRecordPackHeader(raygen_direct_temporal, &direct_temporal_raygen_record));
        CUDA_CHECK(cudaMalloc((void **) &dev_direct_temporal_raygen_record, sizeof(optix::Record <void>)));
        CUDA_CHECK(cudaMemcpy((void *) dev_direct_temporal_raygen_record, &direct_temporal_raygen_record, sizeof(optix::Record <void>), cudaMemcpyHostToDevice));

        optix::Record <void> direct_spatial_raygen_record;
        OPTIX_CHECK(optixSbtRecordPackHeader(raygen_direct_spatial, &direct_spatial_raygen_record));
        CUDA_CHECK(cudaMalloc((void **) &dev_direct_spatial_raygen_record, sizeof(optix::Record <void>)));
        CUDA_CHECK(cudaMemcpy((void *) dev_direct_spatial_raygen_record, &direct_spatial_raygen_record, sizeof(optix::Record <void>), cudaMemcpyHostToDevice));

        // Miss
        CUdeviceptr dev_miss_record;

        optix::Record <void> miss_record;
        OPTIX_CHECK(optixSbtRecordPackHeader(miss, &miss_record));
        CUDA_CHECK(cudaMalloc((void **) &dev_miss_record, sizeof(optix::Record <void>)));
        CUDA_CHECK(cudaMemcpy((void *) dev_miss_record, &miss_record, sizeof(optix::Record <void>), cudaMemcpyHostToDevice));

        direct_initial_sbt.raygenRecord = dev_direct_initial_raygen_record;
        direct_initial_sbt.missRecordBase = dev_miss_record;
        direct_initial_sbt.missRecordStrideInBytes = sizeof(optix::Record <void>);
        direct_initial_sbt.missRecordCount = 1;
        direct_initial_sbt.hitgroupRecordBase = 0;
        direct_initial_sbt.hitgroupRecordStrideInBytes = 0;
        direct_initial_sbt.hitgroupRecordCount = 0;

        // std::memcpy(&direct_temporal_sbt, &direct_initial_sbt, sizeof(OptixShaderBindingTable));
        // direct_temporal_sbt.raygenRecord = dev_direct_temporal_raygen_record;

        // Setup parameters
        launch_info = {};
        launch_info.io = optix_io_create();
        launch_info.direct.reservoirs = 0;
        launch_info.direct.Le = 0;
	launch_info.indirect.probes = alloc_table();

        // Allocate device pointers
        // TODO: for probes, crete lazily on first use
        dev_launch_info = cuda::alloc(sizeof(MambaLaunchInfo));
}

// Final gather functions
__global__
void sobel_normal(cudaSurfaceObject_t normal_surface, float *sobel_target, vk::Extent2D extent)
{
        int index = threadIdx.x + blockIdx.x * blockDim.x;
        int x = index % extent.width;
        int y = index / extent.width;
        y = extent.height - (y + 1);

        if (x >= extent.width || y >= extent.height)
                return;

        float4 raw_normal;
        surf2Dread(&raw_normal, normal_surface, x * sizeof(float4), y);
        float3 normal = make_float3(raw_normal);

        sobel_target[index] = 0;
        for (int i = -1; i <= 1; i++) {
                for (int j = -1; j <= 1; j++) {
                        int x2 = x + i;
                        int y2 = y + j;

                        if (x2 < 0 || x2 >= extent.width || y2 < 0 || y2 >= extent.height)
                                continue;

                        float4 raw_normal2;
                        surf2Dread(&raw_normal2, normal_surface, x2 * sizeof(float4), y2);

                        float3 normal2 = make_float3(raw_normal2);
                        float3 diff = normal - normal2;
                        sobel_target[index] += length(diff);
                }
        }
}

struct ProbeAllocationInfo {
        cudaSurfaceObject_t index_surface;
        cudaSurfaceObject_t position_surface;
	cudaSurfaceObject_t normal_surface;
        float *sobel;

	IrradianceProbeTable *probes;

        vk::Extent2D extent;
};

__global__
void probe_allocation(ProbeAllocationInfo info)
{
        // TODO: use block parallelization
        // i.e. sync threads in block, then use atomicAdd

        int index = threadIdx.x + blockIdx.x * blockDim.x;

        // Each thread goes through 16x16 block to allocate probes
        // int x = (index % info.extent.width) / 16;
        // int y = (index / info.extent.width) / 16;
        int x = index % (info.extent.width/16);
        int y = index / (info.extent.width/16);

        // if (x >= (info.extent.width / 16 + 1) || y >= (info.extent.height / 16 + 1))
        //         return;
        if (x >= (info.extent.width/16) || y >= (info.extent.height/16))
                return;

        int x2 = x * 16;
        int y2 = y * 16;

        // TODO: shared memory: project existing probes onto 16x16 block

        bool covered[16 * 16] { false }; // TODO: pack into int_16 array

        // TODO: need to check if block has invalid regions

        // For now lazily allocate at the central position of the block
        int x3 = x2 + 8;
        int y3 = y2 + 8;
        y3 = info.extent.height - (y3 + 1);

        int32_t raw_index;
        surf2Dread(&raw_index, info.index_surface, x3 * sizeof(int32_t), y3);
        if (raw_index == -1)
                return;

        float4 raw_position;
        surf2Dread(&raw_position, info.position_surface, x3 * sizeof(float4), y3);
        float3 position = make_float3(raw_position);

	float4 raw_normal;
	surf2Dread(&raw_normal, info.normal_surface, x3 * sizeof(float4), y3);
	float3 normal = make_float3(raw_normal);

        float radius = 0.01f;

	// Allocate probe on the table
	// TODO: iterate through LUTs

	IrradianceProbeLUT *lut = &info.probes->luts[0];
	if (!lut->contains(position))
		return;

	uint32_t hash = lut->hash(position);
	int32_t success = lut->request(position);
	if (success == -1)
		return;

	int32_t probe_index = info.probes->next();
	if (probe_index == -1) {
		// TODO: does this need to be atomic?
		// lut->refs[success] = -1;
		return;
	}

	// printf("(%f, %f, %f) hash: %u, success: %d, probe %d\n", position.x, position.y, position.z, hash, success, probe_index);

	// IrradianceProbe *probe = &info.probes->probes[probe_index];
	//
	// *probe = {};
	// probe->position = position;
	// probe->normal = normal;

	info.probes->probes[probe_index].position = position;
	info.probes->probes[probe_index].normal = normal;
	lut->refs[success] = probe_index;
}

struct FinalGather {
        float4 *color;
        float3 *direct;
        float *sobel;

        cudaSurfaceObject_t position_surface;

	IrradianceProbeTable *table;
	bool brute_force;

        vk::Extent2D extent;
};

__global__
void final_gather(FinalGather info)
{
        // TODO: return sky ray if invalid/no hit
        int index = threadIdx.x + blockIdx.x * blockDim.x;

        int x = index % info.extent.width;
        int y = index / info.extent.width;
        if (x >= info.extent.width || y >= info.extent.height)
                return;

	// Get position
	float4 raw_position;
	surf2Dread(&raw_position, info.position_surface, x * sizeof(float4), (info.extent.height - (y + 1)));
	float3 position = make_float3(raw_position);

        // float4 &color = info.color[index];

        bool border = (info.sobel[index] > 0.7f);
        float3 color = 1 * info.direct[index] + 0 * border;
        // color = make_float4(1 * info.direct[index] + 0 * border, 1.0f);

        int x_16 = x / 16;
        int y_16 = y / 16;

	IrradianceProbeLUT *lut = &info.table->luts[0];

	// TODO: flag to render probes
	float3 covered_color = make_float3(0);

	float closest = FLT_MAX;
	int32_t closest_index = -1;

	int32_t result = lut->lookup(position, -1);
	if (result != -1) {
		int32_t probe_index = lut->refs[result];
		IrradianceProbe *probe = &info.table->probes[probe_index];
		float3 diff = position - probe->position;
		float dist = length(diff);

		if (dist < closest) {
			closest = dist;
			closest_index = probe_index;
		}
	}

	// Also consider neighboring probes
	for (int ni = 0; ni < 27; ni++) {
		int32_t result = lut->lookup(position, ni);
		if (result == -1)
			continue;

		int32_t probe_index = lut->refs[result];
		IrradianceProbe *probe = &info.table->probes[probe_index];
		float3 diff = position - probe->position;
		float dist = length(diff);

		if (dist < closest) {
			closest = dist;
			closest_index = probe_index;
		}
	}

	if (closest_index != -1) {
		int32_t probe_index = closest_index;
		IrradianceProbe *probe = &info.table->probes[probe_index];
		float3 diff = position - probe->position;

		cuda::ONB onb = cuda::ONB::from_normal(probe->normal);
		float3 local = onb.inv_local(diff)/lut->resolution;
		local.z = 0;
		float dist = length(local);

		if (dist < 1) {
			float3 wi = make_float3(local.x, local.y, sqrtf(1 - local.x * local.x - local.y * local.y));
			wi = normalize(wi);

			float2 s = probe->to_disk(wi);

			constexpr int32_t size = IrradianceProbe::size;
			int32_t ix = s.x * size;
			int32_t iy = s.y * size;
			int32_t index = ix + iy * size;

			covered_color = make_float3((ix + iy) % 2);
		}
	}

        info.color[index] = make_float4(0.1 * color + covered_color, 1.0f);
	// if (index == 0)
	// 	printf("LUT size is %d\n", info.table->luts[0].counter);
}

// Rendering function
void Mamba::render(EditorViewport *ev,
                const RenderInfo &render_info,
                const std::vector <Entity> &entities,
                const MaterialDaemon *md)
{
        const Camera &camera = render_info.camera;
        const Transform &camera_transform = render_info.camera_transform;
        const vk::Extent2D &extent = ev->extent;

        // TODO: pass common rtx instead of ev..

        // Handle resizing
        if (resize_queue.size() > 0) {
                vk::Extent2D new_extent = resize_queue.back();
                resize_queue = {};

                // Direct lighting buffers
                if (launch_info.direct.reservoirs != 0)
                        CUDA_CHECK(cudaFree((void *) launch_info.direct.reservoirs));

                if (launch_info.direct.previous != 0)
                        CUDA_CHECK(cudaFree((void *) launch_info.direct.previous));

                if (launch_info.direct.Le != 0)
                        CUDA_CHECK(cudaFree((void *) launch_info.direct.Le));

                // TODO: reservoirs should be staggered (size + 1 on each side)
                // so that we can average corners for each pixel
                int size = new_extent.width * new_extent.height;
                launch_info.direct.reservoirs = cuda::alloc <Reservoir <LightInfo>> (size);
                launch_info.direct.previous = cuda::alloc <Reservoir <LightInfo>> (size);
                launch_info.direct.Le = cuda::alloc <float3> (size);

                // Indirect lighting buffers
                if (launch_info.indirect.sobel != 0)
                        CUDA_CHECK(cudaFree((void *) launch_info.indirect.sobel));

                launch_info.indirect.sobel = cuda::alloc <float> (size);
        }

        // Configure launch parameters
        launch_info.time = ev->common_rtx.timer.elapsed_start();
        launch_info.dirty = render_info.camera_transform_dirty;
        launch_info.reset = ev->render_state.mamba_reset
                        | ev->common_rtx.material_reset
                        | ev->common_rtx.transform_reset
                        | manual_reset;

        if (launch_info.reset)
                manual_reset = false;
        ev->render_state.mamba_reset = false;

        launch_info.samples++;
        if (launch_info.dirty)
                launch_info.samples = 0;

        // Configure camera axis
        auto uvw = uvw_frame(camera, camera_transform);

        launch_info.camera.U = cuda::to_f3(uvw.u);
        launch_info.camera.V = cuda::to_f3(uvw.v);
        launch_info.camera.W = cuda::to_f3(uvw.w);
        launch_info.camera.origin = cuda::to_f3(render_info.camera_transform.position);
        launch_info.camera.resolution = { extent.width, extent.height };

        // Configure textures and other buffers
        launch_info.position = ev->framebuffer_images->cu_position_surface;
        launch_info.normal = ev->framebuffer_images->cu_normal_surface;
        launch_info.uv = ev->framebuffer_images->cu_uv_surface;
        launch_info.index = ev->framebuffer_images->cu_material_index_surface;

        launch_info.materials = (cuda::_material *) ev->common_rtx.dev_materials;

        launch_info.area.lights = (AreaLight *) ev->common_rtx.dev_lights;
        launch_info.area.count = ev->common_rtx.lights.size();
        launch_info.area.triangle_count = ev->common_rtx.triangle_count;

        launch_info.sky.texture = ev->environment_map.texture;
        launch_info.sky.enabled = ev->environment_map.valid;

        launch_info.options.temporal = temporal_reuse;
        launch_info.options.spatial = spatial_reuse;

        // Copy parameters and launch
        cuda::copy(dev_launch_info, &launch_info, 1, cudaMemcpyHostToDevice);

        // TODO: parallelize by having one stage for direct, one for indirect
        // (and then for spatil reuse in restir we paralleize with indirect
        // filtering...)
        OPTIX_CHECK(
                optixLaunch(direct_ppl, 0,
                        dev_launch_info,
                        sizeof(MambaLaunchInfo),
                        &direct_initial_sbt, extent.width, extent.height, 1
                )
        );

        CUDA_SYNC_CHECK();

        if (temporal_reuse) {
                std::memcpy(&direct_temporal_sbt, &direct_initial_sbt, sizeof(OptixShaderBindingTable));
                direct_temporal_sbt.raygenRecord = dev_direct_temporal_raygen_record;

                OPTIX_CHECK(
                        optixLaunch(direct_ppl, 0,
                                dev_launch_info,
                                sizeof(MambaLaunchInfo),
                                &direct_temporal_sbt, extent.width, extent.height, 1
                        )
                );

                CUDA_SYNC_CHECK();
        }

        if (spatial_reuse) {
                std::memcpy(&direct_spatial_sbt, &direct_initial_sbt, sizeof(OptixShaderBindingTable));
                direct_spatial_sbt.raygenRecord = dev_direct_spatial_raygen_record;

                OPTIX_CHECK(
                        optixLaunch(direct_ppl, 0,
                                dev_launch_info,
                                sizeof(MambaLaunchInfo),
                                &direct_spatial_sbt, extent.width, extent.height, 1
                        )
                );

                CUDA_SYNC_CHECK();
        }

        // TODO: more advanced parallelization
        uint block_size = 256;
        uint blocks = (extent.width * extent.height + 255) / 256;

        // Final gather
        sobel_normal <<< blocks, block_size >>> (launch_info.normal, launch_info.indirect.sobel, extent);
        CUDA_SYNC_CHECK();

        ProbeAllocationInfo probe_info;
        probe_info.index_surface = launch_info.index;
        probe_info.position_surface = launch_info.position;
	probe_info.normal_surface = launch_info.normal;
        probe_info.sobel = launch_info.indirect.sobel;
	probe_info.probes = launch_info.indirect.probes;
        probe_info.extent = extent;

        probe_allocation <<< blocks, block_size >>> (probe_info);
        CUDA_SYNC_CHECK();

        FinalGather info;
        info.color = ev->common_rtx.dev_color;
        info.direct = launch_info.direct.Le;
        info.position_surface = launch_info.position;
        info.sobel = launch_info.indirect.sobel;
	info.table = launch_info.indirect.probes;
	info.brute_force = brute_force;
        info.extent = extent;

        final_gather <<< blocks, block_size >>> (info);
        CUDA_SYNC_CHECK();

        // Report any IO exchanges
        std::string io = optix_io_read(&launch_info.io);
        // std::cout << "Mamba GI output: \"" << io << "\"" << std::endl;
        optix_io_clear(&launch_info.io);

        // Update previous camera state
        launch_info.previous_view = camera.view_matrix(camera_transform);
        launch_info.previous_projection = camera.perspective_matrix();
        launch_info.previous_origin = cuda::to_f3(render_info.camera_transform.position);

        // TODO: Creat profile graphs across a fixed animation test
}

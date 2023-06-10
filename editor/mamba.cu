// Engine headers
#include "include/profiler.hpp"
#include "include/cuda/alloc.cuh"
#include "include/cuda/brdf.cuh"

// Local headers
#include "editor/common.hpp"
#include "editor/editor_viewport.cuh"
#include "editor/optix/probe.cuh"
#include "mamba.cuh"

using namespace kobra;

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
                OPTIX_DESC_RAYGEN (module, "__raygen__secondary"),
                OPTIX_DESC_HIT    (module, "__closesthit__"),
                OPTIX_DESC_MISS   (module, "__miss__"),
        };

        // Corresponding program groups
        std::vector <OptixProgramGroup *> program_groups = {
                &raygen_direct_primary,
                &raygen_direct_temporal,
                &raygen_direct_spatial,
		&raygen_secondary,
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

	secondary_ppl = optix::link_optix_pipeline(context, {
		raygen_secondary,
		closest_hit,
		miss,
	}, pipeline_compile_options, pipeline_link_options);

        // Create shader binding tables
        direct_initial_sbt = {};
	secondary_sbt = {};

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

	CUdeviceptr dev_secondary_raygen_record;

	optix::Record <void> secondary_raygen_record;
	OPTIX_CHECK(optixSbtRecordPackHeader(raygen_secondary, &secondary_raygen_record));
	CUDA_CHECK(cudaMalloc((void **) &dev_secondary_raygen_record, sizeof(optix::Record <void>)));
	CUDA_CHECK(cudaMemcpy((void *) dev_secondary_raygen_record, &secondary_raygen_record, sizeof(optix::Record <void>), cudaMemcpyHostToDevice));

        // Miss
        CUdeviceptr dev_miss_record;

        optix::Record <void> miss_record;
        OPTIX_CHECK(optixSbtRecordPackHeader(miss, &miss_record));
        CUDA_CHECK(cudaMalloc((void **) &dev_miss_record, sizeof(optix::Record <void>)));
        CUDA_CHECK(cudaMemcpy((void *) dev_miss_record, &miss_record, sizeof(optix::Record <void>), cudaMemcpyHostToDevice));

	// Set up shader binding tables
        direct_initial_sbt.raygenRecord = dev_direct_initial_raygen_record;
        direct_initial_sbt.missRecordBase = dev_miss_record;
        direct_initial_sbt.missRecordStrideInBytes = sizeof(optix::Record <void>);
        direct_initial_sbt.missRecordCount = 1;
        direct_initial_sbt.hitgroupRecordBase = 0;
        direct_initial_sbt.hitgroupRecordStrideInBytes = 0;
        direct_initial_sbt.hitgroupRecordCount = 0;

	secondary_sbt.raygenRecord = dev_secondary_raygen_record;
	secondary_sbt.missRecordBase = dev_miss_record;
	secondary_sbt.missRecordStrideInBytes = sizeof(optix::Record <void>);
	secondary_sbt.missRecordCount = 1;
	secondary_sbt.hitgroupRecordBase = 0;
	secondary_sbt.hitgroupRecordStrideInBytes = 0;
	secondary_sbt.hitgroupRecordCount = 0;

        // Initialize launch parameters
        launch_info = {};
        launch_info.io = optix_io_create();
        launch_info.direct.reservoirs = 0;
        launch_info.direct.Le = 0;
	launch_info.indirect.probes = alloc_table();
        launch_info.indirect.sobel = 0;
        launch_info.indirect.raster = 0;
        launch_info.indirect.raster_aux = 0;
        launch_info.secondary.hits = 0;
	launch_info.secondary.wi = 0;
	launch_info.secondary.Le = 0;
	launch_info.secondary.caustic_hits = 0;
	launch_info.secondary.caustic_wi = 0;

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
	float *raster;

	IrradianceProbeTable *table;

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

	int32_t best_index = -1;
	int32_t misses = 0;

	int32_t best_distance = 0;

	int32_t x_best = x2 + 8;
	int32_t y_best = y2 + 8;

        // TODO: shared memory: project existing probes onto 16x16 block
        // TODO: need to check if block has invalid regions
	// TODO: floodfill parallelized...
	for (int i = 0; i < 16; i++) {
		for (int j = 0; j < 16; j++) {
			int x3 = x2 + i;
			int y3 = y2 + j;

			if (x3 < 0 || x3 >= info.extent.width || y3 < 0 || y3 >= info.extent.height)
				continue;

			int32_t n_index = y3 * info.extent.width + x3;
			if (info.raster[n_index] != 0)
				continue;

			// Search for the closest
			// int N = 16;
			int N = 4;

			int32_t distance = 0;
			for (int k = -N; k <= N; k++) {
				for (int l = -N; l <= N; l++) {
					int x4 = x3 + k;
					int y4 = y3 + l;

					if (x4 < 0 || x4 >= info.extent.width || y4 < 0 || y4 >= info.extent.height)
						continue;

					int32_t n_index2 = y4 * info.extent.width + x4;
					if (info.raster[n_index2] == 0)
						continue;

					distance = max(abs(k) + abs(l), distance);
				}
			}

			if (distance > best_distance) {
				best_distance = distance;
				best_index = n_index;
				x_best = x3;
				y_best = y3;
			}

			misses++;
		}
	}

	if (misses < 0.8 * 16 * 16)
		return;

	// If all are misses, then use middle
	// TODO: use a special heuristic for allocating in the first frame ish
	if (misses == 16 * 16) {
		x_best = x2 + 8;
		y_best = y2 + 8;
	}

        // For now lazily allocate at the central position of the block
        int x3 = x_best;
        int y3 = y_best;

	if (x3 < 0 || x3 >= info.extent.width || y3 < 0 || y3 >= info.extent.height)
		return;

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

	IrradianceProbeLUT *lut = &info.table->luts[0];
	if (!lut->contains(position))
		return;

	uint32_t hash = lut->hash(position);
	int32_t success = lut->request(position);
	if (success == -1)
		return;

	// Allocate the probe
	int32_t probe_index = info.table->next();
	if (probe_index == -1)
		return;

	IrradianceProbe *probe = &info.table->probes[probe_index];

	*probe = {};
	probe->position = position;
	probe->normal = normal;

	lut->refs[success] = probe_index;
}

// Raster probes for allocating probes at a later stage
// TODO: store raster results into uint8_t or something very compact
struct IrradiaceProbeRasterInfo {
	IrradianceProbeTable *table;
	float *raster;

	float *raster_aux;
	bool enable_aux;

	cudaSurfaceObject_t position_surface;
	vk::Extent2D extent;
};

__global__
void raster_probes(IrradiaceProbeRasterInfo info)
{
	int32_t index = threadIdx.x + blockIdx.x * blockDim.x;
	int32_t stride = blockDim.x * gridDim.x;

	IrradianceProbeLUT *lut = &info.table->luts[0];

	// TODO: each takes 8 bits
	int32_t max = info.extent.width * info.extent.height;
	for (int32_t i = index; i < max; i += stride) {
		// TODO: perform 8 tests and broadcast to each 8-bit correspondence
		// TODO: broadcsat from kernel.. (e.g. shared memory)
		int32_t x = i % info.extent.width;
		int32_t y = i / info.extent.width;

		float4 raw_position;
		surf2Dread(&raw_position, info.position_surface, x * sizeof(float4), info.extent.height - (y + 1));
		float3 position = make_float3(raw_position);

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

		float v = 0;
		float aux_v = 0;

		if (closest_index != -1) {
			int32_t probe_index = closest_index;
			IrradianceProbe *probe = &info.table->probes[probe_index];
			float3 diff = position - probe->position;

			cuda::ONB onb = cuda::ONB::from_normal(probe->normal);
			float3 local = onb.inv_local(diff)/lut->resolution;
			local.z = 0;
			float dist = length(local);

			if (dist < 1) {
				v = 1;
				if (info.enable_aux) {
					float3 wi = make_float3(local.x, local.y, sqrtf(1 - local.x * local.x - local.y * local.y));
					wi = normalize(wi);

					float2 s = probe->to_disk(wi);

					constexpr int32_t size = IrradianceProbe::size;
					int32_t ix = s.x * size;
					int32_t iy = s.y * size;
					int32_t index = ix + iy * size;

					aux_v = (ix + iy) % 2;
				}
			}
		}

		info.raster[i] = v;
		info.raster_aux[i] = aux_v;
	}
}

struct FinalGather {
        float4 *color;
        float3 *direct;
        float *sobel;

	CameraAxis axis;

        cudaSurfaceObject_t index_surface;
        cudaSurfaceObject_t position_surface;

	IrradianceProbeTable *table;
	bool render_probes;
	float *raster;

	float3 *secondary_Le;
	bool enable_secondary_Le;

	Sky sky;

        vk::Extent2D extent;
        vk::Extent2D secondary_extent;
};

	__device__
float3 ray_at(const CameraAxis &axis, uint3 idx)
{
        idx.y = axis.resolution.y - (idx.y + 1);
        float u = 2.0f * float(idx.x) / float(axis.resolution.x) - 1.0f;
        float v = 2.0f * float(idx.y) / float(axis.resolution.y) - 1.0f;
	return normalize(u * axis.U - v * axis.V + axis.W);
}

__global__
void final_gather(FinalGather info)
{
        // TODO: return sky ray if invalid/no hit
        int index = threadIdx.x + blockIdx.x * blockDim.x;

        int x = index % info.extent.width;
        int y = index / info.extent.width;
        if (x >= info.extent.width || y >= info.extent.height)
                return;

	// Get index
	int32_t raw_index;
	surf2Dread(&raw_index, info.index_surface, x * sizeof(int32_t), (info.extent.height - (y + 1)));

	if (raw_index == -1) {
		uint3 idx = make_uint3(x, y, 0);
		info.color[index] = sky_at(info.sky, ray_at(info.axis, idx));
		return;
	}

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

	float3 covered_color = make_float3(0);
	if (info.render_probes)
		covered_color = make_float3(info.raster[index]);

	int32_t x_4 = x / 4;
	int32_t y_4 = y / 4;

	int32_t secondary_index = x_4 + y_4 * info.secondary_extent.width;
	bool secondary_in_bounds = (secondary_index >= 0 && secondary_index < info.secondary_extent.width * info.secondary_extent.height);

	float3 secondary_Le = make_float3(0);
	if (secondary_in_bounds && info.enable_secondary_Le)
		secondary_Le = info.secondary_Le[secondary_index];

	// Assign final color
        info.color[index] = make_float4(color + secondary_Le + 0.25f * covered_color, 1.0f);
}

// Rendering function
void Mamba::render(EditorViewport *ev,
                const RenderInfo &render_info,
                const std::vector <Entity> &entities,
                const MaterialDaemon *md)
{
	KOBRA_PROFILE_TASK("Mamba::render");

        const Camera &camera = render_info.camera;
        const Transform &camera_transform = render_info.camera_transform;

	vk::Extent2D extent = ev->extent;
	vk::Extent2D secondary_extent = { extent.width/4 + 1, extent.height/4 + 1 };

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

		if (launch_info.indirect.raster != 0)
                        CUDA_CHECK(cudaFree((void *) launch_info.indirect.raster));

		if (launch_info.indirect.raster_aux != 0)
                        CUDA_CHECK(cudaFree((void *) launch_info.indirect.raster_aux));

                launch_info.indirect.sobel = cuda::alloc <float> (size);
		launch_info.indirect.raster = cuda::alloc <float> (size);
		launch_info.indirect.raster_aux = cuda::alloc <float> (size);

		// Secondary ray buffers
		// TODO: resize method for the structs...
		launch_info.secondary.resolution = { secondary_extent.width, secondary_extent.height };

		if (launch_info.secondary.hits != 0)
			CUDA_CHECK(cudaFree((void *) launch_info.secondary.hits));

		if (launch_info.secondary.caustic_hits != 0)
			CUDA_CHECK(cudaFree((void *) launch_info.secondary.caustic_hits));

		if (launch_info.secondary.wi != 0)
			CUDA_CHECK(cudaFree((void *) launch_info.secondary.wi));

		if (launch_info.secondary.caustic_wi != 0)
			CUDA_CHECK(cudaFree((void *) launch_info.secondary.caustic_wi));

		if (launch_info.secondary.Le != 0)
			CUDA_CHECK(cudaFree((void *) launch_info.secondary.Le));

		launch_info.secondary.hits = cuda::alloc <cuda::SurfaceHit> (secondary_extent.width * secondary_extent.height);
		launch_info.secondary.caustic_hits = cuda::alloc <cuda::SurfaceHit> (secondary_extent.width * secondary_extent.height);

		launch_info.secondary.wi = cuda::alloc <float3> (secondary_extent.width * secondary_extent.height);
		launch_info.secondary.caustic_wi = cuda::alloc <float3> (secondary_extent.width * secondary_extent.height);

		launch_info.secondary.Le = cuda::alloc <float3> (secondary_extent.width * secondary_extent.height);
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

	{
		KOBRA_PROFILE_CUDA_TASK("Direct (Resampled) Lighting");
		OPTIX_CHECK(
			optixLaunch(direct_ppl, 0,
				dev_launch_info,
				sizeof(MambaLaunchInfo),
				&direct_initial_sbt, extent.width, extent.height, 1
			)
		);

		CUDA_SYNC_CHECK();
	}

        if (temporal_reuse) {
		KOBRA_PROFILE_CUDA_TASK("Temporal Reuse");
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
		KOBRA_PROFILE_CUDA_TASK("Spatial Reuse");
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

	// Secondary rays
	{
		KOBRA_PROFILE_CUDA_TASK("Secondary Rays");
		OPTIX_CHECK(
			optixLaunch(secondary_ppl, 0,
				dev_launch_info,
				sizeof(MambaLaunchInfo),
				&secondary_sbt, secondary_extent.width, secondary_extent.height, 1
			)
		);

		CUDA_SYNC_CHECK();
	}

        // TODO: more advanced parallelization
        uint block_size = 256;
        uint blocks = (extent.width * extent.height + 255) / 256;

        // Final gather
	{
		KOBRA_PROFILE_CUDA_TASK("Sobel Filter");
		sobel_normal <<< blocks, block_size >>> (launch_info.normal, launch_info.indirect.sobel, extent);
		CUDA_SYNC_CHECK();
	}

	IrradiaceProbeRasterInfo raster_info;
	raster_info.table = launch_info.indirect.probes;
	raster_info.raster = launch_info.indirect.raster;
	raster_info.position_surface = launch_info.position;
	raster_info.raster_aux = launch_info.indirect.raster_aux;
	raster_info.enable_aux = render_probe_aux;
	raster_info.extent = extent;

	{
		KOBRA_PROFILE_CUDA_TASK("Irradiance Probe Rasterization");
		raster_probes <<< blocks, block_size >>> (raster_info);
		CUDA_SYNC_CHECK();
	}

        ProbeAllocationInfo probe_info;
        probe_info.index_surface = launch_info.index;
        probe_info.position_surface = launch_info.position;
	probe_info.normal_surface = launch_info.normal;
        probe_info.sobel = launch_info.indirect.sobel;
	probe_info.table = launch_info.indirect.probes;
	probe_info.raster = launch_info.indirect.raster;
        probe_info.extent = extent;

	{
		KOBRA_PROFILE_CUDA_TASK("Irradiance Probe Allocation");
		probe_allocation <<< blocks, block_size >>> (probe_info);
		CUDA_SYNC_CHECK();
	}

        FinalGather info;
        info.color = ev->common_rtx.dev_color;
        info.direct = launch_info.direct.Le;
	info.axis = launch_info.camera;
	info.index_surface = launch_info.index;
        info.position_surface = launch_info.position;
        info.sobel = launch_info.indirect.sobel;
	info.table = launch_info.indirect.probes;
	info.render_probes = render_probes;
	info.raster = render_probe_aux ? launch_info.indirect.raster_aux : launch_info.indirect.raster;
	info.secondary_Le = launch_info.secondary.Le;
	info.enable_secondary_Le = indirect_lighting;
	info.sky = launch_info.sky;
        info.extent = extent;
	info.secondary_extent = secondary_extent;

	{
		KOBRA_PROFILE_CUDA_TASK("Final Gather");
		final_gather <<< blocks, block_size >>> (info);
		CUDA_SYNC_CHECK();
	}

        // Report any IO exchanges
        std::string io = optix_io_read(&launch_info.io);
        // std::cout << "Mamba GI output: \"" << io << "\"" << std::endl;
        optix_io_clear(&launch_info.io);

        // Update previous camera state
        launch_info.previous_view = camera.view_matrix(camera_transform);
        launch_info.previous_projection = camera.perspective_matrix();
        launch_info.previous_origin = cuda::to_f3(render_info.camera_transform.position);

        // TODO: Create profile graphs across a fixed animation test
	// TODO: create camera trajectory system to generate a time graph profile graph..

	// TODO: imgui show probe information
}

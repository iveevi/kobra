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

__global__
void probe_allocation(ProbeAllocationInfo);

// Raster probes for allocating probes at a later stage
// TODO: store raster results into uint8_t or something very compact
struct IrradiaceProbeRasterInfo {
	IrradianceProbeTable *table;
	float *raster;

	float3 *raster_aux;
	bool enable_aux;

	cudaSurfaceObject_t position_surface;
	vk::Extent2D extent;
};

// TODO: cache closest 4 probes per pixel... (as long as within lut->resolution)
__global__
void raster_probes(IrradiaceProbeRasterInfo info)
{
	// Color wheel for auxilary data
	constexpr float3 COLOR_WHEEL[12] = {
		float3 { 0.910, 0.490, 0.490 },
		float3 { 0.910, 0.700, 0.490 },
		float3 { 0.910, 0.910, 0.490 },
		float3 { 0.700, 0.910, 0.490 },
		float3 { 0.490, 0.910, 0.490 },
		float3 { 0.490, 0.910, 0.700 },
		float3 { 0.490, 0.910, 0.910 },
		float3 { 0.490, 0.700, 0.910 },
		float3 { 0.490, 0.490, 0.910 },
		float3 { 0.700, 0.490, 0.910 },
		float3 { 0.910, 0.490, 0.910 },
		float3 { 0.910, 0.490, 0.700 }
	};

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

		// First retrieve the lower level LUT
		int32_t result = lookup_L2(lut, position);
		if (result == -1)
			continue;

		assert(result >= 0 && result < MAX_REFS);

		// TODO: neighbnoring lower level LUTs?
		int32_t lut_index = lut->refs[result];

		// assert(lut_index >= 0 && lut_index < info.table->counter);
		// TODO: this is a fail safe... find the real fix
		if (lut_index < 0 || lut_index >= info.table->counter) {
			printf("[!] BAD LUT INDEX: %d\n", lut_index);
			continue;
		}

		IrradianceProbeLUT *nlut = &info.table->luts[lut_index];

		float closest = FLT_MAX;
		int32_t closest_index = -1;

		int32_t result2 = nlut->lookup(position, -1);
		if (result2 != -1) {
			// printf("Hash hit: %d\n", result2);
			assert(result2 >= 0 && result2 < MAX_REFS);

			int32_t probe_index = nlut->refs[result2];
			assert(probe_index >= 0 && probe_index < info.table->counter);

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
			int32_t result = nlut->lookup(position, ni);
			if (result == -1)
				continue;

			int32_t probe_index = nlut->refs[result];
			IrradianceProbe *probe = &info.table->probes[probe_index];
			float3 diff = position - probe->position;
			float dist = length(diff);

			if (dist < closest) {
				closest = dist;
				closest_index = probe_index;
			}
		}

		float v = 0;
		float3 aux_v = make_float3(lut_index % 2);
		// float3 aux_v = make_float3(2.0f * length(position - nlut->center)/lut->size);

		if (closest_index != -1) {
			int32_t probe_index = closest_index;
			IrradianceProbe *probe = &info.table->probes[probe_index];
			float3 diff = position - probe->position;

			cuda::ONB onb = cuda::ONB::from_normal(probe->normal);
			float3 local = onb.inv_local(diff)/nlut->resolution;
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

					int32_t index = probe->to_index(s);
					assert(index >= 0 && index < size * size);

					// aux_v = probe->pdfs[index] > 0 ? probe->Le[index]/probe->pdfs[index] : make_float3(0.0f);
					int32_t mod = (ix + iy) % 2;
					// aux_v = make_float3(mod, 0, 1 - mod);

					float3 color = COLOR_WHEEL[probe_index % 12];
					aux_v = color * (0.5f + 0.5f * mod);
				}
			}
		}

		info.raster[i] = v;
		info.raster_aux[i] = aux_v;
	}
}

// Process/gather secondary rays
struct SecondaryGatherInfo {
	IrradianceProbeTable *table;

	cudaSurfaceObject_t position;
	cudaSurfaceObject_t normal;
	cudaSurfaceObject_t uv;
	cudaSurfaceObject_t index;

	cuda::SurfaceHit *hits;
	float3 *wi;
	float3 *Le;

	vk::Extent2D secondary_extent;
	vk::Extent2D extent;
	int32_t samples;
};

__global__
void secondary_gather(SecondaryGatherInfo info)
{
	int32_t index = threadIdx.x + blockIdx.x * blockDim.x;

	int32_t x = index % info.secondary_extent.width;
	int32_t y = index / info.secondary_extent.width;

	int32_t block_cycle = info.samples % 16;
	int32_t true_x = 4 * x + (block_cycle % 4);
	int32_t true_y = 4 * y + (block_cycle / 4);

	if (x >= info.secondary_extent.width || y >= info.secondary_extent.height)
		return;

	if (true_x >= info.extent.width || true_y >= info.extent.height)
		return;

	// Fetch data
	cuda::SurfaceHit secondary_hit = info.hits[index];
	float3 wi = info.wi[index];
	float3 Le = info.Le[index];

	// TODO: insert/update method
	IrradianceProbeLUT *lut = &info.table->luts[0];

	// TODO: verify with index surface...
	float4 raw_position;
	float4 raw_normal;
	int32_t raw_index;

	surf2Dread(&raw_position, info.position, true_x * sizeof(float4), info.extent.height - (true_y + 1));
	surf2Dread(&raw_normal, info.normal, true_x * sizeof(float4), info.extent.height - (true_y + 1));
	surf2Dread(&raw_index, info.index, true_x * sizeof(int32_t), info.extent.height - (true_y + 1));
	if (raw_index == -1) {
		info.Le[index] = make_float3(0);
		return;
	}

	float3 position = make_float3(raw_position);
	float3 normal = make_float3(raw_normal);

	// float radius = lut->resolution;
	int32_t result = lut->full_lookup(position);
	if (result == -1) {
		info.Le[index] = make_float3(0);
		return;
	}

	cuda::ONB onb = cuda::ONB::from_normal(normal);
	float3 local_wi = normalize(onb.inv_local(wi));

	int32_t probe_index = lut->refs[result];
	IrradianceProbe *probe = &info.table->probes[probe_index];
	float2 uv = probe->to_disk(local_wi);
	int32_t pi = probe->to_index(uv);

	// TODO: convolve with surface normal...
	// TODO: spread to neighbors as well...
	// probe->Le[pi] = Le * abs(local_wi.z);
	// assert(!isnan(Le.x) && !isnan(Le.y) && !isnan(Le.z));
	Le = Le;

	// info.table->lock(probe_index);

	atomicAdd(&probe->Le[pi].x, Le.x);
	atomicAdd(&probe->Le[pi].y, Le.y);
	atomicAdd(&probe->Le[pi].z, Le.z);
	atomicAdd(&probe->pdfs[pi], 1);

	// probe->Le[pi] = Le;
	// info.table->unlock(probe_index);

	float distance = length(position - secondary_hit.x);
	atomicExch(&probe->depth[pi], distance);
}

// Final illumination gather
// TODO: ignore dedicated direct lighting... add it to the probes (but resample)
struct FinalGather {
        float4 *color;
        float3 *direct;
        float *sobel;

	CameraAxis axis;

        cudaSurfaceObject_t index;
        cudaSurfaceObject_t position;
        cudaSurfaceObject_t normal;

	IrradianceProbeTable *table;

	float *raster;
	float3 *raster_aux;
	bool render_probes;
	bool enable_aux;

	float3 *secondary_Le;
	float3 *secondary_wi;
	cuda::SurfaceHit *secondary_hits;

	bool indirect_lighting;
	bool irradiance;

	Sky sky;

        vk::Extent2D extent;
        vk::Extent2D secondary_extent;

	// Options (flags)
	bool direct_lighting;
};

__forceinline__ __device__
float3 cleanse(float3 in)
{
        if (isnan(in.x) || isnan(in.y) || isnan(in.z))
                return make_float3(0.0f);
        return in;
}

__forceinline__ __device__
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

	// Fetch surface data
	float4 raw_position;
	float4 raw_normal;
	int32_t raw_index;

	surf2Dread(&raw_position, info.position, x * sizeof(float4), (info.extent.height - (y + 1)));
	surf2Dread(&raw_normal, info.normal, x * sizeof(float4), (info.extent.height - (y + 1)));
	surf2Dread(&raw_index, info.index, x * sizeof(int32_t), (info.extent.height - (y + 1)));

	if (raw_index == -1) {
		uint3 idx = make_uint3(x, y, 0);
		info.color[index] = sky_at(info.sky, ray_at(info.axis, idx));
		return;
	}

	float3 position = make_float3(raw_position);
	float3 normal = make_float3(raw_normal);

	// TODO: flags for direct and sobel
        bool border = (info.sobel[index] > 0.7f);
        float3 color = info.direct_lighting * info.direct[index] + 0 * border;

        int x_16 = x / 16;
        int y_16 = y / 16;

	IrradianceProbeLUT *lut = &info.table->luts[0];

	float3 covered_color = make_float3(0);
	if (info.render_probes) {
		if (info.enable_aux)
			covered_color = info.raster_aux[index];
		else
			covered_color = make_float3(info.raster[index]);
	}

	int32_t x_4 = x / 4;
	int32_t y_4 = y / 4;

	int32_t secondary_index = x_4 + y_4 * info.secondary_extent.width;
	bool secondary_in_bounds = (secondary_index >= 0 && secondary_index < info.secondary_extent.width * info.secondary_extent.height);

	float3 secondary_Le = make_float3(0);
	if (secondary_in_bounds && info.indirect_lighting) {
		secondary_Le = info.secondary_Le[secondary_index];
	} else if (info.irradiance) {
		IrradianceProbeLUT *lut = &info.table->luts[0];

		constexpr int32_t N = 8;
		int32_t closest[N] = { -1, -1, -1, -1 };
		float distance[N] = { FLT_MAX, FLT_MAX, FLT_MAX, FLT_MAX };

		lut->closest <N> (position, closest, distance);

		if (closest[0] != -1) {
			// int32_t probe_index = lut->refs[closest[0]];
			// IrradianceProbe *probe = &info.table->probes[probe_index];
			// secondary_Le = make_float3(1 - length(probe->position - position)/(2 * lut->resolution));

			float3 sum = make_float3(0);
			float sum_w = 0;
			int32_t count = 0;

			for (int i = 0; i < N; i++) {
				if (closest[i] == -1)
					break;

				int32_t probe_index = lut->refs[closest[i]];
				IrradianceProbe *probe = &info.table->probes[probe_index];
				// sum += probe->Le[probe->to_index(probe->to_disk(position))];

				// if (dot(normal, probe->normal) < 0.4f)
				// 	continue;

				int32_t SIZE = IrradianceProbe::size * IrradianceProbe::size;

				float w = exp(-distance[i] * distance[i] / (2 * lut->resolution * lut->resolution));
				sum_w += w;

				cuda::ONB onb = cuda::ONB::from_normal(probe->normal);
				for (int j = 0; j < SIZE; j++) {
					float3 wo = probe->from_index(j);
					wo = onb.local(wo);
					float geometric = abs(dot(wo, normal));

					float3 Le = probe->Le[j];
					float n = probe->pdfs[j];
					// sum += (n > 0) ? w * geometric * cleanse(probe->Le[i])/(probe->pdfs[i] * float(SIZE)) : make_float3(0);
					sum += (n > 0) ? w * cleanse(probe->Le[i])/(probe->pdfs[i] * float(SIZE)) : make_float3(0);
				}

				count++;
			}

			secondary_Le = sum/(sum_w * float(count));
		}
	}

	// Assign final color
        info.color[index] = make_float4(color + secondary_Le + covered_color, 1.0f);
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
		launch_info.indirect.raster_aux = cuda::alloc <float3> (size);

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

        // TODO: more advanced parallelization
        uint block_size = 256;
        uint blocks = (extent.width * extent.height + 255) / 256;

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

	// Gather and process secondary ray information
	{
		// KOBRA_PROFILE_CUDA_TASK("Gather Secondary Rays");
		//
		// SecondaryGatherInfo gather_info;
		// gather_info.table = launch_info.indirect.probes;
		// gather_info.position = launch_info.position;
		// gather_info.normal = launch_info.normal;
		// gather_info.uv = launch_info.uv;
		// gather_info.index = launch_info.index;
		// gather_info.hits = launch_info.secondary.hits;
		// gather_info.wi = launch_info.secondary.wi;
		// gather_info.Le = launch_info.secondary.Le;
		// gather_info.secondary_extent = secondary_extent;
		// gather_info.extent = extent;
		// gather_info.samples = launch_info.samples;
		//
		// secondary_gather <<< blocks, block_size >>> (gather_info);
		// CUDA_SYNC_CHECK();
	}

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
		// printf("[START] --------------------->\n");
		KOBRA_PROFILE_CUDA_TASK("Irradiance Probe Allocation");
		probe_allocation <<< blocks, block_size >>> (probe_info);
		CUDA_SYNC_CHECK();
		// printf("<--------------------- [END]\n");
	}

        FinalGather info;
        info.color = ev->common_rtx.dev_color;
        info.direct = launch_info.direct.Le;
	info.axis = launch_info.camera;
        info.position = launch_info.position;
        info.normal = launch_info.normal;
	info.index = launch_info.index;
        info.sobel = launch_info.indirect.sobel;
	info.table = launch_info.indirect.probes;
	info.raster = launch_info.indirect.raster;
	info.raster_aux = launch_info.indirect.raster_aux;
	info.render_probes = render_probes;
	info.enable_aux = render_probe_aux;
	info.secondary_Le = launch_info.secondary.Le;
	info.secondary_wi = launch_info.secondary.wi;
	info.secondary_hits = launch_info.secondary.hits;
	info.indirect_lighting = options.indirect_lighting;
	info.irradiance = options.irradiance;
	info.sky = launch_info.sky;
        info.extent = extent;
	info.secondary_extent = secondary_extent;
	info.direct_lighting = options.direct_lighting;

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

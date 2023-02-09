#ifndef KOBRA_AMADEUS_REPG_H_
#define KOBRA_AMADEUS_REPG_H_

// Engine headers
#include "armada.cuh"
#include "reservoir.cuh"

namespace kobra {

namespace amadeus {

struct Cell {
	// Texture of reservoirs and path guiding info
};

// Launch parameters
struct RePG_Parameters : ArmadaLaunchInfo {
	// Scene traversal
	OptixTraversableHandle traversable;

	// Resampling
	Reservoir <DirectLightingSample> *reservoirs;
	cuda::Material *materials_buffer;
};

// TODO: resampling and shadow rays done in another stream

// Reuse and Path Guiding (RePG) kernel
class RePG : public AttachmentRTX {
	// SBT record types
	using RaygenRecord = optix::Record <int>;
	using MissRecord = optix::Record <int>;
	using HitgroupRecord = optix::Record <optix::Hit>;

	vk::Extent2D m_extent;

	// Pipeline related information
	OptixModule m_module;

	OptixProgramGroup m_ray_generation;
	OptixProgramGroup m_closest_hit;
	OptixProgramGroup m_miss;
	OptixProgramGroup m_miss_shadow;

	OptixPipeline m_pipeline;

	// Buffer for launch parameters
	RePG_Parameters m_parameters;
	CUdeviceptr m_cuda_parameters;

	// TODO: stream

	// Create the program groups and pipeline
	void create_pipeline(const OptixDeviceContext &optix_context) {
		static constexpr const char OPTIX_PTX_FILE[] = "bin/ptx/amadeus_repg.ptx";

		// Load module
		m_module = optix::load_optix_module(
			optix_context, OPTIX_PTX_FILE,
			ppl_compile_options, module_options
		);

		// Load programs
		OptixProgramGroupOptions program_options = {};

		// Descriptions of all the programs
		std::vector <OptixProgramGroupDesc> program_descs = {
			OPTIX_DESC_RAYGEN(m_module, "__raygen__"),
			OPTIX_DESC_HIT(m_module, "__closesthit__"),
			OPTIX_DESC_MISS(m_module, "__miss__"),
			OPTIX_DESC_MISS(m_module, "__miss__shadow")
		};

		// Corresponding program groups
		std::vector <OptixProgramGroup *> program_groups = {
			&m_ray_generation,
			&m_closest_hit,
			&m_miss,
			&m_miss_shadow
		};

		optix::load_program_groups(
			optix_context,
			program_descs,
			program_options,
			program_groups
		);

		m_pipeline = optix::link_optix_pipeline(
			optix_context,
			{
				m_ray_generation,
				m_closest_hit,
				m_miss,
				m_miss_shadow
			},
			ppl_compile_options,
			ppl_link_options
		);
	}

	OptixShaderBindingTable m_sbt;
public:
	// Constructor
	RePG() : AttachmentRTX(1) {}

	// Attaching and unloading
	void attach(const ArmadaRTX &armada_rtx) override {
		m_extent = armada_rtx.extent();

		// First load the pipeline
		create_pipeline(armada_rtx.system()->context());

		// Allocate the SBT
		std::vector <RaygenRecord> ray_generation_records(1);
		std::vector <MissRecord> miss_records(2);

		// Fill the SBT
		optix::pack_header(m_ray_generation, ray_generation_records[0]);

		optix::pack_header(m_miss, miss_records[0]);
		optix::pack_header(m_miss_shadow, miss_records[1]);

		// Create the SBT
		m_sbt = {};

		m_sbt.raygenRecord = cuda::make_buffer_ptr(ray_generation_records);

		m_sbt.missRecordBase = cuda::make_buffer_ptr(miss_records);
		m_sbt.missRecordStrideInBytes = sizeof(MissRecord);
		m_sbt.missRecordCount = miss_records.size();

		m_sbt.hitgroupRecordBase = 0;
		m_sbt.hitgroupRecordStrideInBytes = 0;
		m_sbt.hitgroupRecordCount = 0;

		// Initialize the parameters buffer
		m_cuda_parameters = (CUdeviceptr) cuda::alloc <RePG_Parameters> (1);
	}

	void load() override {
		// Allocate the reservoirs and indirect buffer
		std::vector <Reservoir <DirectLightingSample>> reservoirs(m_extent.width * m_extent.height);
		m_parameters.reservoirs = cuda::make_buffer(reservoirs);
		m_parameters.materials_buffer = cuda::alloc <cuda::Material> (m_extent.width * m_extent.height);
	}

	void unload() override {
		// Free the reservoirs
		cuda::free(m_parameters.reservoirs);
		cuda::free(m_parameters.materials_buffer);
	}

	// Rendering
	void render(const ArmadaRTX *armada_rtx,
			const ArmadaLaunchInfo &launch_info,
			const std::optional <OptixTraversableHandle> &handle,
			std::vector <HitRecord> *hit_records,
			const vk::Extent2D &extent) override {
		// Check if hit groups need to be updated, and update them if necessary
		if (hit_records) {
			for (auto &hitgroup_record : *hit_records)
				optix::pack_header(m_closest_hit, hitgroup_record);

			// Free old buffer
			if (m_sbt.hitgroupRecordBase)
				cuda::free(m_sbt.hitgroupRecordBase);

			// Update the SBT
			m_sbt.hitgroupRecordBase = cuda::make_buffer_ptr(*hit_records);
			m_sbt.hitgroupRecordStrideInBytes = sizeof(HitRecord);
			m_sbt.hitgroupRecordCount = hit_records->size();
		}

		// Copy the parameters and launch
		std::memcpy(&m_parameters, &launch_info, sizeof(ArmadaLaunchInfo));

		if (handle)
			m_parameters.traversable = *handle;

		cuda::copy(m_cuda_parameters, &m_parameters, 1, cudaMemcpyHostToDevice);

		// Execute the pipeline
		OPTIX_CHECK(
			optixLaunch(
				m_pipeline, 0,
				m_cuda_parameters, sizeof(RePG_Parameters),
				&m_sbt, extent.width, extent.height, 1
			)
		);

		CUDA_SYNC_CHECK();
	}
};

}

}

#endif

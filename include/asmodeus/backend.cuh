#ifndef KOBRA_ASMODEUS_BACKEND_H_
#define KOBRA_ASMODEUS_BACKEND_H_

// OptiX headers
#include <optix.h>
#include <optix_stubs.h>

// Engine headers
#include "../backend.hpp"
#include "../logger.hpp"
#include "../optix/lighting.cuh"
#include "../renderer.hpp"
#include "../ecs.hpp"

namespace kobra {

namespace asmodeus {

/* Backend for Asmodeus, the system for real-time global illumination:
 *
 * Controls bare bones resources from APIs like OptiX, Vulkan, etc. Also manages
 * memory and instances for this purpose, and provides a simple interface for
 * batching the tracing kernels.
 */
struct Backend {
	// TODO: ensure that the same device is used for all resources,
	// across both Vulkan and OptiX
	
	// For the raytracing backend, only one
	enum class BackendType {
		eOptiX,
		eVulkan
	} rtx_backend;
	
	// Vulkan structures
	Device device;
	
	// OptiX critical structures
	OptixDeviceContext optix_context = 0;

	// Acceleration structure management
	struct Instance {
		const Submesh *submesh;
		const Transform *transform;
		OptixTraversableHandle handle;
	};

	std::vector <Instance> instances;

	// Map each pipeline to an Hit SBT allocator
	struct Pipeline;

	using HitSbtAllocator = std::function <
		void (
			const Backend &,
			const Pipeline &,
			const std::vector <Instance> &,
			CUdeviceptr &, size_t &
		)
	>;

	// The backend can store multiple pipelines
	struct Pipeline {
		int expected_miss;
		int expected_hit;

		// Program groups are stored only for reference,
		// they must still be created by users
		OptixProgramGroup ray_generation = 0;
		std::vector <OptixProgramGroup> miss;
		std::vector <OptixProgramGroup> hit;

		// Pipelines have their own SBTs and TLAS
		OptixPipeline pipeline;
		OptixShaderBindingTable sbt;
		OptixTraversableHandle tlas;
		CUstream stream = 0; // TODO: allocate streams for each pipeline

		// Last time we updated the TLAS
		int last_update = -1;

		// Hit SBT allocator
		HitSbtAllocator hit_sbt_allocator;

		// Constuctor with expected number of miss and hit groups
		Pipeline(int = 1, int = 1);

		// Launch the pipeline
		void launch(CUdeviceptr, size_t, int, int);
	};
	
	std::vector <Pipeline> pipelines;
	
	// Scene data cache
	struct {
		std::vector <const Rasterizer *> c_rasterizers;
	} scene;

	// Construction
	static Backend make(const Context &, const BackendType &);
};

// Pipeline methods
// TODO: member functions for pipelines
void set_programs(Backend::Pipeline &,
	OptixDeviceContext,
	OptixProgramGroup,
	const std::vector <OptixProgramGroup> &,
	const std::vector <OptixProgramGroup> &,
	const OptixPipelineCompileOptions &,
	const OptixPipelineLinkOptions &
);

void initialize_sbt(Backend::Pipeline &,
	CUdeviceptr,
	CUdeviceptr, size_t,
	const Backend::HitSbtAllocator &
);

// TODO: launch methods (functional)

// Methods
bool update(Backend &, const ECS &);
OptixTraversableHandle construct_tlas(Backend &, int);
cudaTextureObject_t import_texture(const Backend &, const std::string &);

}

}

#endif

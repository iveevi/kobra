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
	
	// OptiX structures
	struct {
		OptixDeviceContext context = 0;
		OptixShaderBindingTable sbt;
	} optix; 

	// Acceleration structure management
	struct Instance {
		OptixTraversableHandle handle;
		void *sbt_record;
	};

	struct {
		std::vector <Instance> instances;
		OptixTraversableHandle tlas;
	} as;
	
	// Scene data cache
	struct {
		std::vector <const Rasterizer *> c_rasterizers;
	} scene;

	// Construction
	static Backend make(const Context &context, const BackendType &rtx_backend) {
		Backend backend;

		// Initialize basic variables
		KOBRA_ASSERT(
			rtx_backend == BackendType::eOptiX,
			"Only OptiX backend is supported for now"
		);

		backend.rtx_backend = rtx_backend;
		backend.device = context.dev();

		return backend;
	}
};

// Methods
void update(Backend &, const ECS &);

}

}

#endif

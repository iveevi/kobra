#ifndef KOBRA_ASMODEUS_WSRIS_H_
#define KOBRA_ASMODEUS_WSRIS_H_

// Standard headers
#include <map>
#include <vector>

// OptiX headers
#include <optix.h>
#include <optix_stubs.h>

// Engine headers
#include "../backend.hpp"
#include "../core/async.hpp"
#include "../core/kd.cuh"
#include "../optix/parameters.cuh"
#include "../timer.hpp"
#include "../vertex.hpp"
#include "wsris_kd_parameters.cuh"
#include "backend.cuh"

namespace kobra {

// Forward declarations
class ECS;
class Camera;
class Transform;
class Rasterizer;

namespace asmodeus {

struct WorldSpaceKdReservoirs {
	// Critical Vulkan structures
	vk::raii::Device *device = nullptr;
	vk::raii::PhysicalDevice *phdev = nullptr;
	vk::raii::DescriptorPool *descriptor_pool = nullptr;

	// CUDA launch stream
	CUstream optix_stream = 0;

	// Vulkan structures
	vk::Extent2D extent = { 0, 0 };

	// Optix structures
	OptixDeviceContext optix_context = nullptr;
	OptixPipeline optix_pipeline = nullptr;
	OptixShaderBindingTable optix_sbt = {};

	// OptiX modules
	// TODO: we should delete this?
	OptixModule optix_module = nullptr;

	struct {
		OptixTraversableHandle handle = 0;
	} optix;

	// Program groups
	struct {
		OptixProgramGroup raygen = nullptr;
		OptixProgramGroup miss = nullptr;
		OptixProgramGroup hit = nullptr;

		OptixProgramGroup shadow_miss = nullptr;
		OptixProgramGroup shadow_hit = nullptr;
	} optix_programs;

	// Launch parameters
	optix::WorldSpaceKdReservoirsParameters launch_params;

	CUdeviceptr launch_params_buffer = 0;

	// Host buffer analogues
	// TODO: common algorithm for BVH construction...
	struct {
		std::vector <optix::QuadLight> quad_lights;
		std::vector <optix::TriangleLight> tri_lights;
	} host;

	// Cached data
	struct {
		std::vector <const Rasterizer *> rasterizers;
	} cache;

	// Timer
	Timer timer;

	// Others
	float4 *positions = nullptr;
	bool initial_kd_tree = false;

	// Functions
	static WorldSpaceKdReservoirs make(const Context &, const vk::Extent2D &);

	// Proprety methods
	size_t size() {
		return extent.width * extent.height;
	}

	// Buffer accessors
	CUdeviceptr color_buffer() {
		return (CUdeviceptr) launch_params.color_buffer;
	}

	CUdeviceptr normal_buffer() {
		return (CUdeviceptr) launch_params.normal_buffer;
	}

	CUdeviceptr albedo_buffer() {
		return (CUdeviceptr) launch_params.albedo_buffer;
	}

	CUdeviceptr position_buffer() {
		return (CUdeviceptr) launch_params.position_buffer;
	}

	// Other methods
	void set_envmap(const std::string &);

	void render(
		const ECS &, const Camera &,
		const Transform &,
		unsigned int, bool = false
	);
};

// World Space Grid-based Reservoirs
struct GridBasedReservoirs {
	Backend *backend;

	vk::Extent2D extent;

	static GridBasedReservoirs make(Backend &, const vk::Extent2D &);
private:
	OptixModule module;
};

}

}

#endif

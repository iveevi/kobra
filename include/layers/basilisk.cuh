#ifndef KOBRA_LAYERS_BASILISK_H_
#define KOBRA_LAYERS_BASILISK_H_

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
#include "../optix/sbt.cuh"
#include "../optix/parameters.cuh"
#include "../timer.hpp"
#include "../vertex.hpp"

namespace kobra {

// Forward declarations
class ECS;
class Camera;
class Transform;
class Renderable;

namespace layers {

// Regular path tracer
struct Basilisk {
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
	OptixModule optix_module = nullptr;
	OptixModule optix_restir_module = nullptr;
	OptixModule optix_voxel_module = nullptr;

	struct {
		OptixTraversableHandle handle = 0;
	} optix;

	// Program groups
	struct {
		OptixProgramGroup raygen = nullptr;
		OptixProgramGroup miss = nullptr;
		OptixProgramGroup hit = nullptr;
		OptixProgramGroup hit_restir = nullptr;
		OptixProgramGroup hit_restir_pt = nullptr;
		OptixProgramGroup hit_voxel = nullptr;

		OptixProgramGroup shadow_miss = nullptr;
		OptixProgramGroup shadow_hit = nullptr;
	} optix_programs;

	// Launch parameters
	optix::BasiliskParameters launch_params;

	CUdeviceptr launch_params_buffer = 0;

	// Host buffer analogues
	// TODO: common algorithm for BVH construction...
	struct {
		std::vector <optix::QuadLight> quad_lights;
		std::vector <optix::TriangleLight> tri_lights;
	} host;

	// Cached data
	struct {
		std::vector <const Renderable *> rasterizers;
	} cache;

	// Timer
	Timer timer;

	// Others
	float4 *positions = nullptr;
	bool initial_kd_tree = false;
	core::AsyncTask *async_task = nullptr;

	// Functions
	static Basilisk make(const Context &, const vk::Extent2D &);
};

// Proprety methods
inline size_t size(const Basilisk &layer)
{
	return layer.extent.width * layer.extent.height;
}

// Buffer accessors
inline CUdeviceptr color_buffer(const Basilisk &layer)
{
	return (CUdeviceptr) layer.launch_params.color_buffer;
}

inline CUdeviceptr normal_buffer(const Basilisk &layer)
{
	return (CUdeviceptr) layer.launch_params.normal_buffer;
}

inline CUdeviceptr albedo_buffer(const Basilisk &layer)
{
	return (CUdeviceptr) layer.launch_params.albedo_buffer;
}

inline CUdeviceptr position_buffer(const Basilisk &layer)
{
	return (CUdeviceptr) layer.launch_params.position_buffer;
}

// Other methods
void set_envmap(Basilisk &, const std::string &);

void compute(Basilisk &, const ECS &, const Camera &, const Transform &,
		unsigned int, bool = false);

}

}

#endif

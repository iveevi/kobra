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
#include "../optix/parameters.cuh"
#include "../timer.hpp"
#include "../vertex.hpp"

namespace kobra {

// Forward declarations
class ECS;
class Camera;
class Transform;
class Rasterizer;

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
	RenderPass render_pass = nullptr;

	Pipeline pipeline = nullptr;
	PipelineLayout ppl = nullptr;

	vk::Extent2D extent = { 0, 0 };

	// Descriptor set bindings
	static const std::vector <DSLB> dsl_bindings;

	// Descriptor set layout
	vk::raii::DescriptorSetLayout dsl = nullptr;

	// Descriptor sets
	vk::raii::DescriptorSet dset = nullptr;

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
		OptixProgramGroup hit_voxel = nullptr;

		OptixProgramGroup shadow_miss = nullptr;
		OptixProgramGroup shadow_hit = nullptr;
	} optix_programs;

	// Launch parameters
	optix::BasiliskParameters launch_params;

	CUdeviceptr launch_params_buffer = 0;
	CUdeviceptr truncated = 0;

	// Host buffer analogues
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

	// Data for rendering
	ImageData result_image = nullptr;
	BufferData result_buffer = nullptr;
	vk::raii::Sampler result_sampler = nullptr;

	// Functions
	static Basilisk make(const Context &);
};

// Proprety methods
inline size_t size(const Basilisk &layer)
{
	return layer.extent.width * layer.extent.height;
}

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

// Other methods
void set_envmap(Basilisk &, const std::string &);

void compute(Basilisk &, const ECS &, const Camera &, const Transform &,
		unsigned int, bool = false);

}

}

#endif
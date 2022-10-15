#ifndef KOBRA_LAYERS_WADJET_H_
#define KOBRA_LAYERS_WADJET_H_

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
struct Wadjet {
	// Critical Vulkan structures
	vk::raii::Device *device = nullptr;
	vk::raii::PhysicalDevice *phdev = nullptr;
	vk::raii::DescriptorPool *descriptor_pool = nullptr;

	// CUDA launch stream
	CUstream optix_stream = 0;

	// Depth buffer
	DepthBuffer depth = nullptr;

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
	OptixModule optix_module = nullptr;
	OptixPipeline optix_pipeline = nullptr;
	OptixShaderBindingTable optix_sbt = {};

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
	optix::WadjetParameters launch_params;

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

	// Output buffer
	std::vector <uint32_t> color_buffer;

	// Data for rendering
	ImageData result_image = nullptr;
	BufferData result_buffer = nullptr;
	vk::raii::Sampler result_sampler = nullptr;

	// Functions
	static Wadjet make(const Context &);
};

// Other methods
void set_envmap(Wadjet &, const std::string &);
void capture(Wadjet &, std::vector <uint8_t> &);

void compute(Wadjet &, const ECS &, const Camera &, const Transform &, bool = false);

void render(Wadjet &,
	const CommandBuffer &,
	const Framebuffer &,
	const RenderArea & = {{-1, -1}, {-1, -1}});

}

}

#endif

#ifndef MERCURY_H_
#define MERCURY_H_

#include "global.hpp"

// Minimum sizes
#define INITIAL_OBJECTS		128UL
#define INITIAL_LIGHTS		128UL
#define INITIAL_MATERIALS	128UL

// Sizes of objects and lights
// are assumed to be the maximum
static const size_t MAX_OBJECT_SIZE = sizeof(Triangle);
static const size_t MAX_LIGHT_SIZE = sizeof(PointLight);

using namespace mercury;

// Print aligned_vec4
// TODO: common header
inline std::ostream& operator<<(std::ostream& os, const glm::vec4 &v)
{
	return (os << "(" << v.x << ", " << v.y
		<< ", " << v.z << ", " << v.w << ")");
}

inline std::ostream &operator<<(std::ostream &os, const aligned_vec4 &v)
{
	return (os << v.data);
}

// Print BoundingBox
inline std::ostream &operator<<(std::ostream &os, const mercury::BoundingBox &b)
{
	return (os << "(" << b.min << " --> " << b.max << ")");
}

// App
class MercuryApplication : public mercury::App {
	// TODO: some of these member should be moved back to App
	VkRenderPass			render_pass;
	VkCommandPool			command_pool;

	std::vector <VkCommandBuffer>	command_buffers;

	VkDescriptorPool		descriptor_pool;
	VkDescriptorSetLayout		descriptor_set_layout;
	VkDescriptorSet			descriptor_set;

	VkShaderModule			compute_shader;

	// Sync objects
	std::vector <VkFence>		in_flight_fences;
	std::vector <VkFence>		images_in_flight;

	std::vector <VkSemaphore>	smph_image_available;
	std::vector <VkSemaphore>	smph_render_finished;

	// Profiler
	Profiler		profiler;

	// BVH resources
	mercury::BVH bvh;

	// Copy buffer helper
	GPUWorld gworld;
	
	// GPU buffers
	BufferManager <uint>	_bf_pixels;
	BufferManager <uint8_t>	_bf_world;
	Buffer4f		_bf_objects;	// TODO: at some point change to aligned_uvec4
	Buffer4f		_bf_lights;
	Buffer4f		_bf_materials;
	Buffer4f		_bf_vertices;
	Buffer4m		_bf_transforms;
	Buffer4f		_bf_debug;

	// ImGui context and methods
	// TODO: the context should not have any sync objects
	Vulkan::ImGuiContext imgui_ctx;

	// TODO: wrap inside a struct
	bool capturing = false;
	Timer capture_timer;
	Capture capture;

	/////////////////////
	// Private methods //
	/////////////////////

	void maker(const Vulkan *, size_t);
	bool map_buffers(Vulkan *);
	void allocate_buffers();
	void dump_debug_data(Vulkan *vk);
	void make_profiler_tree(const Profiler::Frame &, float = -1.0);
	void make_imgui(size_t);
public:
	// Constructor
	MercuryApplication(Vulkan *);

	void update_world();
	void present();
	void frame() override;

	void update_command_buffers();
	void update_descriptor_set();

	// Desctiptor set layout bindings
	static const std::vector <VkDescriptorSetLayoutBinding> dsl_bindings;
};

// Profiler application
class ProfilerApplication : public mercury::App {

};


#endif
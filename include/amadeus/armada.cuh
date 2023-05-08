#ifndef KOBRA_AMADEUS_ARMADA_H_
#define KOBRA_AMADEUS_ARMADA_H_

// Standard headers
#include <atomic>
#include <map>
#include <optional>
#include <variant>
#include <vector>

// OptiX headers
#include <optix.h>
#include <optix_stubs.h>

// Engine headers
#include "include/backend.hpp"
#include "include/core/async.hpp"
#include "include/core/kd.cuh"
#include "include/daemons/mesh.hpp"
#include "include/daemons/transform.hpp"
#include "include/optix/parameters.cuh"
#include "include/optix/sbt.cuh"
#include "include/timer.hpp"
#include "include/vertex.hpp"
#include "accelerator.cuh"

namespace kobra {

// Forward declarations
class Camera;
class Transform;
class Renderable;

namespace amadeus {

// Aliases
using HitRecord = optix::Record <optix::Hit>;

// Forward declarations
class ArmadaRTX;

// Armada RTX basic launch parameters
struct ArmadaLaunchInfo {
	glm::vec2 resolution;

	struct Camera {
		glm::vec3 center;

		glm::vec3 ax_u;
		glm::vec3 ax_v;
		glm::vec3 ax_w;

		glm::mat4 projection;
		glm::mat4 view;
	} camera;

	struct {
		optix::QuadLight *quad_lights;
		optix::TriangleLight *tri_lights;

		int quad_count;
		int tri_count;
	} lights;

	// Output buffers (color + AOV)
	struct {
		glm::vec4 *color;
		glm::vec4 *normal;
		glm::vec4 *albedo;
		glm::vec4 *position;
	} buffers;

	// List of materials
	cuda::_material *materials;

	cudaTextureObject_t environment_map;
	bool has_environment_map;

	bool accumulate;
	float time;
	int max_depth;
	int samples;
};

// TODO: organize better, e.g. put into attachment method
// OptiX compilation configurations
static constexpr OptixPipelineCompileOptions ppl_compile_options = {
	.usesMotionBlur = false,
	.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_LEVEL_INSTANCING,
	.numPayloadValues = 2,
	.numAttributeValues = 0,
	.exceptionFlags = KOBRA_OPTIX_EXCEPTION_FLAGS,
	.pipelineLaunchParamsVariableName = "parameters",
	.usesPrimitiveTypeFlags = (unsigned int) OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE,
};

static constexpr OptixModuleCompileOptions module_options = {
	.optLevel = KOBRA_OPTIX_OPTIMIZATION_LEVEL,
	.debugLevel = KOBRA_OPTIX_DEBUG_LEVEL,
};

static constexpr OptixPipelineLinkOptions ppl_link_options = {
	.maxTraceDepth = 11,
};

// RTX attachment sub-layers
class AttachmentRTX {
	int m_hit_group_count;
public:
	// Constructor
	AttachmentRTX(int hit_group_count) : m_hit_group_count(hit_group_count) {}

	// Attaching and unloading
	virtual void attach(const ArmadaRTX &) = 0;

	virtual void load() = 0;
	virtual void unload() = 0;

	// Options; default no action
	using OptionValue = std::variant <int, float, bool, std::string>;

	virtual void set_option(const std::string &, const OptionValue &) {}

	virtual OptionValue get_option(const std::string &) const {
		return {};
	}

	// Rendering
	virtual void render(
		const ArmadaRTX *,
		const ArmadaLaunchInfo &,
		const std::optional <OptixTraversableHandle> &,
		std::vector <HitRecord> *,
		const vk::Extent2D &
	) = 0;

	// Friends
	friend class ArmadaRTX;
};

// Armada RTX for real-time, physically-based raytracing methods
class ArmadaRTX {
	// Raytracing backend
	std::shared_ptr <Accelerator> m_system;
	std::shared_ptr <daemons::MeshDaemon> m_mesh_memory;

	// Critical Vulkan structures
	vk::raii::Device *m_device = nullptr;
	vk::raii::PhysicalDevice *m_phdev = nullptr;

	// Texture loader
	TextureLoader *m_texture_loader = nullptr;

	// Vulkan structures
	vk::Extent2D m_extent = {0, 0};

	// Host buffer analogues
	// TODO: common algorithm for BVH construction...
	struct _instance_ref {
		const Transform *transform = nullptr;
		const Submesh *submesh = nullptr;
                int id = -1;

		bool operator <(const _instance_ref &other) const {
			if (transform != other.transform)
				return transform < other.transform;

			return submesh < other.submesh;
		}
	};

	struct {
		std::vector <optix::QuadLight> quad_lights;
		std::vector <optix::TriangleLight> tri_lights;
		std::vector <HitRecord> hit_records;

		std::set <_instance_ref> emissive_submeshes;
		std::map <const Submesh *, size_t> emissive_submesh_offsets;
		int emissive_count = 0;

		std::vector <cuda::_material> materials;
		std::vector <std::set <_instance_ref>> material_submeshes;

		std::vector <daemons::MeshDaemon::Cachelet> cachelets;

                std::vector <const Entity *> entities;
		std::vector <int> submesh_indices;
		// std::vector <const Submesh *> submeshes;
		// std::vector <const Transform *> submesh_transforms;

		// Update state for the hit records
                int total_meshes;
		long long int last_updated;
		std::map <std::string, long long int> times;
	} m_host;

	// Timer
	Timer m_timer;

	// Local launch parameters and results
	ArmadaLaunchInfo m_launch_info;

	// Attachment sub-layers
	std::map <std::string, std::shared_ptr <AttachmentRTX>> m_attachments;
	std::string m_active_attachment;
	std::string m_previous_attachment;

	// Acceleration structure state
	struct {
		long long int last_updated;
		std::map <std::string, long long int> times;
		bool null = true;

                OptixTraversableHandle handle;
                CUdeviceptr ptr;
	} m_tlas;

	// Private methods
	void update_triangle_light_buffers(
		const daemons::Transform *,
		const daemons::MaterialDaemon *,
		const std::set <_instance_ref> &
	);

	void update_sbt_data(const daemons::MaterialDaemon *);
	void update_materials(const daemons::MaterialDaemon *, const std::set <uint32_t> &);

	struct preprocess_update {
		std::optional <OptixTraversableHandle> handle;
		std::vector <HitRecord> *hit_records;
	};

	preprocess_update preprocess_scene(
                const std::vector <Entity> &,
                const daemons::Transform &,
                const daemons::MaterialDaemon *,
                const Camera &, const Transform &);
public:
	// Default constructor
	ArmadaRTX() = default;

	// Constructor
	ArmadaRTX(
		const Context &,
		const std::shared_ptr <Accelerator> &,
		const std::shared_ptr <daemons::MeshDaemon> &,
		const vk::Extent2D &
	);

	// Proprety methods
	size_t size() const {
		return m_extent.width * m_extent.height;
	}

	vk::Extent2D extent() const {
		return m_extent;
	}

	const std::shared_ptr <Accelerator> &system() const {
		return m_system;
	}

	std::vector <std::string> attachments() const {
		std::vector <std::string> names;
		for (const auto &[name, attachment] : m_attachments)
			names.push_back(name);
		return names;
	}

	std::string active_attachment() const {
		return m_active_attachment;
	}

	// Configure options
	void set_option(const std::string &field, const AttachmentRTX::OptionValue &value) {
		KOBRA_LOG_FUNC(Log::OK) << "Setting option " << field<< std::endl;
		auto attachment = m_attachments.at(m_active_attachment);
		attachment->set_option(field, value);
	}

	AttachmentRTX::OptionValue get_option(const std::string &field) const {
		auto attachment = m_attachments.at(m_active_attachment);
		return attachment->get_option(field);
	}

	// Buffer accessors
	glm::vec4 *color_buffer() {
		return m_launch_info.buffers.color;
	}

	glm::vec4 *normal_buffer() {
		return m_launch_info.buffers.normal;
	}

	glm::vec4 *albedo_buffer() {
		return m_launch_info.buffers.albedo;
	}

	glm::vec4 *position_buffer() {
		return m_launch_info.buffers.position;
	}

	// Attachment methods
	void attach(const std::string &name, const std::shared_ptr <AttachmentRTX> &attachment) {
		KOBRA_LOG_FUNC(Log::OK) << "Attaching attachment " << name << std::endl;
		m_attachments[name] = attachment;
		m_attachments[name]->attach(*this);
		m_host.times[name] = 0;
		m_tlas.times[name] = 0;
		m_active_attachment = name;
	}

	void activate(const std::string &name) {
		m_active_attachment = name;
	}

	// Set depth
	void set_depth(int depth) {
		m_launch_info.max_depth = depth;
	}

	// Environment map
	void set_envmap_enabled(bool enabled) {
		m_launch_info.has_environment_map = enabled;
	}

	// Methods
	void set_envmap(const std::string &);

	void render(const System *,
                const std::vector <Entity> &,
                const daemons::Transform &,
		const Camera &,
		const Transform &,
		bool = false);
};

}

}

#endif

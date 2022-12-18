#ifndef KOBRA_AMADEUS_ARMADA_H_
#define KOBRA_AMADEUS_ARMADA_H_

// Standard headers
#include <map>
#include <optional>
#include <vector>

// OptiX headers
#include <optix.h>
#include <optix_stubs.h>

// Engine headers
#include "system.cuh"
#include "../backend.hpp"
#include "../core/async.hpp"
#include "../core/kd.cuh"
#include "../optix/parameters.cuh"
#include "../optix/sbt.cuh"
#include "../timer.hpp"
#include "../vertex.hpp"
#include "../layers/mesh_memory.hpp"

namespace kobra {

// Forward declarations
class ECS;
class Camera;
class Transform;
class Renderable;

namespace amadeus {

// Forward declarations
class ArmadaRTX;

// Armada RTX basic launch parameters
struct ArmadaLaunchInfo {
	glm::vec2 resolution;
	
	struct {
		glm::vec3 center;
		glm::vec3 ax_u;
		glm::vec3 ax_v;
		glm::vec3 ax_w;
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
		glm::vec3 *normal;
		glm::vec3 *albedo;
		glm::vec3 *position;
	} buffers;

	cudaTextureObject_t environment_map;
	bool has_environment_map;

	int samples;
	bool accumulate;
	float time;
};

// RTX attachment sub-layers
class AttachmentRTX {
	int m_hit_group_count;
public:
	// Constructor
	AttachmentRTX(int hit_group_count) : m_hit_group_count(hit_group_count) {}

	// Attaching and unloading
	virtual void attach(const ArmadaRTX &) = 0;
	virtual void unload() = 0;

	// Rendering
	virtual void render(
		const ArmadaRTX *,
		const ArmadaLaunchInfo &,
		const std::optional <OptixTraversableHandle> &,
		const vk::Extent2D &
	) = 0;

	// Friends
	friend class ArmadaRTX;
};

// Armada RTX for real-time, physically-based raytracing methods
class ArmadaRTX {
	// Aliases
	using HitRecord = optix::Record <optix::Hit>;

	// Raytracing backend
	std::shared_ptr <System> m_system;
	std::shared_ptr <layers::MeshMemory> m_mesh_memory;

	// Critical Vulkan structures
	vk::raii::Device *m_device = nullptr;
	vk::raii::PhysicalDevice *m_phdev = nullptr;

	// Vulkan structures
	vk::Extent2D m_extent = {0, 0};

	// Host buffer analogues
	// TODO: common algorithm for BVH construction...
	struct {
		std::vector <optix::QuadLight> quad_lights;
		std::vector <optix::TriangleLight> tri_lights;
		std::vector <HitRecord> hit_records;
	} m_host;

	// Timer
	Timer m_timer;
	
	// Null state; whether any information has been loaded
	bool m_null_state = true;

	// Local launch parameters and results
	ArmadaLaunchInfo m_launch_info;

	// Attachment sub-layers
	std::map <std::string, std::shared_ptr <AttachmentRTX>> m_attachments;
	std::string m_active_attachment;

	// Acceleration structure state
	struct {
		long long int last_updated;
		std::map <std::string, long long int> times;
		bool null = true;
	} m_tlas;

	// Private methods
	void update_light_buffers(
		const std::vector <const Light *> &,
		const std::vector <const Transform *> &,
		const std::vector <const Submesh *> &,
		const std::vector <const Transform *> &
	);

	void update_sbt_data(
		const std::vector <layers::MeshMemory::Cachelet> &,
		const std::vector <const Submesh *> &,
		const std::vector <const Transform *> &
	);

	std::optional <OptixTraversableHandle>
	preprocess_scene(const ECS &, const Camera &, const Transform &);
public:
	// Default constructor
	ArmadaRTX() = default;

	// Constructor
	ArmadaRTX(
		const Context &,
		const std::shared_ptr <System> &,
		const std::shared_ptr <layers::MeshMemory> &,
		const vk::Extent2D &
	);

	// Proprety methods
	size_t size() {
		return m_extent.width * m_extent.height;
	}

	vk::Extent2D extent() {
		return m_extent;
	}

	const std::shared_ptr <System> &system() const {
		return m_system;
	}

	const std::vector <HitRecord> &hit_records() const {
		return m_host.hit_records;
	}
	
	// Buffer accessors
	glm::vec4 *color_buffer() {
		return m_launch_info.buffers.color;
	}

	glm::vec3 *normal_buffer() {
		return m_launch_info.buffers.normal;
	}

	glm::vec3 *albedo_buffer() {
		return m_launch_info.buffers.albedo;
	}

	glm::vec3 *position_buffer() {
		return m_launch_info.buffers.position;
	}

	// Attachment methods
	void attach(const std::string &name, const std::shared_ptr <AttachmentRTX> &attachment) {
		m_attachments[name] = attachment;
		m_tlas.times[name] = 0;
		activate(name);
	}

	void activate(const std::string &name) {
		// Unload previous attachment
		if (m_active_attachment.size() > 0)
			m_attachments[m_active_attachment]->unload();

		m_active_attachment = name;
		m_attachments[name]->attach(*this);
	}

	// Methods
	void set_envmap(const std::string &);

	void render(
		const ECS &,
		const Camera &,
		const Transform &,
		bool = false
	);
};

}

}

#endif

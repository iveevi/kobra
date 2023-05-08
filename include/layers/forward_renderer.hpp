#pragma once

// Standard headers
#include <map>
#include <optional>
#include <vector>

// Engine headers
#include "include/backend.hpp"
#include "include/lights.hpp"
#include "include/system.hpp"
#include "include/vertex.hpp"

namespace kobra {

// Forward declarations
class Camera;
class Transform;
class Renderable;

namespace layers {

class ForwardRenderer {
public:
	using RenderableDset = std::vector <vk::raii::DescriptorSet>;

	// Pipeline package
	struct PipelinePackage {
		vk::raii::Pipeline pipeline = nullptr;
		vk::raii::PipelineLayout ppl = nullptr;
		vk::raii::DescriptorSetLayout dsl = nullptr;

		// TODO: also an update function?
		// TODO: alias to simplify...
		std::function <void (const vk::raii::DescriptorSet &)> configure_dset = nullptr;

		std::map <const Renderable *, RenderableDset> dsets;
	};

	// Critical Vulkan structures
	vk::raii::Device *device = nullptr;
	vk::raii::PhysicalDevice *phdev = nullptr;
	vk::raii::DescriptorPool *descriptor_pool = nullptr;
	vk::raii::RenderPass render_pass = nullptr;

	// Texture loader
	TextureLoader *loader = nullptr;

	// Pipeline package maps
	std::map <std::string, PipelinePackage> pipeline_packages;

	// Constructors
	ForwardRenderer() = default;
	ForwardRenderer(const Context &);

	// Create a new pipeline package
	void add_pipeline(
		const std::string &,
		const std::string &,
		const std::vector <DescriptorSetLayoutBinding> & = {},
		const std::function <void (const vk::raii::DescriptorSet &)> & = nullptr
	);

	// Parameters for rendering
	struct Parameters {
                System *system = nullptr;
		std::vector <std::tuple <const Renderable *, const Transform *>> renderables;
		std::vector <std::tuple <const Light *, const Transform *>> lights;
		std::string pipeline_package = BUILTIN_PIPELINE_PACKAGE;
		std::string environment_map = "";
	};

	void render(const Parameters &,
		const Camera &,
		const Transform &,
		const vk::raii::CommandBuffer &,
		const vk::raii::Framebuffer &,
		const vk::Extent2D &,
		const RenderArea & = RenderArea::full()
	);
private:
	static constexpr char BUILTIN_PIPELINE_PACKAGE[] = "__builtin__";

	// Environment mapping resources
	struct {
		vk::raii::DescriptorSet dset = nullptr;
		vk::raii::DescriptorSetLayout dsl = nullptr;
		vk::raii::Pipeline pipeline = nullptr;
		vk::raii::PipelineLayout ppl = nullptr;

		BufferData vbo = nullptr;
		BufferData ibo = nullptr;

		std::string environment_map = "";

		bool initialized = false;
	} m_skybox;

	// Methods
	std::optional <PipelinePackage> make_pipline_package(
		const std::string &,
		const std::string &,
		const std::vector <DescriptorSetLayoutBinding> &,
		const std::function <void (const vk::raii::DescriptorSet &)> &
	);

	RenderableDset make_renderable_dset(PipelinePackage &, uint32_t);

	void configure_renderable_dset(
                const System *,
		const PipelinePackage &,
		const ForwardRenderer::RenderableDset &,
		const Renderable *
	);

	void configure_environment_map(const std::string &);
};

}

}

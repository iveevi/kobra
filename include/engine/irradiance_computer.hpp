#ifndef KOBRA_ENGINE_IRRADIANCE_COMPUTER_H_
#define KOBRA_ENGINE_IRRADIANCE_COMPUTER_H_

// Engine headers
#include "../backend.hpp"
#include "../shader_program.hpp"

namespace kobra {

namespace engine {

class IrradianceComputer {
public:
	size_t mips;
	std::vector <kobra::ImageData> irradiance_maps;	

	IrradianceComputer() = default;
	IrradianceComputer(int);
	
	void bind(const vk::raii::Device &, const vk::raii::DescriptorSet &, uint32_t);

	vk::raii::Fence compute(const kobra::Context &, const kobra::ImageData &);
private:
	// Cached Vulkan structures
	vk::raii::CommandBuffer m_command_buffer = nullptr;
	vk::raii::DescriptorSets m_irradiance_dsets = nullptr;
	vk::raii::Pipeline m_irradiance_pipeline = nullptr;
	vk::raii::PipelineLayout m_irradiance_ppl = nullptr;
	vk::raii::Sampler m_environment_sampler = nullptr;

	// Cached samplers
	std::vector <vk::raii::Sampler> m_samplers;
};

}

}

#endif

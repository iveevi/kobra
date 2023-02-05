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
	size_t samples;
	std::vector <kobra::ImageData> irradiance_maps;	

	IrradianceComputer() = default;
	IrradianceComputer(const Context &, const ImageData &, int, int);
	
	void bind(const vk::raii::Device &, const vk::raii::DescriptorSet &, uint32_t);

	void sample(const vk::raii::CommandBuffer &);
private:
	// Cached Vulkan structures
	vk::raii::CommandBuffer m_command_buffer = nullptr;
	vk::raii::DescriptorSets m_irradiance_dsets = nullptr;
	vk::raii::Pipeline m_irradiance_pipeline = nullptr;
	vk::raii::PipelineLayout m_irradiance_ppl = nullptr;
	vk::raii::Sampler m_environment_sampler = nullptr;

	// Cached samplers
	std::vector <vk::raii::Sampler> m_samplers;

	// Weight buffers
	std::vector <BufferData> m_weight_buffers;

	// Sparse sampling for sane performance
	int m_max_samples;
	int m_sparsity;
	int m_sparsity_index;
};

}

}

#endif

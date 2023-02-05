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
	bool cached;
	std::vector <const kobra::ImageData *> irradiance_maps;	

	IrradianceComputer() = default;
	IrradianceComputer(const Context &, const ImageData &, int, int, const std::string & = "");
	
	void bind(const vk::raii::Device &, const vk::raii::DescriptorSet &, uint32_t);
	void save_irradiance_maps(const Context &, const std::string &);

	bool sample(const vk::raii::CommandBuffer &);
private:
	// Cached Vulkan structures
	vk::raii::CommandBuffer m_command_buffer = nullptr;
	vk::raii::DescriptorSet m_irradiance_dset = nullptr;
	vk::raii::Pipeline m_irradiance_pipeline = nullptr;
	vk::raii::PipelineLayout m_irradiance_ppl = nullptr;
	vk::raii::Sampler m_environment_sampler = nullptr;

	// Cached samplers
	std::vector <vk::raii::Sampler> m_samplers;
	vk::Extent2D m_extent;

	// Weight buffer
	BufferData m_weight_buffer = nullptr;

	// Sparse sampling for sane performance
	int m_max_samples;
	int m_sparsity;
	int m_sparsity_index;
	
	// Locally stored images
	std::vector <kobra::ImageData> m_irradiance_maps;	
};

}

}

#endif

#include "../../include/engine/irradiance_computer.hpp"

namespace kobra {

namespace engine {

struct Irradiance_PushConstants {
	float roughness;
	int samples;
	int width;
	int height;
	int sparsity;
	int sparsity_index;
	int max_samples;
};

static const std::vector <kobra::DescriptorSetLayoutBinding>
	IRRADIANCE_COMPUTER_LAYOUT_BINDINGS {
	{
		0, vk::DescriptorType::eCombinedImageSampler,
		1, vk::ShaderStageFlagBits::eCompute
	},
	{
		1, vk::DescriptorType::eStorageImage,
		1, vk::ShaderStageFlagBits::eCompute
	},
	{
		2, vk::DescriptorType::eStorageBuffer,
		1, vk::ShaderStageFlagBits::eCompute
	}
};

IrradianceComputer::IrradianceComputer(const Context &context, const ImageData &environment_map, int _mips, int max)
		: mips(_mips), samples(0), m_max_samples(max), m_sparsity(3), m_sparsity_index(0)
{
	const vk::raii::PhysicalDevice &phdev = *context.phdev;
	const vk::raii::Device &device = *context.device;
	const vk::raii::CommandPool &command_pool = *context.command_pool;
	const vk::raii::DescriptorPool &descriptor_pool = *context.descriptor_pool;

	// IRRADIANCE MIP MAP CREATION...
	// Load the compute shader
	const std::string IRRADIANCE_COMPUTER_SHADER = KOBRA_DIR "/source/shaders/irradiance.glsl";
	const std::string IRRADIANCE_COMPUTER_SOURCE = kobra::common::read_file(IRRADIANCE_COMPUTER_SHADER);

	kobra::ShaderProgram irradiance_computer {
		IRRADIANCE_COMPUTER_SOURCE,
		vk::ShaderStageFlagBits::eCompute
	};

	vk::raii::ShaderModule opt_irradiance_computer = std::move(*irradiance_computer.compile(device));

	// Create a compute pipeline
	// TODO: create a class to make this much easier...
	vk::raii::DescriptorSetLayout irradiance_dsl =
		kobra::make_descriptor_set_layout(
			device, IRRADIANCE_COMPUTER_LAYOUT_BINDINGS
		);

	std::vector <vk::DescriptorSetLayout> irradiance_dsls
		(mips, *irradiance_dsl);

	m_irradiance_dsets =
		vk::raii::DescriptorSets {
			device, {
				*descriptor_pool,
				irradiance_dsls
			}
		};

	vk::PushConstantRange irradiance_pcr {
		vk::ShaderStageFlagBits::eCompute,
		0, sizeof(Irradiance_PushConstants)
	};

	m_irradiance_ppl = {
		device,
		{{}, *irradiance_dsl, irradiance_pcr}
	};

	m_irradiance_pipeline = {
		device,
		nullptr,
		vk::ComputePipelineCreateInfo {
			{},
			vk::PipelineShaderStageCreateInfo {
				{},
				vk::ShaderStageFlagBits::eCompute,
				*opt_irradiance_computer,
				"main"
			},
			*m_irradiance_ppl
		}
	};

	m_environment_sampler = kobra::make_sampler(device, environment_map);

	// Create destination images
	uint32_t width = environment_map.extent.width;
	uint32_t height = environment_map.extent.height;

	for (int i = 0; i < mips; i++) {
		irradiance_maps.emplace_back(
			phdev, device,
			vk::Format::eR32G32B32A32Sfloat,
			vk::Extent2D {width, height},
			vk::ImageTiling::eOptimal,
			vk::ImageUsageFlagBits::eStorage | vk::ImageUsageFlagBits::eSampled,
			vk::MemoryPropertyFlagBits::eDeviceLocal,
			vk::ImageAspectFlagBits::eColor
		);

		m_weight_buffers.emplace_back(
			phdev, device,
			4 * sizeof(float) * width * height,
			vk::BufferUsageFlagBits::eStorageBuffer,
			vk::MemoryPropertyFlagBits::eHostVisible
				| vk::MemoryPropertyFlagBits::eHostCoherent
		);
	}

	// Make sure all the images are in the right layout
	kobra::submit_now(device, vk::raii::Queue {device, 0, 0}, command_pool,
		[&](const vk::raii::CommandBuffer &cmd) {
			for (auto &irradiance_map : irradiance_maps)
				irradiance_map.transition_layout(cmd, vk::ImageLayout::eShaderReadOnlyOptimal);
		}
	);

	// Bind the images to the descriptor sets
	for (int i = 0; i < mips; i++) {
		std::array <vk::DescriptorImageInfo, 2> image_infos {
			vk::DescriptorImageInfo {
				*m_environment_sampler,
				*environment_map.view,
				vk::ImageLayout::eShaderReadOnlyOptimal
			},
			vk::DescriptorImageInfo {
				nullptr,
				*irradiance_maps[i].view,
				vk::ImageLayout::eGeneral
			}
		};

		std::vector <vk::DescriptorBufferInfo> buffer_infos {
			vk::DescriptorBufferInfo {
				*m_weight_buffers[i].buffer,
				0, VK_WHOLE_SIZE
			}
		};

		std::array <vk::WriteDescriptorSet, 3> writes {
			vk::WriteDescriptorSet {
				*m_irradiance_dsets[i],
				0, 0, 1,
				vk::DescriptorType::eCombinedImageSampler,
				&image_infos[0]
			},
			vk::WriteDescriptorSet {
				*m_irradiance_dsets[i],
				1, 0, 1,
				vk::DescriptorType::eStorageImage,
				&image_infos[1]
			},
			vk::WriteDescriptorSet {
				*m_irradiance_dsets[i],
				2, 0, 1,
				vk::DescriptorType::eStorageBuffer,
				nullptr,&buffer_infos[0]
			}
		};

		device.updateDescriptorSets(writes, {});
	}
}

void IrradianceComputer::bind(const vk::raii::Device &device, const vk::raii::DescriptorSet &dset, uint32_t binding)
{
	// First create the samplers
	if (m_samplers.size() != mips) {
		m_samplers.clear();

		for (int i = 0; i < mips; i++) {
			m_samplers.emplace_back(
				kobra::make_sampler(device, irradiance_maps[i])
			);
		}
	}

	// Create image descrtiptors
	std::vector <vk::DescriptorImageInfo> image_infos;
	for (int i = 0; i < mips; i++) {
		image_infos.emplace_back(
			*m_samplers[i],
			*irradiance_maps[i].view,
			vk::ImageLayout::eShaderReadOnlyOptimal
		);
	}

	// Create the descriptor write
	vk::WriteDescriptorSet irradiance_dset_write {
		*dset,
		binding, 0, (uint32_t) image_infos.size(),
		vk::DescriptorType::eCombinedImageSampler,
		image_infos.data()
	};

	// Update the descriptor set
	device.updateDescriptorSets(irradiance_dset_write, nullptr);
}

void IrradianceComputer::sample(const vk::raii::CommandBuffer &cmd)
{
	if (samples > m_max_samples)
		return;

	for (auto &irradiance_map : irradiance_maps)
		irradiance_map.transition_layout(cmd, vk::ImageLayout::eGeneral);

	// TODO: one kernel, all mips, share the random numbers
	// but compute all samples idependently... (keep sample
	// count high...)
	for (int i = 0; i < mips; i++) {
		if (samples > 0 && i == 0)
			continue;

		Irradiance_PushConstants push_constants {
			i/float(mips - 1),
			int(samples),
			int(irradiance_maps[i].extent.width),
			int(irradiance_maps[i].extent.height),
			m_sparsity,
			m_sparsity_index,
			m_max_samples
		};

		cmd.bindPipeline(
			vk::PipelineBindPoint::eCompute,
			*m_irradiance_pipeline
		);

		cmd.pushConstants <Irradiance_PushConstants> (
			*m_irradiance_ppl,
			vk::ShaderStageFlagBits::eCompute,
			0, push_constants
		);

		cmd.bindDescriptorSets(
			vk::PipelineBindPoint::eCompute,
			*m_irradiance_ppl,
			0, {*m_irradiance_dsets[i]}, {}
		);

		cmd.dispatch(
			irradiance_maps[i].extent.width,
			irradiance_maps[i].extent.height,
			1
		);
	}

	for (auto &irradiance_map : irradiance_maps)
		irradiance_map.transition_layout(cmd, vk::ImageLayout::eShaderReadOnlyOptimal);

	m_sparsity_index++;
	if (m_sparsity_index >= m_sparsity) {
		m_sparsity_index = 0;
		samples++;
	}
}

}

}

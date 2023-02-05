// Standard headers
#include <stb/stb_image_write.h>

// Engine headers
#include "../../include/engine/irradiance_computer.hpp"

namespace kobra {

namespace engine {

struct Irradiance_PushConstants {
	int samples;
	int width;
	int height;
	int sparsity;
	int sparsity_index;
	int max_samples;
};

IrradianceComputer::IrradianceComputer
		(const Context &context, const ImageData &environment_map,
		int _mips, int max, const std::string &cache_prefix)
		: mips(_mips), samples(0), cached(false),
		m_max_samples(max), m_sparsity(10),
		m_sparsity_index(0),
		m_extent(environment_map.extent)
{
	// TODO: read from a metadata file with records number of samples...
	// For now assume if the cache exists, it has the max number of samples

	// TODO: caching system
	/* Check if the cache exists
	if (!cache_prefix.empty()) {
		for (int i = 0; i < mips; i++) {
			std::string ext = "_" + std::to_string(i) + "_" + std::to_string(mips) + ".png";
			std::string filename = cache_prefix + ext;

			std::cout << "Does " << filename << " exist? "
				<< std::boolalpha << std::filesystem::exists(filename) << std::endl;

			const ImageData &image = context.texture_loader->load_texture(filename);
			irradiance_maps.push_back(&image);
		}

		samples = m_max_samples;
		cached = true;
		return;
	} */

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

	vk::raii::ShaderModule opt_irradiance_computer = std::move(
		*irradiance_computer.compile(
			device,
			{
				{"MIP_MAPS", std::to_string(mips)},
			}
		)
	);

	// Create a compute pipeline
	// TODO: create a class to make this much easier...
	const std::vector <kobra::DescriptorSetLayoutBinding> bindings {
		{
			0, vk::DescriptorType::eCombinedImageSampler,
			1, vk::ShaderStageFlagBits::eCompute
		},
		{
			1, vk::DescriptorType::eStorageImage,
			mips, vk::ShaderStageFlagBits::eCompute
		},
		{
			2, vk::DescriptorType::eStorageBuffer,
			1, vk::ShaderStageFlagBits::eCompute
		}
	};

	vk::raii::DescriptorSetLayout irradiance_dsl =
		kobra::make_descriptor_set_layout(
			device, bindings
		);

	std::vector <vk::DescriptorSetLayout> irradiance_dsls
		{ *irradiance_dsl };

	m_irradiance_dset = std::move(
		vk::raii::DescriptorSets {
			device, {
				*descriptor_pool,
				irradiance_dsls
			}
		}.front()
	);

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
		m_irradiance_maps.emplace_back(
			phdev, device,
			vk::Format::eR32G32B32A32Sfloat,
			vk::Extent2D {width, height},
			vk::ImageTiling::eOptimal,
			vk::ImageUsageFlagBits::eStorage
				| vk::ImageUsageFlagBits::eSampled
				| vk::ImageUsageFlagBits::eTransferSrc,
			vk::MemoryPropertyFlagBits::eDeviceLocal,
			vk::ImageAspectFlagBits::eColor
		);
	}

	// Save to the public images
	irradiance_maps.resize(mips);
	for (int i = 0; i < mips; i++)
		irradiance_maps[i] = &m_irradiance_maps[i];

	// TODO: this takes a LOT of memory... make a version that doesnt
	// require additional buffers to cache the weights...
	m_weight_buffer = BufferData {
		phdev, device,
		4 * mips * sizeof(float) * width * height,
		vk::BufferUsageFlagBits::eStorageBuffer,
		vk::MemoryPropertyFlagBits::eHostVisible
			| vk::MemoryPropertyFlagBits::eHostCoherent
	};

	// Make sure all the images are in the right layout
	kobra::submit_now(device, vk::raii::Queue {device, 0, 0}, command_pool,
		[&](const vk::raii::CommandBuffer &cmd) {
			for (auto &irradiance_map : m_irradiance_maps)
				irradiance_map.transition_layout(cmd, vk::ImageLayout::eShaderReadOnlyOptimal);
		}
	);

	std::vector <vk::DescriptorImageInfo> image_infos {
		vk::DescriptorImageInfo {
			*m_environment_sampler,
			*environment_map.view,
			vk::ImageLayout::eShaderReadOnlyOptimal
		}
	};

	for (int i = 0; i < mips; i++) {
		image_infos.emplace_back(
			nullptr,
			*m_irradiance_maps[i].view,
			vk::ImageLayout::eGeneral
		);
	}

	std::vector <vk::DescriptorBufferInfo> buffer_infos {
		vk::DescriptorBufferInfo {
			*m_weight_buffer.buffer,
			0, VK_WHOLE_SIZE
		}
	};

	std::array <vk::WriteDescriptorSet, 3> writes {
		vk::WriteDescriptorSet {
			*m_irradiance_dset,
			0, 0, 1,
			vk::DescriptorType::eCombinedImageSampler,
			&image_infos[0]
		},
		vk::WriteDescriptorSet {
			*m_irradiance_dset,
			1, 0, uint32_t(mips),
			vk::DescriptorType::eStorageImage,
			&image_infos[1]
		},
		vk::WriteDescriptorSet {
			*m_irradiance_dset,
			2, 0, 1,
			vk::DescriptorType::eStorageBuffer,
			nullptr, &buffer_infos[0]
		}
	};

	device.updateDescriptorSets(writes, {});
}

void IrradianceComputer::bind(const vk::raii::Device &device, const vk::raii::DescriptorSet &dset, uint32_t binding)
{
	// First create the samplers
	if (m_samplers.size() != mips) {
		m_samplers.clear();

		for (int i = 0; i < mips; i++) {
			m_samplers.emplace_back(
				kobra::make_sampler(device, *irradiance_maps[i])
			);
		}
	}

	// Create image descrtiptors
	std::vector <vk::DescriptorImageInfo> image_infos;
	for (int i = 0; i < mips; i++) {
		image_infos.emplace_back(
			*m_samplers[i],
			*irradiance_maps[i]->view,
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

void IrradianceComputer::save_irradiance_maps(const Context &context, const std::string &prefix)
{
	// Create a staging buffer
	BufferData staging_buffer {
		*context.phdev, *context.device,
		m_extent.width * m_extent.height * sizeof(float) * 4,
		vk::BufferUsageFlagBits::eTransferDst,
		vk::MemoryPropertyFlagBits::eHostVisible
			| vk::MemoryPropertyFlagBits::eHostCoherent
	};

	KOBRA_LOG_FUNC(Log::INFO) << "Saving irradiance maps to " << prefix << "*" << std::endl;

	for (int i = 0; i < mips; i++) {
		// TODO: save as HDR
		std::string ext = "_" + std::to_string(i) + "_" + std::to_string(mips) + ".png";
		std::string filename = prefix + ext;

		// Download the image to the buffer
		vk::BufferImageCopy copy {
			0, 0, 0,
			{vk::ImageAspectFlagBits::eColor, 0, 0, 1},
			{0, 0, 0},
			{m_extent.width, m_extent.height, 1}
		};

		// Transition the image to transfer source
		vk::raii::Queue queue {*context.device, 0, 0};
		kobra::submit_now(*context.device, queue, *context.command_pool,
			[&](const vk::raii::CommandBuffer &cmd) {
				m_irradiance_maps[i].transition_layout(
					cmd, vk::ImageLayout::eTransferSrcOptimal
				);

				cmd.copyImageToBuffer(
					*m_irradiance_maps[i].image,
					vk::ImageLayout::eTransferSrcOptimal,
					*staging_buffer.buffer,
					copy
				);

				m_irradiance_maps[i].transition_layout(
					cmd, vk::ImageLayout::eShaderReadOnlyOptimal
				);
			}
		);

		// Map the buffer
		std::vector <float> data(m_extent.width * m_extent.height * 4);
		staging_buffer.download(data);

		// Convert to 8-bit per channel
		std::vector <uint8_t> data8(data.size());
		for (int i = 0; i < data.size(); i++)
			data8[i] = uint8_t(data[i] * 255.0f);

		std::cout << "\tDownloading " << filename << std::endl;

		stbi_flip_vertically_on_write(true);
		stbi_write_png(filename.c_str(),
			m_extent.width, m_extent.height, 4, data8.data(),
			m_extent.width * 4
		);
	}
}

bool IrradianceComputer::sample(const vk::raii::CommandBuffer &cmd)
{
	if (samples > m_max_samples)
		return true;

	for (auto &irradiance_map : m_irradiance_maps)
		irradiance_map.transition_layout(cmd, vk::ImageLayout::eGeneral);

	Irradiance_PushConstants push_constants {
		int(samples),
		int(irradiance_maps[0]->extent.width),
		int(irradiance_maps[0]->extent.height),
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
		0, {*m_irradiance_dset}, {}
	);

	cmd.dispatch(
		irradiance_maps[0]->extent.width,
		irradiance_maps[0]->extent.height,
		1
	);

	for (auto &irradiance_map : m_irradiance_maps)
		irradiance_map.transition_layout(cmd, vk::ImageLayout::eShaderReadOnlyOptimal);

	m_sparsity_index++;
	if (m_sparsity_index >= m_sparsity) {
		m_sparsity_index = 0;
		samples++;
	}

	return false;
}

}

}

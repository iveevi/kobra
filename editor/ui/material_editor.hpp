#pragma once

// ImGui headers
#include <imgui.h>

// Engine headers
#include "include/backend.hpp"
#include "include/ui/attachment.hpp"
#include "include/daemons/material.hpp"

// Local headers
#include "editor/common.hpp"
#include "editor/material_preview.hpp"

// Material editor UI attachment
class MaterialEditor : public ui::ImGuiAttachment {
	int m_prev_material_index = -1;

        vk::Sampler sampler;
        vk::DescriptorSet dset_material_preview;

	vk::DescriptorSet m_diffuse_set;
	vk::DescriptorSet m_normal_set;

	glm::vec3 emission_base = glm::vec3(0.0f);
	float emission_strength = 0.0f;

	// Editor *m_editor = nullptr;
	kobra::TextureLoader *m_texture_loader = nullptr;

	vk::DescriptorSet imgui_allocate_image(const std::string &path) {
		const kobra::ImageData &image = m_texture_loader->load_texture(path);
		const vk::raii::Sampler &sampler = m_texture_loader->load_sampler(path);

		return ImGui_ImplVulkan_AddTexture(
			static_cast <VkSampler> (*sampler),
			static_cast <VkImageView> (*image.view),
			static_cast <VkImageLayout> (image.layout)
		);
	}
public:
        vk::Device dev;

        MaterialDaemon *md = nullptr;
        MaterialPreview *mp = nullptr;
	int material_index = -1;

	MaterialEditor() = delete;
	MaterialEditor(const vk::Device &dev_, MaterialPreview *mp_, TextureLoader *texture_loader, MaterialDaemon *md_)
			: dev { dev_ }, mp { mp_ }, m_texture_loader { texture_loader }, md { md_ } {
                // Allocate the material preview descriptor set
                vk::SamplerCreateInfo sampler_info {};
                sampler_info.magFilter = vk::Filter::eLinear;
                sampler_info.minFilter = vk::Filter::eLinear;
                sampler_info.mipmapMode = vk::SamplerMipmapMode::eLinear;
                sampler_info.addressModeU = vk::SamplerAddressMode::eRepeat;
                sampler_info.addressModeV = vk::SamplerAddressMode::eRepeat;
                sampler_info.addressModeW = vk::SamplerAddressMode::eRepeat;
                sampler_info.mipLodBias = 0.0f;
                sampler_info.anisotropyEnable = VK_FALSE;
                sampler_info.maxAnisotropy = 16.0f;
                sampler_info.compareEnable = VK_FALSE;
                sampler_info.compareOp = vk::CompareOp::eNever;
                sampler_info.minLod = 0.0f;
                sampler_info.maxLod = 0.0f;
                sampler_info.borderColor = vk::BorderColor::eFloatOpaqueWhite;
                sampler_info.unnormalizedCoordinates = VK_FALSE;

                sampler = dev.createSampler(sampler_info);
                dset_material_preview = ImGui_ImplVulkan_AddTexture(
                        static_cast <VkSampler> (sampler),
                        static_cast <VkImageView> (mp->display.view),
                        static_cast <VkImageLayout> (vk::ImageLayout::eGeneral)
                );
        }

        ~MaterialEditor() {
                dev.destroySampler(sampler);
        }

	void render() override;
};

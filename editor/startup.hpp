#pragma once

// Engine headers
#include "api.hpp"
#include "app.hpp"
#include "include/app.hpp"

struct Startup : kobra::BaseApp {
        std::vector <std::string> projects;
        std::vector <bool> verified;

        char project_name[256] = { 0 };
        char project_path[256] = { 0 };

        api::Image default_thumbnail;
        // vk::DescriptorSet default_thumbnail_set;
        
        std::vector <vk::DescriptorSet> thumbnail_sets;

        kobra::TextureLoader *texture_loader = nullptr;

        Startup(const vk::raii::PhysicalDevice &, const std::vector <const char *> &);
        ~Startup();

	void record(const vk::raii::CommandBuffer &, const vk::raii::Framebuffer &) override;
};

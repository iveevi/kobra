// Standard headers
#include <filesystem>
#include <fstream>
#include <vector>

// ImGui headers
#include <imgui.h>

#include <imgui/backends/imgui_impl_glfw.h>
#include <imgui/backends/imgui_impl_vulkan.h>

// File dialog headers
#include <nfd.h>

// Engine headers
#include "editor/common.hpp"
#include "include/backend.hpp"
#include "include/image.hpp"
#include "include/project.hpp"

// Editor headers
#include "startup.hpp"

// Location of configuration file
constexpr const char *CONFIG_FILE = ".config/kobra/projects";

std::vector <std::string> load_config()
{
        // Check if file exists
        std::string home { std::getenv("HOME") };
        std::cout << "Home directory: " << home << std::endl;

        std::filesystem::path config_path { home };
        config_path /= CONFIG_FILE;

        std::cout << "Searching for configuration file: " << config_path << std::endl;

        if (!std::filesystem::exists(config_path)) {
                std::cerr << "Configuration file not found: " << CONFIG_FILE << std::endl;
                std::cerr << "Creating new configuration file..." << std::endl;

                // Create the directory if it doesn't exist
                // NOTE: If the .config directory doesn't exist, then terminate
                std::filesystem::path config_dir { config_path.parent_path() };
                std::cout << "Creating directory: " << config_dir << std::endl;
                std::cout << "Full path: " << std::filesystem::weakly_canonical(config_dir) << std::endl;

                std::filesystem::create_directories(config_dir);

                std::cout << "Directory created" << std::endl;
                std::cout << "Check: " << std::filesystem::exists(config_dir) << std::endl;

                // Create file and return empty vector
                std::ofstream config_file { config_path };
                config_file << "# Kobra project configuration file" << std::endl;
                config_file << "# Each line is a project path" << std::endl;
                config_file.close();

                return {};
        }

        std::cerr << "Configuration file found: " << CONFIG_FILE << std::endl;

        // Read file: each line is a project path
        std::ifstream config_file { config_path };

        std::vector <std::string> projects;
        std::string line;
        while (std::getline(config_file, line)) {
                if (line[0] == '#')
                        continue;
                projects.push_back(line);
        }

        return projects;
}

void append_to_config(const std::filesystem::path &dir)
{
        std::string home { std::getenv("HOME") };
        std::filesystem::path config_path { home };
        config_path /= CONFIG_FILE;

        std::ofstream config_file { config_path, std::ios_base::app };
        config_file << dir.string() << std::endl;
        config_file.close();
}

std::vector <bool> vertify_projects(const std::vector <std::string> &projects)
{
        std::vector <bool> verified;
        verified.reserve(projects.size());

        for (const auto &project : projects) {
                std::filesystem::path project_path { project };
                verified.push_back(std::filesystem::exists(project_path));

                // TODO: also check if each project's scene file exists
        }

        return verified;
}

Startup::Startup(const vk::raii::PhysicalDevice &phdev,
                const std::vector <const char *> &extensions)
                : kobra::BaseApp {
                        phdev, "Kobra Project Startup",
                        vk::Extent2D {1000, 1000},
                        extensions
                }
{
	ImGui::CreateContext();
	ImGui_ImplGlfw_InitForVulkan(window->handle, true);

        // Create dedicated Vulkan resources for ImGUI
        std::vector <vk::DescriptorPoolSize> pool_sizes {
                vk::DescriptorPoolSize(vk::DescriptorType::eSampler, 1000),
                vk::DescriptorPoolSize(vk::DescriptorType::eCombinedImageSampler, 1000),
                vk::DescriptorPoolSize(vk::DescriptorType::eSampledImage, 1000),
                vk::DescriptorPoolSize(vk::DescriptorType::eStorageImage, 1000),
                vk::DescriptorPoolSize(vk::DescriptorType::eUniformTexelBuffer, 1000),
                vk::DescriptorPoolSize(vk::DescriptorType::eStorageTexelBuffer, 1000),
                vk::DescriptorPoolSize(vk::DescriptorType::eUniformBuffer, 1000),
                vk::DescriptorPoolSize(vk::DescriptorType::eStorageBuffer, 1000),
                vk::DescriptorPoolSize(vk::DescriptorType::eUniformBufferDynamic, 1000),
                vk::DescriptorPoolSize(vk::DescriptorType::eStorageBufferDynamic, 1000),
                vk::DescriptorPoolSize(vk::DescriptorType::eInputAttachment, 1000)
        };

        // TODO: pass max sets as a parameter
        descriptor_pool = kobra::make_descriptor_pool(device, pool_sizes);

        ImGui_ImplVulkan_InitInfo init_info = {};
        init_info.Instance = *kobra::get_vulkan_instance();
        init_info.PhysicalDevice = *phdev;
        init_info.Device = *device;
        init_info.Queue = *graphics_queue;
        init_info.DescriptorPool = *descriptor_pool;
        init_info.MinImageCount = 3;
        init_info.ImageCount = 3;
        init_info.MSAASamples = VK_SAMPLE_COUNT_1_BIT; // TODO: pass as a parameter
        
        ImGui_ImplVulkan_Init(&init_info, *render_pass);
        
        // Load font
        ImGuiIO &io = ImGui::GetIO();
        io.Fonts->AddFontFromFileTTF(KOBRA_FONTS_DIR "/Frutiger/Frutiger.ttf", 18.0f);

        // Enable docking
        io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;
        
        vk::raii::Queue temp_queue {device, 0, 0};
        kobra::submit_now(device, temp_queue, command_pool,
                [&](const vk::raii::CommandBuffer &cmd) {
                        ImGui_ImplVulkan_CreateFontsTexture(*cmd);
                }
        );

        // Destroy CPU-side resources
        ImGui_ImplVulkan_DestroyFontUploadObjects();

        // Load projects
        projects = load_config();
        verified = vertify_projects(projects);

        // Descriptor set for default thumbnail
        // kobra::RawImage thumbail_source = kobra::load_texture(KOBRA_DIR "/kobra_icon.png");
        //
        // vk::PhysicalDeviceMemoryProperties mem_props = phdev.getMemoryProperties();
        //
        // vk::Format format = (thumbail_source.type == kobra::RawImage::RGBA_32_F) ?
        //         vk::Format::eR32G32B32A32Sfloat : vk::Format::eR8G8B8A8Unorm;
        //
        // default_thumbnail = api::make_image(*device, {
        //         .width = thumbail_source.width,
        //         .height = thumbail_source.height,
        //         .format = format,
        //         .usage = vk::ImageUsageFlagBits::eSampled
        //                 | vk::ImageUsageFlagBits::eTransferDst,
        // }, mem_props);

        texture_loader = new kobra::TextureLoader(get_device());

        // TODO: should be make texture, make sampler, etc...
        const kobra::ImageData &thumbnail = texture_loader->load_texture(KOBRA_DIR "/kobra_icon.png", false);
        const vk::raii::Sampler &sampler = texture_loader->load_sampler(KOBRA_DIR "/kobra_icon.png");
        // vk::raii::Sampler sampler = kobra::make_continuous_sampler(device);

        /* default_thumbnail_set = ImGui_ImplVulkan_AddTexture(
                static_cast <VkSampler> (*sampler),
                static_cast <VkImageView> (*thumbnail.view),
                static_cast <VkImageLayout> (vk::ImageLayout::eShaderReadOnlyOptimal)
        ); */

        for (const auto &project : projects) {
                vk::DescriptorSet set = ImGui_ImplVulkan_AddTexture(
                        static_cast <VkSampler> (*sampler),
                        static_cast <VkImageView> (*thumbnail.view),
                        static_cast <VkImageLayout> (vk::ImageLayout::eShaderReadOnlyOptimal)
                );

                thumbnail_sets.push_back(set);
        }
        
        // Set the ImGui style
        set_imgui_theme();
}

Startup::~Startup()
{
        std::cout << "Destroying startup app" << std::endl;
        delete texture_loader;

        std::cout << "Destroying ImGui resources" << std::endl;

        ImGui_ImplVulkan_Shutdown();
        ImGui_ImplGlfw_Shutdown();
        ImGui::DestroyContext();
}

std::string project_parent_directory()
{
	nfdchar_t *path = nullptr;

        // TODO: default project directory should be configurable
        std::string default_path = std::string(getenv("HOME"));
        nfdresult_t result = NFD_PickFolder(&path, default_path.c_str());

        if (result == NFD_OKAY) {
                std::cout << "Project directory: " << path << std::endl;
                return std::string(path);
        } else if (result == NFD_CANCEL) {
                std::cout << "User canceled dialog" << std::endl;
        } else {
                std::cout << "Error: " << NFD_GetError() << std::endl;
        }

        return "";
}

void Startup::record(const vk::raii::CommandBuffer &cmd, const vk::raii::Framebuffer &fb)
{
        cmd.begin({});

        // Apply the render area
        kobra::RenderArea::full().apply(cmd, window->extent);

        // Start render pass
        std::array <vk::ClearValue, 2> clear_values {
                vk::ClearValue {
                        vk::ClearColorValue {
                                std::array <float, 4> {0.0f, 0.0f, 0.0f, 1.0f}
                        }
                },
                vk::ClearValue {
                        vk::ClearDepthStencilValue {
                                1.0f, 0
                        }
                }
        };

        cmd.beginRenderPass(
                vk::RenderPassBeginInfo {
                        *render_pass,
                        *fb,
                        vk::Rect2D {
                                vk::Offset2D {0, 0},
                                window->extent
                        },
                        static_cast <uint32_t> (clear_values.size()),
                        clear_values.data()
                },
                vk::SubpassContents::eInline
        );

        // Start ImGUI frame
        ImGui_ImplVulkan_NewFrame();
        ImGui_ImplGlfw_NewFrame();

        // Begin ImGUI frame
        ImGui::NewFrame();

        // Dockspace
        ImGui::DockSpaceOverViewport(ImGui::GetMainViewport());

        // TODO: create a static layout...
        auto &io = ImGui::GetIO();

        // Scrollable list of projects
        ImGui::Begin("Project Viewer");
        
        bool open_error = false;
        if (projects.empty()) {
                // Center the text and fill the entire window
                ImVec2 window_size = ImGui::GetWindowSize();
                ImVec2 text_size = ImGui::CalcTextSize("No projects found");

                ImGui::SetCursorPos({
                        (window_size.x - text_size.x) / 2.0f,
                        (window_size.y - text_size.y) / 2.0f
                });

                ImGui::Text("No projects found");
        } else {
                ImGui::Columns(5, nullptr, false);

                int size = projects.size();
                for (size_t i = 0; i < size; i++) {
                        ImVec4 color = verified[i] ?
                                ImVec4(1.0f, 1.0f, 1.0f, 1.0f)
                                : ImVec4(1.0f, 0.5f, 0.5f, 1.0f);

                        // Name of project is simply its directory name
                        std::string project_name = projects[i];
                        project_name = project_name.substr(project_name.find_last_of("/\\") + 1);

                        // TODO: temporary thumbnail for now...
                        // ImGui::Image(default_thumbnail_set, ImVec2(100, 100));

                        // Thumbnail as a button
                        if (ImGui::ImageButton(thumbnail_sets[i], ImVec2(128, 128))) {
                                if (verified[i]) {
                                        std::cout << "Project " << project_name << " selected" << std::endl;
                                        g_application.project = projects[i];
                                        terminate_now();
                                } else {
                                        std::cout << "Project " << project_name << " not verified" << std::endl;
                                        open_error = true;
                                }
                        }

                        // Align the text to the center of the thumbnail
                        ImVec2 thumbnail_pos = ImGui::GetItemRectMin();
                        ImVec2 thumbnail_size = ImGui::GetItemRectSize();
                        ImVec2 text_size = ImGui::CalcTextSize(project_name.c_str());

                        ImGui::SetCursorPos({
                                thumbnail_pos.x + (thumbnail_size.x - text_size.x) / 2.0f,
                                thumbnail_pos.y + thumbnail_size.y
                        });

                        ImGui::PushStyleColor(ImGuiCol_Text, color);
                        ImGui::Text("%s", project_name.c_str());
                        ImGui::PopStyleColor();

                        // TODO: thumbails as SVG...
                }
        }

        if (open_error) {
                std::cout << "Invalid project selected" << std::endl;
                ImGui::OpenPopup("Invalid Project");
        }

        // Popup for invalid project
        // TODO: lambda for centering
        ImGui::SetNextWindowPos(ImVec2(io.DisplaySize.x * 0.5f, io.DisplaySize.y * 0.5f), ImGuiCond_Always, ImVec2(0.5f, 0.5f));
        if (ImGui::BeginPopup("Invalid Project")) {
                ImGui::Text("Project is invalid!");

                if (ImGui::Button("OK"))
                        ImGui::CloseCurrentPopup();

                ImGui::EndPopup();
        }

        ImGui::End();

        // Project manager
        ImGui::Begin("Project Manager");

        // TODO: center the button and etc...
        if (ImGui::Button("Create Project")) {
                std::cout << "Create Project!" << std::endl;

                ImGui::OpenPopup("Project Creation");
        }

        // Popup for project creation
        ImGui::SetNextWindowPos(ImVec2(io.DisplaySize.x * 0.5f, io.DisplaySize.y * 0.5f), ImGuiCond_Always, ImVec2(0.5f, 0.5f));
        if (ImGui::BeginPopup("Project Creation")) {
                // Set the size of the popup
                ImGui::SetWindowSize(ImVec2(400, 200));

                // Name
                ImGui::Text("Project Name:");

                // char project_name[256] = {};
                ImGui::InputText("##project_name", project_name, 256);

                // Path
                ImGui::Text("Project Location:");

                // char project_location[256] = {};
                ImGui::InputText("##project_location", project_path, 256);

                ImGui::SameLine();

                if (ImGui::Button("Browse")) {
                        std::cout << "Browse for project location" << std::endl;
                        std::string path = project_parent_directory();
                        if (!path.empty())
                                strcpy(project_path, path.c_str());
                }

                // Confirm or cancel project creation
                bool open_warning = false;
                if (ImGui::Button("Create")) {
                        // Popup warn if project name or location is empty
                        if (strlen(project_name) == 0 || strlen(project_path) == 0) {
                                std::cout << "Project name or location is empty!" << std::endl;
                                open_warning = true;
                                ImGui::CloseCurrentPopup();
                        } else {
                                std::filesystem::path dir = std::filesystem::path(project_path);
                                dir /= project_name;

                                // Make sure the directory is created
                                std::cout << "Create project at " << dir << std::endl;
                                kobra::Project::basic(get_context(), dir).save();

                                append_to_config(dir);

                                projects = load_config();
                                verified = vertify_projects(projects);

                                // Allocate a thumbnail set for the new project
                                // TODO: method
                                const kobra::ImageData &thumbnail = texture_loader->load_texture(KOBRA_DIR "/kobra_icon.png", false);
                                const vk::raii::Sampler &sampler = texture_loader->load_sampler(KOBRA_DIR "/kobra_icon.png");

                                vk::DescriptorSet set = ImGui_ImplVulkan_AddTexture(
                                        static_cast <VkSampler> (*sampler),
                                        static_cast <VkImageView> (*thumbnail.view),
                                        static_cast <VkImageLayout> (vk::ImageLayout::eShaderReadOnlyOptimal)
                                );

                                thumbnail_sets.push_back(set);

                                // Reset and close
                                project_name[0] = '\0';
                                project_path[0] = '\0';

                                ImGui::CloseCurrentPopup();
                        }
                }

                ImGui::SameLine();

                if (ImGui::Button("Cancel")) {
                        std::cout << "Cancel project creation" << std::endl;

                        project_name[0] = '\0';
                        project_path[0] = '\0';

                        ImGui::CloseCurrentPopup();
                }

                ImGui::EndPopup();

                if (open_warning) {
                        ImGui::OpenPopup("Project Creation Warning");
                }
        }

        // Popup for project creation warning
        ImGui::SetNextWindowPos(ImVec2(io.DisplaySize.x * 0.5f, io.DisplaySize.y * 0.5f), ImGuiCond_Always, ImVec2(0.5f, 0.5f));
        if (ImGui::BeginPopup("Project Creation Warning")) {
                ImGui::Text("Project name or location is empty!");

                bool return_popup = false;
                if (ImGui::Button("OK")) {
                        return_popup = true;
                        ImGui::CloseCurrentPopup();
                }

                ImGui::EndPopup();
                        
                if (return_popup)
                        ImGui::OpenPopup("Project Creation");
        }

        ImGui::End();

        // TODO: show project prorperties at the bottom of the window

        // End ImGUI frame
        ImGui::Render();

        // Write to the command buffer
        ImGui_ImplVulkan_RenderDrawData(ImGui::GetDrawData(), *cmd);

        cmd.endRenderPass();
        cmd.end();
}

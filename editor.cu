#include "editor/common.hpp"
#include "editor/material_preview.hpp"

// Forward declarations
struct ProgressBar;
struct Console;
struct MaterialEditor;
// struct RTXRenderer;
struct Viewport;
struct SceneGraph;
struct EntityProperties;

// TODO: add updated (emissive) materials as lights...

// TODO: logging attachment
// TODO: info tab that shows logging and framerate...
// TODO: viewport attachment

Application g_application;

struct WindowBounds {
        glm::vec2 min;
        glm::vec2 max;
};

inline bool within(const glm::vec2 &x, const WindowBounds &wb)
{
        return x.x >= wb.min.x && x.x <= wb.max.x
                && x.y >= wb.min.y && x.y <= wb.max.y;
}

inline glm::vec2 normalize(const glm::vec2 &x, const WindowBounds &wb)
{
        return glm::vec2 {
                (x.x - wb.min.x) / (wb.max.x - wb.min.x),
                (x.y - wb.min.y) / (wb.max.y - wb.min.y),
        };
}

struct InputRequest {
        glm::vec2 position;
        glm::vec2 delta;
        glm::vec2 start;
        
        enum {
                eNone,
                ePress,
                eRelease,
                eDrag
        } type;
};

struct InputContext {
        // Viewport window
        struct : WindowBounds {
                float sensitivity = 0.001f;
                float yaw = 0.0f;
                float pitch = 0.0f;
                bool dragging = false;
        } viewport;

        // Material preview window
        struct : WindowBounds {
                float sensitivity = 0.05f;
                bool dragging = false;
        } material_preview;

        // Input requests
        std::queue <InputRequest> requests;

        // Dragging states
        bool dragging = false;
        bool alt_dragging = false;
        glm::vec2 drag_start;
} input_context;

// TODO: only keep the state here...
struct Editor : public kobra::BaseApp {
	Scene m_scene;
	Project m_project;

        MaterialPreview *material_preview;

	layers::ForwardRenderer m_forward_renderer;

        std::shared_ptr <kobra::daemons::Transform> transform_daemon;

	std::shared_ptr <kobra::layers::UI> m_ui;

	std::shared_ptr <Console> m_console;
	std::shared_ptr <MaterialEditor> m_material_editor;

        struct {
                std::shared_ptr <Viewport> viewport;
        } m_ui_attachments;

	// Renderers
	struct {
		std::shared_ptr <kobra::amadeus::System> system;
		std::shared_ptr <kobra::layers::MeshMemory> mesh_memory;

		kobra::layers::Denoiser denoiser;
		kobra::layers::Framer framer;

		std::mutex movement_mutex;
		std::queue <uint32_t> movement;

		bool denoise = false;
	} m_renderers;

	// Viewport
	struct {
		// Scene viewing camera
		Camera camera { 45.0f, 1.0f };
		Transform camera_transform;

		ImVec2 min = {1/0.0f, 1/0.0f};
		ImVec2 max = {-1.0f, -1.0f};
		
                vk::raii::Sampler sampler = nullptr;
	} m_viewport;

	// Buffers
	struct {
		CUdeviceptr traced;
		std::vector <uint8_t> traced_cpu;
	} m_buffers;

	std::pair <int, int> m_selection = {-1, -1};
        std::set <int> m_highlighted_entities;

	// Input state
	// TODO: bring all other related fields here
	struct {
		bool viewport_hovered = false;
		bool viewport_focused = false;

		// TODO: put this into another struct...
		std::queue <std::string> capture_requests;
		kobra::BufferData capture_buffer = nullptr;
		std::vector <uint8_t> capture_data;
		std::string current_capture_path;
	} m_input;

        std::shared_ptr <EditorViewport> m_editor_renderer;

	Editor(const vk::raii::PhysicalDevice &, const std::vector <const char *> &);
	~Editor();

	void record(const vk::raii::CommandBuffer &, const vk::raii::Framebuffer&) override;
	void resize(const vk::Extent2D &) override;
	void after_present() override;

	// TODO: frustrum culling structure to cull once per pass (store status
	// in a map) and then is passed to other layers for rendering
};
	
void mouse_callback(void *, const kobra::io::MouseEvent &);
void keyboard_callback(void *, const kobra::io::KeyboardEvent &);

int main()
{
	// Load Vulkan physical device
	auto predicate = [](const vk::raii::PhysicalDevice &dev) {
		return kobra::physical_device_able(dev,  {
			VK_KHR_SWAPCHAIN_EXTENSION_NAME,
			VK_KHR_GET_MEMORY_REQUIREMENTS_2_EXTENSION_NAME,
			VK_KHR_EXTERNAL_MEMORY_EXTENSION_NAME,
			VK_KHR_EXTERNAL_SEMAPHORE_EXTENSION_NAME,
			// VK_NV_FILL_RECTANGLE_EXTENSION_NAME,
		});
	};

	vk::raii::PhysicalDevice phdev = kobra::pick_physical_device(predicate);

	Editor editor {
		phdev, {
			VK_KHR_SWAPCHAIN_EXTENSION_NAME,
			VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME,
			VK_KHR_EXTERNAL_MEMORY_EXTENSION_NAME,
			// VK_NV_FILL_RECTANGLE_EXTENSION_NAME,
		},
	};

	editor.run();
}

// Info UI Attachment
struct Console : public kobra::ui::ImGuiAttachment {
	struct LogItem {
		kobra::Log level;
		std::string time;
		std::string source;
		std::string message;
	};

	std::vector <LogItem> m_lines;
	std::string m_message;

	void add_log(kobra::Log level, const std::string &time, const std::string &header,
			const std::string &source, const std::string &message) {
		// TODO: instead of rendering the header, render a spite if
		// error or warning...
		m_lines.push_back({level, time, source, message});
	}

	// TODO: multiple fonts; use monospace for this (e.g. JetBrains Mono)
	Console() {
		// Attach logger handler
		kobra::add_log_handler(this, std::bind(
				&Console::add_log, this,
				std::placeholders::_1, std::placeholders::_2,
				std::placeholders::_3, std::placeholders::_4,
				std::placeholders::_5
			)
		);
	}

	~Console() {
		kobra::remove_log_handler(this);
	}

	void render() override {
		// Output and performance tabs
		ImGui::Begin("Console");

		ImGui::SetWindowSize(ImVec2(500, 500), ImGuiCond_FirstUseEver);

		ImGui::Text("Output");

		// TODO: scroll to bottom
		// TODO: color code...
		// TODO: vertica barbetween timestamp (and source), message tpye, and message
		ImGui::Columns(3, "output", true);
		ImGui::Separator();
		ImGui::Text("Timestamp");
		ImGui::NextColumn();
		ImGui::Text("Source");
		ImGui::NextColumn();
		ImGui::Text("Message");
		ImGui::NextColumn();
		ImGui::Separator();

		for (const auto &line : m_lines) {
			ImVec4 color = ImVec4(1.0f, 1.0f, 1.0f, 1.0f);
			if (line.level == kobra::Log::ERROR)
				color = ImVec4(1.0f, 0.5f, 0.5f, 1.0f);
			else if (line.level == kobra::Log::WARN)
				color = ImVec4(1.0f, 1.0f, 0.5f, 1.0f);
			else if (line.level == kobra::Log::INFO)
				color = ImVec4(0.5f, 1.0f, 0.5f, 1.0f);
			else if (line.level == kobra::Log::OK)
				color = ImVec4(0.5f, 0.5f, 1.0f, 1.0f);

			// Color
			ImGui::PushStyleColor(ImGuiCol_Text, color);

			ImGui::Text("%s", line.time.c_str());
			ImGui::NextColumn();

			// Italicize source
			// TODO: this needs a different font
			ImGui::Text("%s", line.source.c_str());
			ImGui::NextColumn();

			ImGui::Text("%s", line.message.c_str());
			ImGui::NextColumn();

			ImGui::PopStyleColor();
		}

		ImGui::End();
	}
};

// Material editor UI attachment
class MaterialEditor : public kobra::ui::ImGuiAttachment {
	int m_prev_material_index = -1;

        vk::Sampler sampler;
        vk::DescriptorSet dset_material_preview;

	vk::DescriptorSet m_diffuse_set;
	vk::DescriptorSet m_normal_set;

	glm::vec3 emission_base = glm::vec3(0.0f);
	float emission_strength = 0.0f;

	Editor *m_editor = nullptr;
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
	int material_index = -1;

	MaterialEditor() = delete;
	MaterialEditor(Editor *editor, kobra::TextureLoader *texture_loader)
			: m_editor {editor}, m_texture_loader {texture_loader} {
                // Allocate the material preview descriptor set
                vk::SamplerCreateInfo sampler_info {
                        {}, vk::Filter::eLinear, vk::Filter::eLinear,
                        vk::SamplerMipmapMode::eLinear,
                        vk::SamplerAddressMode::eRepeat,
                        vk::SamplerAddressMode::eRepeat,
                        vk::SamplerAddressMode::eRepeat,
                        0.0f, VK_FALSE, 16.0f, VK_FALSE,
                        vk::CompareOp::eNever, 0.0f, 0.0f,
                        vk::BorderColor::eFloatOpaqueWhite, VK_FALSE
                };

                sampler = (*m_editor->device).createSampler(sampler_info);

                MaterialPreview *mp = m_editor->material_preview;
                dset_material_preview = ImGui_ImplVulkan_AddTexture(
                        static_cast <VkSampler> (sampler),
                        static_cast <VkImageView> (mp->display.view),
                        static_cast <VkImageLayout> (vk::ImageLayout::eGeneral)
                );
        }

        ~MaterialEditor() {
                (*m_editor->device).destroySampler(sampler);
                // ImGui_ImplVulkan_RemoveTexture(static_cast <VkDescriptorSet> (dset_material_preview));
        }

	void render() override {
		ImGui::Begin("Material Editor");
		if (material_index < 0) {
                        input_context.material_preview.min = glm::vec2 { -1.0f, -1.0f };
                        input_context.material_preview.max = glm::vec2 { -1.0f, -1.0f };
			ImGui::End();
			return;
		}

                // TODO: need to make this a bit more dynamic
                // and leave space for the material data...

                // Transfer material properties to the material preview renderer
                MaterialPreview *mp = m_editor->material_preview;
                mp->index = material_index;

                ImGui::Image(dset_material_preview, ImVec2(256, 256));

                ImVec2 pmin = ImGui::GetItemRectMin();
                ImVec2 pmax = ImGui::GetItemRectMax();

                ImGui::GetForegroundDrawList()->AddRect(pmin, pmax, IM_COL32(255, 255, 0, 255));

                input_context.material_preview.min = glm::vec2 { pmin.x, pmin.y };
                input_context.material_preview.max = glm::vec2 { pmax.x, pmax.y };

		// Check if it is a new material
		bool is_not_loaded = m_prev_material_index != material_index;
		m_prev_material_index = material_index;

		// For starters, print material data
		ImGui::Text("Material data:");
		ImGui::Separator();

		kobra::Material *material = &kobra::Material::all[material_index];

		glm::vec3 diffuse = material->diffuse;
		glm::vec3 specular = material->specular;
		// glm::vec3 ambient = material->ambient; TODO: remove this
		// property...
		float roughness = material->roughness;

		// Decompose the emission if it is not loaded
		if (is_not_loaded) {
			emission_base = glm::vec3(0.0f);
			emission_strength = 0.0f;

			// If any component is greater than 1, normalize it
			glm::vec3 emission = material->emission;
			if (emission.r > 1.0f || emission.g > 1.0f || emission.b > 1.0f) {
				emission_strength = glm::length(emission);
				emission_base = emission / emission_strength;
			}
		}

		bool updated_material = false;

		if (ImGui::ColorEdit3("Diffuse", &diffuse.r)) {
			material->diffuse = diffuse;
			updated_material = true;
		}

		if (ImGui::ColorEdit3("Specular", &specular.r)) {
			material->specular = specular;
			updated_material = true;
		}

		// TODO: remove ambient from material

		// TODO: use an HSL color picker + intensity slider
		if (ImGui::ColorEdit3("Emission", &emission_base.r)) {
			material->emission = emission_strength * emission_base;
			updated_material = true;
		}

		if (ImGui::SliderFloat("Intensity", &emission_strength, 0.0f, 1000.0f)) {
			material->emission = emission_strength * emission_base;
			updated_material = true;
		}

		// TODO: emission intensity
		if (ImGui::SliderFloat("Roughness", &roughness, 0.0f, 1.0f)) {
			material->roughness = std::max(roughness, 0.001f);
			updated_material = true;
		}

		// Transmission index of refraction
		if (ImGui::SliderFloat("IOR", &material->refraction, 1.0f, 3.0f))
			updated_material = true;

		// TODO: option for transmission
		bool transmission = (material->type == eTransmission);
		if (ImGui::Checkbox("Transmission", &transmission)) {
			material->type = transmission ? eTransmission : eDiffuse;
			updated_material = true;
		}

		ImGui::Separator();

		if (material->has_albedo()) {
			ImGui::Text("Diffuse Texture:");

			std::string diffuse_path = material->albedo_texture;
			if (is_not_loaded)
				m_diffuse_set = imgui_allocate_image(diffuse_path);

			ImGui::Image(m_diffuse_set, ImVec2(256, 256));
			ImGui::Separator();
		}

		if (material->has_normal()) {
			ImGui::Text("Normal Texture:");

			std::string normal_path = material->normal_texture;
			if (is_not_loaded)
				m_normal_set = imgui_allocate_image(normal_path);

			ImGui::Image(m_normal_set, ImVec2(256, 256));
			ImGui::Separator();
		}

		// Notify the daemon that the material has been updated
		if (updated_material) {
			kobra::Material::daemon.update(material_index);
			std::lock_guard <std::mutex> lock_guard
				(m_editor->m_renderers.movement_mutex);
			m_editor->m_renderers.movement.push(0);
		}

		ImGui::End();
	}
};

// void load_attachment(Editor *editor)
// {
// 	std::cout << "Loading RTX plugin..." << std::endl;
//
// 	std::string current_path = std::filesystem::current_path();
// 	nfdchar_t *path = nullptr;
// 	nfdfilteritem_t filter = {"RTX Plugin", "rtxa"};
// 	nfdresult_t result = NFD_OpenDialog(&path, &filter, 1, current_path.c_str());
//
// 	if (result == NFD_OKAY) {
// 		std::cout << "Loading " << path << std::endl;
//
// 		void *handle = dlopen(path, RTLD_LAZY);
// 		if (!handle) {
// 			std::cerr << "Error: " << dlerror() << std::endl;
// 			return;
// 		}
//
// 		// Load the plugin
// 		struct Attachment {
// 			const char *name;
// 			kobra::amadeus::AttachmentRTX *ptr;
// 		};
//
// 		typedef Attachment (*plugin_t)();
//
// 		plugin_t plugin = (plugin_t) dlsym(handle, "load_attachment");
// 		if (!plugin) {
// 			kobra::logger("Editor::load_attachment", kobra::Log::ERROR)
// 				<< "Error: " << dlerror() << "\n";
// 			return;
// 		}
//
// 		// TODO: use rtxa extension, and ignore metadata
// 		std::cout << "Loading plugin..." << std::endl;
// 		Attachment attachment = plugin();
// 		std::cout << "Attachment loaded: " << attachment.name << "@" << attachment.ptr << std::endl;
// 		if (!attachment.ptr) {
// 			KOBRA_LOG_FILE(kobra::Log::ERROR) << "Error: plugin is null\n";
// 			dlclose(handle);
// 			return;
// 		}
//
// 		editor->m_renderers.armada_rtx->attach(
// 			attachment.name,
// 			std::shared_ptr <kobra::amadeus::AttachmentRTX> (attachment.ptr)
// 		);
//
// 		{
// 			std::lock_guard <std::mutex> lock_guard
// 				(editor->m_renderers.movement_mutex);
// 			editor->m_renderers.movement.push(0);
// 		}
//
// 		std::cout << "All attachments:" << std::endl;
// 		for (auto &attachment : editor->m_renderers.armada_rtx->attachments()) {
// 			std::cout << "\t" << attachment << std::endl;
// 		}
//
// 		dlclose(handle);
//
// 		// Signal
// 	} else if (result == NFD_CANCEL) {
// 		std::cout << "User cancelled" << std::endl;
// 	} else {
// 		std::cout << "Error: " << NFD_GetError() << std::endl;
// 	}
// }
//
// // RTX Renderer UI attachment
// // TODO: put all these attachments in separate headers
// class RTXRenderer : public kobra::ui::ImGuiAttachment {
// 	Editor *m_editor = nullptr;
// 	int m_path_depth = 0;
// 	bool m_enable_envmap = true;
// public:
// 	RTXRenderer() = delete;
// 	RTXRenderer(Editor *editor)
// 			: m_editor {editor},
// 			m_path_depth {2},
// 			m_enable_envmap {true} {
// 		m_editor->m_renderers.armada_rtx->set_depth(m_path_depth);
// 		m_editor->m_renderers.armada_rtx->set_envmap_enabled(m_enable_envmap);
// 	}
//
// 	void render() override {
// 		ImGui::Begin("RTX Renderer");
//
// 		// Setting the path depth
// 		if (ImGui::SliderInt("Path Depth", &m_path_depth, 0, 10)) {
// 			m_editor->m_renderers.armada_rtx->set_depth(m_path_depth);
// 			std::lock_guard <std::mutex> lock_guard
// 				(m_editor->m_renderers.movement_mutex);
// 			m_editor->m_renderers.movement.push(0);
// 		}
//
// 		// TODO: roussian roulette, different integrators, and loading
// 		// RTX attachments
//
// 		// Drop down to choose the RTX attachment
// 		auto attachments = m_editor->m_renderers.armada_rtx->attachments();
// 		auto current = m_editor->m_renderers.armada_rtx->active_attachment();
// 		if (ImGui::BeginCombo("RTX Attachment", current.c_str())) {
// 			for (auto &attachment : attachments) {
// 				bool is_selected = (current == attachment);
// 				if (ImGui::Selectable(attachment.c_str(), is_selected)) {
// 					m_editor->m_renderers.armada_rtx->activate(attachment);
// 					std::lock_guard <std::mutex> lock_guard
// 						(m_editor->m_renderers.movement_mutex);
// 					m_editor->m_renderers.movement.push(0);
// 				}
//
// 				if (is_selected)
// 					ImGui::SetItemDefaultFocus();
// 			}
//
// 			ImGui::EndCombo();
// 		}
//
// 		// Checkboxes for enabling/disabling denoising
// 		ImGui::Checkbox("Denoise", &m_editor->m_renderers.denoise);
//
// 		bool russian_roulette = false;
// 		auto opt = m_editor->m_renderers.armada_rtx->get_option("russian_roulette");
// 		if (std::holds_alternative <bool> (opt))
// 			russian_roulette = std::get <bool> (opt);
//
// 		if (ImGui::Checkbox("Russian Roulette", &russian_roulette)) {
// 			m_editor->m_renderers.armada_rtx->set_option("russian_roulette", russian_roulette);
// 			std::lock_guard <std::mutex> lock_guard
// 				(m_editor->m_renderers.movement_mutex);
// 			m_editor->m_renderers.movement.push(0);
// 		}
//
// 		// Environment map
// 		if (ImGui::Checkbox("Environment Map", &m_enable_envmap)) {
// 			m_editor->m_renderers.armada_rtx->set_envmap_enabled(m_enable_envmap);
// 			std::lock_guard <std::mutex> lock_guard
// 				(m_editor->m_renderers.movement_mutex);
// 			m_editor->m_renderers.movement.push(0);
// 		}
//
// 		ImGui::Spacing();
// 		if (ImGui::Button("Load RTX Plugin")) {
// 			// TODO: do this async...
// 			load_attachment(m_editor);
// 		}
//
// 		ImGui::End();
// 	}
// };

void request_capture(Editor *editor)
{
	std::cout << "Snap!" << std::endl;

	// Create file dialog to get save path
	nfdchar_t *path = nullptr;
	nfdresult_t result = NFD_SaveDialog(&path, nullptr, 0, nullptr, nullptr);

	if (result == NFD_OKAY) {
		std::cout << "Path: " << path << std::endl;
		editor->m_input.capture_requests.push(path);
	} else if (result == NFD_CANCEL) {
		std::cout << "User cancelled" << std::endl;
	} else {
		std::cout << "Error: " << NFD_GetError() << std::endl;
	}
}

void import_asset(Editor *editor)
{
	std::cout << "Importing asset..." << std::endl;

	// Create file dialog to get save path
	nfdchar_t *path = nullptr;
	nfdresult_t result = NFD_OpenDialog(&path, nullptr, 0, nullptr);

	if (result == NFD_OKAY) {
                // Extract name of the entity from the file name
                // TODO: check for duplicates
                std::filesystem::path asset_path = path;
                kobra::Mesh mesh = *kobra::Mesh::load(asset_path.string());
                auto ecs = editor->m_scene.ecs;
                auto entity = ecs->make_entity(asset_path.stem());
                entity.add <kobra::Mesh> (mesh);
                entity.add <kobra::Renderable> (editor->get_context(), &entity.get <kobra::Mesh> ());
	} else if (result == NFD_CANCEL) {
		std::cout << "User cancelled" << std::endl;
	} else {
		std::cout << "Error: " << NFD_GetError() << std::endl;
	}
}

// Viewport UI attachment
// TODO: keep all viewport editor state in this class
// e.g. the renderers, etc...
class Viewport : public kobra::ui::ImGuiAttachment {
	Editor *m_editor = nullptr;
	vk::DescriptorSet m_dset;

	ImVec2 m_old_size = ImVec2(0.0f, 0.0f);
	float m_old_aspect = 0.0f;
	vk::Image m_old_image = nullptr;
        
        kobra::Transform *transform = nullptr;
        glm::mat4 proxy;

        const kobra::Camera *camera = nullptr;
        const kobra::Transform *camera_transform = nullptr;
        
        ImGuizmo::OPERATION current_operation = ImGuizmo::TRANSLATE;
public:
	Viewport() = delete;
	Viewport(Editor *editor) : m_editor {editor} {
		NFD_Init();
		m_old_aspect = m_editor->m_viewport.camera.aspect;
		m_old_size = {0, 0};
	}

        void set_operation(ImGuizmo::OPERATION op) {
                current_operation = op;
        }

        void set_transform() {
                transform = nullptr;
        }

        void set_transform(kobra::Entity &entity) {
                transform = &entity.get <kobra::Transform> ();
                proxy = transform->matrix();
        }

	// TODO: pass commandbuffer to this function
	void render() override {
		if (ImGui::BeginMainMenuBar()) {
			if (ImGui::BeginMenu("File")) {
				if (ImGui::MenuItem("Save")) {
					std::cout << "Saving (TODO: open file dialog)" << std::endl;
					m_editor->m_project.save("./scene");
				}
				ImGui::EndMenu();
			}

			if (ImGui::BeginMenu("Edit")) {
                                if (ImGui::MenuItem("Import Asset"))
                                        import_asset(m_editor);
				ImGui::EndMenu();
			}

			if (ImGui::BeginMenu("View")) {
				if (ImGui::MenuItem("Capture Viewport Image"))
					request_capture(m_editor);

				// TODO: viewport render setup
				// TODO: maybe in a separate window?

				ImGui::EndMenu();
			}

			ImGui::EndMainMenuBar();
		}

		// Minimum size for the viewport
		ImGui::SetNextWindowSizeConstraints(
			ImVec2(256, 256),
			ImVec2(FLT_MAX, FLT_MAX)
		);

		// TODO: separate attachment for the main menu bar
		ImGui::Begin("Viewport", nullptr, ImGuiWindowFlags_MenuBar);
                        
                MenuOptions options {
                        .camera = &m_editor->m_viewport.camera,
                        .speed = &g_application.speed,
                };

                // m_editor->m_editor_renderer->menu(options);
                show_menu(m_editor->m_editor_renderer, options);

		vk::Image image = *m_editor->m_editor_renderer->viewport_image();
		if (image == m_old_image) {
			// Get current window size
			ImVec2 window_size = ImGui::GetContentRegionAvail();

			// TODO: set the window aspect ratio
			ImGui::Image(m_dset, window_size);
               
                        // TODO: function to extract window bounds...
                        ImVec2 pmin = ImGui::GetItemRectMin();
                        ImVec2 pmax = ImGui::GetItemRectMax();

                        input_context.viewport.min = glm::vec2 { pmin.x, pmin.y };
                        input_context.viewport.max = glm::vec2 { pmax.x, pmax.y };

			// Check if the image has changed size
			ImVec2 image_size = ImGui::GetItemRectSize();
			if (image_size.x != m_old_size.x ||
				image_size.y != m_old_size.y) {
				m_old_size = image_size;

				// Add to sync queue
				// TODO: refactor...
				m_editor->sync_queue.push({
					"Resizing viewport",
					[&]() {
                                                if (m_old_size.x > 0 && m_old_size.y > 0) {
                                                        vk::Extent2D extent {
                                                                (uint32_t) m_old_size.x,
                                                                (uint32_t) m_old_size.y
                                                        };

                                                        m_editor->m_editor_renderer->resize(extent);
                                                }
					}
				});
			}

			// Get pixel range of the image
			ImVec2 image_min = ImGui::GetItemRectMin();
			ImVec2 image_max = ImGui::GetItemRectMax();

			m_editor->m_input.viewport_focused = ImGui::IsWindowFocused();
			m_editor->m_input.viewport_hovered = ImGui::IsItemHovered();

			m_editor->m_viewport.min = image_min;
			m_editor->m_viewport.max = image_max;

			// Fix aspect ratio if needed
			float aspect = (image_max.x - image_min.x) /
				(image_max.y - image_min.y);

			if (fabs(aspect - m_old_aspect) > 1e-6) {
				m_editor->m_viewport.camera.aspect = aspect;
				m_old_aspect = aspect;
			}
		} else {
			m_dset = ImGui_ImplVulkan_AddTexture(
				static_cast <VkSampler>
				(*m_editor->m_viewport.sampler),

				static_cast <VkImageView>
				(*m_editor->m_editor_renderer->viewport_image_view()),

				static_cast <VkImageLayout>
				(vk::ImageLayout::eShaderReadOnlyOptimal)
			);
		}
                
                if (transform) {
                        camera = &m_editor->m_viewport.camera;
                        camera_transform = &m_editor->m_viewport.camera_transform;

                        static ImGuizmo::MODE current_mode(ImGuizmo::WORLD);

                        ImGuizmo::SetDrawlist();

                        ImGuiIO &io = ImGui::GetIO();
                        float windowWidth = (float) ImGui::GetWindowWidth();
                        float windowHeight = (float) ImGui::GetWindowHeight();

                        ImGuizmo::SetRect(ImGui::GetWindowPos().x, ImGui::GetWindowPos().y,
                                windowWidth, windowHeight);

                        glm::mat4 view = camera->view_matrix(*camera_transform);
                        glm::mat4 proj = camera->perspective_matrix();

                        bool changed = ImGuizmo::Manipulate(
                                glm::value_ptr(view),
                                glm::value_ptr(proj),
                                current_operation, current_mode,
                                glm::value_ptr(proxy),
                                nullptr, nullptr
                        );

                        // NOTE: the code below works...
                        // if (changed)
                        //         std::cout << "Transforedm Updated (via Gizmo)!" << std::endl;

                        // TODO: check if the transform has changed, and if so
                        // then signal the daemon...

                        *transform = proxy;
                }

		m_old_image = image;
		ImGui::End();
	}
};

struct Performance : public kobra::ui::ImGuiAttachment {
	std::chrono::high_resolution_clock::time_point start_time;
public:
	Performance() {
		start_time = std::chrono::high_resolution_clock::now();
	}

	void render() override {
		ImGui::Begin("Performance");
		ImGui::Text("Framterate: %.1f", ImGui::GetIO().Framerate);

		// Plot the frame times over 5 seconds
		using frame_time = std::pair <float, float>;
		static std::vector <frame_time> frames;

		float fps = ImGui::GetIO().Framerate;
		float time = std::chrono::duration <float> (std::chrono::high_resolution_clock::now() - start_time).count();
		frames.push_back({time, fps});

		// Remove old frame times
		while (frames.size() > 0 && frames.front().first < time - 5.0f)
			frames.erase(frames.begin());

		// Plot the frame times
		ImPlot::SetNextAxesLimits(0, 5, 0, 165, ImGuiCond_Always);
		if (ImPlot::BeginPlot("Frame times")) {
			std::vector <float> times;
			std::vector <float> fpses;

			float min_time = frames.front().first;
			for (auto &frame : frames) {
				times.push_back(frame.first - min_time);
				fpses.push_back(frame.second);
			}

			// Set limits
			ImPlot::PlotLine("Framrate", times.data(), fpses.data(), times.size());
			ImPlot::EndPlot();
		}

		ImGui::End();
	}
};

// Scene graph
struct SceneGraph : public kobra::ui::ImGuiAttachment {
	const kobra::Scene *m_scene = nullptr;
public:
	SceneGraph() = default;

	void set_scene(const kobra::Scene *scene) {
		m_scene = scene;
	}

	void render() override {
		ImGui::Begin("Scene Graph");
		
		if (m_scene != nullptr) {
			auto &ecs = *m_scene->ecs;
			for (auto &entity : ecs)
				ImGui::Text("%s", entity.name.c_str());
		}

		// Open a popup when the user right clicks on the scene graph
		if (ImGui::BeginPopupContextWindow()) {
			if (ImGui::BeginMenu("Add Entity")) {
				if (ImGui::BeginMenu("Renderable")) {
					if (ImGui::MenuItem("Box")) {
						Mesh box = Mesh::box();
						// TODO: method to request new material from a daemon...
						box.submeshes[0].material_index = Material::all.size();
						Material::all.push_back(Material::default_material());
						auto &entity = m_scene->ecs->make_entity("Box");
						entity.add <Mesh> (box);
						entity.add <Renderable> (g_application.context, &entity.get <Mesh> ());
					}

					if (ImGui::MenuItem("Plane")) {
						Mesh plane = Mesh::plane();
						plane.submeshes[0].material_index = Material::all.size();
						Material::all.push_back(Material::default_material());
						auto &entity = m_scene->ecs->make_entity("Plane");
						entity.add <Mesh> (plane);
						entity.add <Renderable> (g_application.context, &entity.get <Mesh> ());
					}

					ImGui::EndMenu();
				}

				ImGui::EndMenu();
			}

			ImGui::EndPopup();
		}

		ImGui::End();
	}
};

static const char *environment_map_path = KOBRA_DIR
        "/resources/skies/background_1.jpg";

void set_imgui_theme()
{
        ImVec4* colors = ImGui::GetStyle().Colors;
        colors[ImGuiCol_Text]                   = ImVec4(1.00f, 1.00f, 1.00f, 1.00f);
        colors[ImGuiCol_TextDisabled]           = ImVec4(0.50f, 0.50f, 0.50f, 1.00f);
        colors[ImGuiCol_WindowBg]               = ImVec4(0.10f, 0.10f, 0.10f, 1.00f);
        colors[ImGuiCol_ChildBg]                = ImVec4(0.00f, 0.00f, 0.00f, 0.00f);
        colors[ImGuiCol_PopupBg]                = ImVec4(0.19f, 0.19f, 0.19f, 0.92f);
        colors[ImGuiCol_Border]                 = ImVec4(0.19f, 0.19f, 0.19f, 0.29f);
        colors[ImGuiCol_BorderShadow]           = ImVec4(0.00f, 0.00f, 0.00f, 0.24f);
        colors[ImGuiCol_FrameBg]                = ImVec4(0.05f, 0.05f, 0.05f, 0.54f);
        colors[ImGuiCol_FrameBgHovered]         = ImVec4(0.19f, 0.19f, 0.19f, 0.54f);
        colors[ImGuiCol_FrameBgActive]          = ImVec4(0.20f, 0.22f, 0.23f, 1.00f);
        colors[ImGuiCol_TitleBg]                = ImVec4(0.00f, 0.00f, 0.00f, 1.00f);
        colors[ImGuiCol_TitleBgActive]          = ImVec4(0.06f, 0.06f, 0.06f, 1.00f);
        colors[ImGuiCol_TitleBgCollapsed]       = ImVec4(0.00f, 0.00f, 0.00f, 1.00f);
        colors[ImGuiCol_MenuBarBg]              = ImVec4(0.14f, 0.14f, 0.14f, 1.00f);
        colors[ImGuiCol_ScrollbarBg]            = ImVec4(0.05f, 0.05f, 0.05f, 0.54f);
        colors[ImGuiCol_ScrollbarGrab]          = ImVec4(0.34f, 0.34f, 0.34f, 0.54f);
        colors[ImGuiCol_ScrollbarGrabHovered]   = ImVec4(0.40f, 0.40f, 0.40f, 0.54f);
        colors[ImGuiCol_ScrollbarGrabActive]    = ImVec4(0.56f, 0.56f, 0.56f, 0.54f);
        colors[ImGuiCol_CheckMark]              = ImVec4(0.33f, 0.67f, 0.86f, 1.00f);
        colors[ImGuiCol_SliderGrab]             = ImVec4(0.34f, 0.34f, 0.34f, 0.54f);
        colors[ImGuiCol_SliderGrabActive]       = ImVec4(0.56f, 0.56f, 0.56f, 0.54f);
        colors[ImGuiCol_Button]                 = ImVec4(0.05f, 0.05f, 0.05f, 0.54f);
        colors[ImGuiCol_ButtonHovered]          = ImVec4(0.19f, 0.19f, 0.19f, 0.54f);
        colors[ImGuiCol_ButtonActive]           = ImVec4(0.20f, 0.22f, 0.23f, 1.00f);
        colors[ImGuiCol_Header]                 = ImVec4(0.00f, 0.00f, 0.00f, 0.52f);
        colors[ImGuiCol_HeaderHovered]          = ImVec4(0.00f, 0.00f, 0.00f, 0.36f);
        colors[ImGuiCol_HeaderActive]           = ImVec4(0.20f, 0.22f, 0.23f, 0.33f);
        colors[ImGuiCol_Separator]              = ImVec4(0.28f, 0.28f, 0.28f, 0.29f);
        colors[ImGuiCol_SeparatorHovered]       = ImVec4(0.44f, 0.44f, 0.44f, 0.29f);
        colors[ImGuiCol_SeparatorActive]        = ImVec4(0.40f, 0.44f, 0.47f, 1.00f);
        colors[ImGuiCol_ResizeGrip]             = ImVec4(0.28f, 0.28f, 0.28f, 0.29f);
        colors[ImGuiCol_ResizeGripHovered]      = ImVec4(0.44f, 0.44f, 0.44f, 0.29f);
        colors[ImGuiCol_ResizeGripActive]       = ImVec4(0.40f, 0.44f, 0.47f, 1.00f);
        colors[ImGuiCol_Tab]                    = ImVec4(0.00f, 0.00f, 0.00f, 0.52f);
        colors[ImGuiCol_TabHovered]             = ImVec4(0.14f, 0.14f, 0.14f, 1.00f);
        colors[ImGuiCol_TabActive]              = ImVec4(0.20f, 0.20f, 0.20f, 0.36f);
        colors[ImGuiCol_TabUnfocused]           = ImVec4(0.00f, 0.00f, 0.00f, 0.52f);
        colors[ImGuiCol_TabUnfocusedActive]     = ImVec4(0.14f, 0.14f, 0.14f, 1.00f);
        colors[ImGuiCol_DockingPreview]         = ImVec4(0.33f, 0.67f, 0.86f, 1.00f);
        colors[ImGuiCol_DockingEmptyBg]         = ImVec4(1.00f, 0.00f, 0.00f, 1.00f);
        colors[ImGuiCol_PlotLines]              = ImVec4(1.00f, 0.00f, 0.00f, 1.00f);
        colors[ImGuiCol_PlotLinesHovered]       = ImVec4(1.00f, 0.00f, 0.00f, 1.00f);
        colors[ImGuiCol_PlotHistogram]          = ImVec4(1.00f, 0.00f, 0.00f, 1.00f);
        colors[ImGuiCol_PlotHistogramHovered]   = ImVec4(1.00f, 0.00f, 0.00f, 1.00f);
        colors[ImGuiCol_TableHeaderBg]          = ImVec4(0.00f, 0.00f, 0.00f, 0.52f);
        colors[ImGuiCol_TableBorderStrong]      = ImVec4(0.00f, 0.00f, 0.00f, 0.52f);
        colors[ImGuiCol_TableBorderLight]       = ImVec4(0.28f, 0.28f, 0.28f, 0.29f);
        colors[ImGuiCol_TableRowBg]             = ImVec4(0.00f, 0.00f, 0.00f, 0.00f);
        colors[ImGuiCol_TableRowBgAlt]          = ImVec4(1.00f, 1.00f, 1.00f, 0.06f);
        colors[ImGuiCol_TextSelectedBg]         = ImVec4(0.20f, 0.22f, 0.23f, 1.00f);
        colors[ImGuiCol_DragDropTarget]         = ImVec4(0.33f, 0.67f, 0.86f, 1.00f);
        colors[ImGuiCol_NavHighlight]           = ImVec4(1.00f, 0.00f, 0.00f, 1.00f);
        colors[ImGuiCol_NavWindowingHighlight]  = ImVec4(1.00f, 0.00f, 0.00f, 0.70f);
        colors[ImGuiCol_NavWindowingDimBg]      = ImVec4(1.00f, 0.00f, 0.00f, 0.20f);
        colors[ImGuiCol_ModalWindowDimBg]       = ImVec4(1.00f, 0.00f, 0.00f, 0.35f);

        ImGuiStyle& style = ImGui::GetStyle();
        style.WindowPadding                     = ImVec2(8.00f, 8.00f);
        style.FramePadding                      = ImVec2(5.00f, 2.00f);
        style.CellPadding                       = ImVec2(6.00f, 6.00f);
        style.ItemSpacing                       = ImVec2(6.00f, 6.00f);
        style.ItemInnerSpacing                  = ImVec2(6.00f, 6.00f);
        style.TouchExtraPadding                 = ImVec2(0.00f, 0.00f);
        style.IndentSpacing                     = 25;
        style.ScrollbarSize                     = 15;
        style.GrabMinSize                       = 10;
        style.WindowBorderSize                  = 1;
        style.ChildBorderSize                   = 1;
        style.PopupBorderSize                   = 1;
        style.FrameBorderSize                   = 1;
        style.TabBorderSize                     = 1;
        style.WindowRounding                    = 7;
        style.ChildRounding                     = 4;
        style.FrameRounding                     = 3;
        style.PopupRounding                     = 4;
        style.ScrollbarRounding                 = 9;
        style.GrabRounding                      = 3;
        style.LogSliderDeadzone                 = 4;
        style.TabRounding                       = 4;
}

// Editor implementation
Editor::Editor(const vk::raii::PhysicalDevice &phdev,
		const std::vector <const char *> &extensions)
		: kobra::BaseApp {
			phdev, "Kobra Engine",
			vk::Extent2D {1500, 1000},
			extensions
		}
{
	m_console = std::make_shared <Console> ();

	// TODO: constructor should be loaded very fast, everything else should
	// be loaded as needed...
	int MIP_LEVELS = 5;

	// Load environment map
	// TODO: load HDR...
	kobra::ImageData &environment_map = m_texture_loader
		.load_texture(environment_map_path);

	KOBRA_LOG_FUNC(kobra::Log::WARN) << "Starting irradiance computations...\n";

        // Material preview
        material_preview = make_material_preview(get_context());
        load_environment_map(material_preview, environment_map_path);

	// Load all the layers
	m_forward_renderer = kobra::layers::ForwardRenderer(get_context());

	// Configure ImGui
	ImGui::CreateContext();
	ImPlot::CreateContext();
	ImGui_ImplGlfw_InitForVulkan(window.m_handle, true);
        set_imgui_theme();

	// Docking
	ImGuiIO &imgui_io = ImGui::GetIO();
	imgui_io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;
	imgui_io.ConfigWindowsMoveFromTitleBarOnly = true;

	auto font = std::make_pair(KOBRA_FONTS_DIR "/Frutiger/Frutiger.ttf", 18);
	m_ui = std::make_shared <kobra::layers::UI> (
		get_context(), window,
		graphics_queue, font,
		vk::AttachmentLoadOp::eClear
	);

	// Load scene
 	m_project.load_project("scene");
	
	// m_scene.load(get_context(), project.scene);
	m_scene = m_project.load_scene(get_context());
	assert(m_scene.ecs);

        transform_daemon = std::make_shared <kobra::daemons::Transform> (m_scene.ecs.get());

	// IO callbacks
	io.mouse_events.subscribe(mouse_callback, this);
	io.keyboard_events.subscribe(keyboard_callback, this);

	// TODO: irradiance computer load from cache...

	// TODO: each layer that renders should have its own frmebuffer, or at
	// least a way to specify the image to render to (and then the layer
	// creates a framebuffer...)

	// Load all the renderers
	m_renderers.system = std::make_shared <kobra::amadeus::System> (transform_daemon.get());
	m_renderers.mesh_memory = std::make_shared <kobra::layers::MeshMemory> (get_context());
	m_viewport.sampler = kobra::make_continuous_sampler(device);

	// Attach UI layers
	m_material_editor = std::make_shared <MaterialEditor> (this, &m_texture_loader);

	auto scene_graph = std::make_shared <SceneGraph> ();
	scene_graph->set_scene(&m_scene);

        m_ui_attachments.viewport = std::make_shared <Viewport> (this);

	// m_ui->attach(m_image_viewer);
	m_ui->attach(m_console);
	m_ui->attach(m_material_editor);
	// m_ui->attach(std::make_shared <RTXRenderer> (this));
	m_ui->attach(std::make_shared <Performance> ());
        m_ui->attach(m_ui_attachments.viewport);
	m_ui->attach(scene_graph);
        
        // TODO: UI attachemtn that shows frametime as little chunks per frame

        // EditorViewport
        m_editor_renderer = std::make_shared <EditorViewport>
                (get_context(), m_renderers.system, m_renderers.mesh_memory);

	// Load and set the icon
	std::string icon_path = KOBRA_DIR "/kobra_icon.png";
	std::cout << "Loading icon from " << icon_path << std::endl;

	GLFWimage icon;
	stbi_set_flip_vertically_on_load(false);
	icon.pixels = stbi_load(icon_path.c_str(), &icon.width, &icon.height, nullptr, 4);
	glfwSetWindowIcon(window.m_handle, 1, &icon);
	stbi_image_free(icon.pixels);
	stbi_set_flip_vertically_on_load(true);

	// Configure global comunication state
	g_application.context = get_context();
}

Editor::~Editor()
{
        destroy_material_preview(material_preview);

	device.waitIdle();

	// TODO: method for total destruction
	ImGui_ImplVulkan_Shutdown();
	ImGui_ImplGlfw_Shutdown();
	ImGui::DestroyContext();
}

void handle_camera_input(Editor *editor)
{
	auto &transform = editor->m_viewport.camera_transform;
        auto &io = editor->io;

        float speed = g_application.speed * editor->frame_time;

        glm::vec3 forward = transform.forward();
        glm::vec3 right = transform.right();
        glm::vec3 up = transform.up();

        bool moved = false;
        if (io.input->is_key_down(GLFW_KEY_W)) {
                transform.move(forward * speed);
                moved = true;
        } else if (io.input->is_key_down(GLFW_KEY_S)) {
                transform.move(-forward * speed);
                moved = true;
        }

        if (io.input->is_key_down(GLFW_KEY_A)) {
                transform.move(-right * speed);
                moved = true;
        } else if (io.input->is_key_down(GLFW_KEY_D)) {
                transform.move(right * speed);
                moved = true;
        }

        if (io.input->is_key_down(GLFW_KEY_E)) {
                transform.move(up * speed);
                moved = true;
        } else if (io.input->is_key_down(GLFW_KEY_Q)) {
                transform.move(-up * speed);
                moved = true;
        }

        if (moved) {
                std::lock_guard <std::mutex> lock(editor->m_renderers.movement_mutex);
                // TODO: signal to transform daeon instead? but the
                // viewport camera is not an ECS entity
                editor->m_renderers.movement.push(0);
        }
}

void Editor::record(const vk::raii::CommandBuffer &cmd,
		const vk::raii::Framebuffer &framebuffer)
{
	// Camera movement
	if (m_input.viewport_focused || input_context.dragging || input_context.alt_dragging)
                handle_camera_input(this);

	std::vector <const kobra::Renderable *> renderables;
	std::vector <const kobra::Transform *> renderable_transforms;

	std::vector <const kobra::Light *> lights;
	std::vector <const kobra::Transform *> light_transforms;

	auto renderables_transforms = m_scene.ecs->tuples <kobra::Renderable, kobra::Transform> ();
	auto lights_transforms = m_scene.ecs->tuples <kobra::Light, kobra::Transform> ();

	auto ecs = m_scene.ecs;
	for (int i = 0; i < ecs->size(); i++) {
		if (ecs->exists <kobra::Renderable> (i)) {
			const auto *renderable = &ecs->get <kobra::Renderable> (i);
			const auto *transform = &ecs->get <kobra::Transform> (i);

			renderables.push_back(renderable);
			renderable_transforms.push_back(transform);
		}

		if (ecs->exists <kobra::Light> (i)) {
			const auto *light = &ecs->get <kobra::Light> (i);
			const auto *transform = &ecs->get <kobra::Transform> (i);

			lights.push_back(light);
			light_transforms.push_back(transform);
		}
	}

	kobra::layers::ForwardRenderer::Parameters params {
		.renderables = renderables_transforms,
		.lights = lights_transforms,
		// .pipeline_package = "environment",
	};

	params.environment_map = environment_map_path;

	cmd.begin({});
                // Editor renderer
                RenderInfo render_info { cmd };
                render_info.camera = m_viewport.camera;
                render_info.camera_transform = m_viewport.camera_transform;
                // render_info.cmd = &cmd;
                render_info.extent = m_editor_renderer->extent;
                // render_info.framebuffer = &m_viewport.framebuffer;
                render_info.highlighted_entities = m_highlighted_entities;

                std::vector <Entity> renderable_entities = m_scene.ecs->tuple_entities <Renderable> ();
                m_editor_renderer->render(render_info, renderable_entities, *transform_daemon);

                /* m_editor_renderer->render_gbuffer(render_info, renderable_entities);
                m_editor_renderer->render_present(render_info); */

		// m_irradiance_computer.sample(cmd);
		/* if (m_irradiance_computer.sample(cmd)
				&& !m_irradiance_computer.cached
				&& !m_saved_irradiance) {
			m_irradiance_computer.save_irradiance_maps(
				get_context(),
				"irradiance_maps"
			);

			m_saved_irradiance = true;
		} */

		// Handle requests
		std::optional <InputRequest> selection_request;
		while (!input_context.requests.empty()) {
			InputRequest request = input_context.requests.front();
			input_context.requests.pop();

			selection_request = request;
		}

		if (selection_request)
			input_context.requests.push(*selection_request);

                ImageData &viewport_image = m_editor_renderer->viewport();
		viewport_image.layout = vk::ImageLayout::ePresentSrcKHR;
		viewport_image.transition_layout(cmd, vk::ImageLayout::eShaderReadOnlyOptimal);

		if (!m_input.capture_requests.empty()) {
			std::string path = m_input.capture_requests.front();
			// Onkly take the first request
			m_input.capture_requests = std::queue <std::string> ();

			std::cout << "Capturing to " << path << std::endl;
			m_input.current_capture_path = path;

			// Allocate the buffer for the image if it hasn't been allocated
			if (m_input.capture_buffer.size == 0) {
				std::cout << "Allocating capture buffer" << std::endl;

				// TODO: get the format in order to compute
				// size...
				int size = sizeof(uint32_t) * viewport_image.extent.width
					* viewport_image.extent.height;

				m_input.capture_buffer = kobra::BufferData(
					phdev, device, size,
					vk::BufferUsageFlagBits::eTransferDst,
					vk::MemoryPropertyFlagBits::eHostVisible
						| vk::MemoryPropertyFlagBits::eHostCoherent
				);
			}

			// Copy the image to the buffer
			viewport_image.transition_layout(cmd, vk::ImageLayout::eTransferSrcOptimal);

			cmd.copyImageToBuffer(
				*viewport_image.image,
				vk::ImageLayout::eTransferSrcOptimal,
				*m_input.capture_buffer.buffer,
				{vk::BufferImageCopy {
					0, 0, 0,
					{vk::ImageAspectFlagBits::eColor, 0, 0, 1},
					{0, 0, 0},
					{viewport_image.extent.width, viewport_image.extent.height, 1}
				}}
			);

			viewport_image.transition_layout(cmd, vk::ImageLayout::eShaderReadOnlyOptimal);

			// Add capture to sync queue
			sync_queue.push({
				"Capture Viewport Image",
				[&]() {
					m_input.capture_data.resize(m_input.capture_buffer.size);
					m_input.capture_buffer.download(m_input.capture_data);

					// Convert from BGRA to RGBA
					for (int i = 0; i < m_input.capture_data.size(); i += 4)
						std::swap(m_input.capture_data[i], m_input.capture_data[i + 2]);

					std::string path = m_input.current_capture_path;
					RawImage {
						m_input.capture_data,
						viewport_image.extent.width,
						viewport_image.extent.height,
						4
					}.write(path);

					kobra::logger("Editor", kobra::Log::OK)
						<< "Captured viewport image to "
						<< path << std::endl;
				}
			});
		}

                // Material preview
                // TODO: only if there is an active material
                // and input has been received on the corresponding window
                render_material_preview(*cmd, material_preview);

		// Render the UI last
		m_ui->render(cmd,
			framebuffer, window.m_extent,
			kobra::RenderArea::full(), {true}
		);
	cmd.end();

	// TODO: after present actions...
}

void Editor::resize(const vk::Extent2D &) {}

void handle_viewport_input(Editor *editor, const InputRequest &request)
{
        auto &viewport = input_context.viewport;

        // Clicking on entities
        if (request.type == InputRequest::ePress) {
                // If using Gizmo, ignore
                if (within(request.position, input_context.viewport) && !ImGuizmo::IsOver()) {
                        glm::vec2 fixed = normalize(request.position, input_context.viewport);

                        std::vector <Entity> renderable_entities = editor->m_scene.ecs->tuple_entities <Renderable> ();
                        auto indices = editor->m_editor_renderer->selection_query(renderable_entities, fixed);

                        // Update the material editor
                        if (indices.size() == 0) {
                                editor->m_material_editor->material_index = -1;
                                editor->m_ui_attachments.viewport->set_transform();
                                editor->m_highlighted_entities.clear();
                        } else {
                                auto selection = indices[0];
                                kobra::Renderable &renderable = editor->m_scene.ecs->get <kobra::Renderable> (selection.first);
                                editor->m_ui_attachments.viewport->set_transform(editor->m_scene.ecs->get_entity(selection.first));
                                uint32_t material_index = renderable.material_indices[selection.second];
                                editor->m_material_editor->material_index = material_index;
                                editor->m_highlighted_entities = {(int) material_index};
                        }
                }
        } else if (request.type == InputRequest::eRelease) {
		glfwSetInputMode(editor->window.m_handle, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
                viewport.dragging = false;
        } else if (request.type == InputRequest::eDrag) {
                bool started_within = within(request.start, input_context.viewport);
                bool currently_within = within(request.position, input_context.viewport);

                if (started_within && (currently_within || viewport.dragging)) {
                        if (!viewport.dragging)
                                glfwSetInputMode(editor->window.m_handle, GLFW_CURSOR, GLFW_CURSOR_DISABLED);

                        viewport.yaw -= viewport.sensitivity * request.delta.x;
                        viewport.pitch -= viewport.sensitivity * request.delta.y;

                        if (viewport.pitch > 89.0f)
                                viewport.pitch = 89.0f;
                        if (viewport.pitch < -89.0f)
                                viewport.pitch = -89.0f;
                
                        kobra::Transform &transform = editor->m_viewport.camera_transform;
                        transform.rotation.x = viewport.pitch;
                        transform.rotation.y = viewport.yaw;

                        std::lock_guard <std::mutex> lock(editor->m_renderers.movement_mutex);
                        editor->m_renderers.movement.push(0);
                        viewport.dragging = true;
                }
	}
}

void handle_material_preview_input(Editor *editor, const InputRequest &request)
{
        // TODO: zoom in and out as well
        auto &mp = input_context.material_preview;

        // Clicking on entities
        if (request.type == InputRequest::eRelease) {
		glfwSetInputMode(editor->window.m_handle, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
                mp.dragging = false;
        } else if (request.type == InputRequest::eDrag) {
                bool started_within = within(request.start, input_context.material_preview);
                bool currently_within = within(request.position, input_context.material_preview);

                if (started_within && (currently_within || mp.dragging)) {
                        if (!mp.dragging)
                                glfwSetInputMode(editor->window.m_handle, GLFW_CURSOR, GLFW_CURSOR_DISABLED);

                        static float theta = 0.0f;
                        static float phi = 0.0f;

                        float dtheta = mp.sensitivity * request.delta.x;
                        float dphi = -mp.sensitivity * request.delta.y;

                        theta += dtheta;
                        phi += dphi;

                        if (phi > 89.0f)
                                phi = 89.0f;
                        if (phi < -89.0f)
                                phi = -89.0f;

                        glm::vec3 eye = 3.0f * glm::vec3 {
                                cos(glm::radians(theta)) * cos(glm::radians(phi)),
                                sin(glm::radians(phi)),
                                sin(glm::radians(theta)) * cos(glm::radians(phi))
                        };
                        
                        // editor->material_preview->view_transform = glm::lookAt(eye, glm::vec3(0), up);
                        editor->material_preview->origin = eye;

                        mp.dragging = true;
                }
	}
}

void Editor::after_present()
{
        // TODO: handle all requests
	if (!input_context.requests.empty()) {
		// TODO: ideally should only be one type of request per after_present
		InputRequest request = input_context.requests.front();
		input_context.requests.pop();

                handle_viewport_input(this, request);
                handle_material_preview_input(this, request);
	}

	// Ping all systems using materials
	kobra::Material::daemon.ping_all();

        // Daemon update cycle
        transform_daemon->update();

        // Make sure the queue is empty
        input_context.requests = std::queue <InputRequest> ();
}

void mouse_callback(void *us, const kobra::io::MouseEvent &event)
{
	static const int select_button = GLFW_MOUSE_BUTTON_LEFT;
	static const int pan_button = GLFW_MOUSE_BUTTON_RIGHT;

        InputRequest request;
        request.type = InputRequest::eNone;

        // TODO: use glm vec2 for position...
        request.position = { float(event.xpos), float(event.ypos) };
        if (event.action == GLFW_PRESS)
                request.type = InputRequest::ePress;
        else if (event.action == GLFW_RELEASE)
                request.type = InputRequest::eRelease;

	// Panning around
        // TODO: move all the stuff below to the input context
	static float px = 0.0f;
	static float py = 0.0f;

	// Deltas and directions
	float dx = event.xpos - px;
	float dy = event.ypos - py;

	// Check if panning
	bool is_drag_button = (event.button == pan_button);
	if (event.action == GLFW_PRESS && is_drag_button) {
                input_context.drag_start = request.position;
		input_context.dragging = true;
        } else if (event.action == GLFW_RELEASE && is_drag_button) {
                input_context.drag_start = { -1.0f, -1.0f };
		input_context.dragging = false;
        }

        // TODO: put the io in the input context
        // bool &alt_dragging = input_context.alt_dragging;
	// bool is_alt_down = editor->io.input->is_key_down(GLFW_KEY_LEFT_ALT);
	// if (!alt_dragging && is_alt_down && editor->m_input.viewport_hovered) {
	// 	alt_dragging = true;
	// } else if (alt_dragging && !is_alt_down && !editor->m_input.viewport_hovered) {
	// 	alt_dragging = false;
	// }

        // if (dragging || alt_dragging) {
        //         request.type = InputRequest::eDrag;
        //         request.delta = { dx, dy };
        // }
        
        if (input_context.dragging) {
                request.type = InputRequest::eDrag;
                request.delta = { dx, dy };
                request.start = input_context.drag_start;
        }

	// Update previous position
	px = event.xpos;
	py = event.ypos;

        // Signal the input
        input_context.requests.push(request);
}

void keyboard_callback(void *us, const kobra::io::KeyboardEvent &event)
{
	Editor *editor = static_cast <Editor *> (us);
	if (event.action == GLFW_PRESS) {
                auto &viewport = editor->m_ui_attachments.viewport;
		
		if (event.key == GLFW_KEY_ESCAPE) {
                        // TODO: callback
			editor->m_selection = {-1, -1};
			editor->m_material_editor->material_index = -1;
                        editor->m_highlighted_entities.clear();
                        viewport->set_transform();
		}

                // Gizmo modes
                if (event.key == GLFW_KEY_R)
                        viewport->set_operation(ImGuizmo::ROTATE);
                if (event.key == GLFW_KEY_T)
                        viewport->set_operation(ImGuizmo::TRANSLATE);
                if (event.key == GLFW_KEY_Y)
                        viewport->set_operation(ImGuizmo::SCALE);
	}
}

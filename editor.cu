#include "editor/common.hpp"
#include "editor/editor_viewport.cuh"
#include "editor/inspector.hpp"
#include "editor/material_preview.hpp"
#include "editor/scene_graph.hpp"
#include "editor/startup.hpp"
#include "editor/ui/material_editor.hpp"

// Forward declarations
struct ProgressBar;
struct Console;
struct Viewport;
struct EntityProperties;

// TODO: add updated (emissive) materials as lights...

// TODO: logging attachment
// TODO: info tab that shows logging and framerate...
// TODO: viewport attachment

Application g_application;
InputContext input_context;

// TODO: only keep the state here...
struct Editor : public kobra::BaseApp {
	Scene m_scene;
	Project m_project;

        MaterialPreview *material_preview;

	layers::ForwardRenderer m_forward_renderer;

        std::shared_ptr <kobra::TransformDaemon> transform_daemon;

	std::shared_ptr <kobra::layers::UserInterface> m_ui;

        // User interface attachments
        Inspector *m_inspector;

	std::shared_ptr <Console> m_console;
	std::shared_ptr <MaterialEditor> m_material_editor;

        struct {
                std::shared_ptr <Viewport> viewport;
        } m_ui_attachments;

	// Renderers
	struct {
		std::shared_ptr <kobra::amadeus::Accelerator> system;
		std::shared_ptr <kobra::MeshDaemon> mesh_memory;

		kobra::layers::Denoiser denoiser;
		kobra::layers::Framer framer;

		std::mutex movement_mutex;
		std::queue <uint32_t> movement;

		bool denoise = false;
	} m_renderers;

	// Viewport
	struct {
		// Scene viewing camera
                // TODO: vector of such structs (multiple viewports)
		Camera camera { 45.0f, 1.0f };
		Transform camera_transform;
                bool camera_transform_dirty = false;

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
void set_imgui_theme();

int main()
{
	NFD_Init();

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

        Startup *startup  = new Startup {
		phdev, {
			VK_KHR_SWAPCHAIN_EXTENSION_NAME,
			VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME,
			VK_KHR_EXTERNAL_MEMORY_EXTENSION_NAME,
		},
        };

        startup->run();
        delete startup;

        // g_application.project = "scene";
        if (g_application.project.empty())
                return 0;

	Editor *editor = new Editor{
		phdev, {
			VK_KHR_SWAPCHAIN_EXTENSION_NAME,
			VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME,
			VK_KHR_EXTERNAL_MEMORY_EXTENSION_NAME,
		},
	};

	editor->run();
        delete editor;
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
                KOBRA_LOG_FUNC(Log::WARN) << "Needs to be implemented" << std::endl;
                // // Extract name of the entity from the file name
                // TODO: check for duplicates
                std::filesystem::path asset_path = path;
                auto [mesh, materials] = *kobra::Mesh::load(asset_path.string());

                auto system = editor->m_scene.system;
                auto entity = system->make_entity(asset_path.stem());
                entity.add <kobra::Mesh> (mesh);
                entity.add <kobra::Renderable> (editor->get_context(), &entity.get <kobra::Mesh> ());

                kobra::Mesh &mesh_ref = entity.get <kobra::Mesh> ();
                for (int i = 0; i < materials.size(); i++) {
                        auto &mat = materials[i];
                        if (mat.name.empty())
                                mat.name = entity.name + "_material_" + std::to_string(i);

                        std::cout << "Adding material: " << mat.name << std::endl;
                        int32_t index = load(system->material_daemon, mat);
                        mesh_ref.submeshes[i].material_index = index;
                }
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
					m_editor->m_project.save();
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
	ImGui_ImplGlfw_InitForVulkan(window->handle, true);
        set_imgui_theme();

	// Docking
	ImGuiIO &imgui_io = ImGui::GetIO();
	imgui_io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;
	imgui_io.ConfigWindowsMoveFromTitleBarOnly = true;

	auto font = std::make_pair(KOBRA_FONTS_DIR "/Frutiger/Frutiger.ttf", 18);
	m_ui = std::make_shared <kobra::layers::UserInterface> (
		get_context(), *window,
		graphics_queue, font,
		vk::AttachmentLoadOp::eClear
	);

	// Load scene
 	m_project.load_project(g_application.project);

	// m_scene.load(get_context(), project.scene);
	m_scene = m_project.load_scene(get_context());
	assert(m_scene.system);

        transform_daemon = std::make_shared <kobra::TransformDaemon> (m_scene.system.get());

	// IO callbacks
	io.mouse_events.subscribe(mouse_callback, this);
	io.keyboard_events.subscribe(keyboard_callback, this);

	// TODO: irradiance computer load from cache...

	// TODO: each layer that renders should have its own frmebuffer, or at
	// least a way to specify the image to render to (and then the layer
	// creates a framebuffer...)

	// Load all the renderers
	m_renderers.system = std::make_shared <kobra::amadeus::Accelerator> (transform_daemon.get());
	m_renderers.mesh_memory = std::make_shared <kobra::MeshDaemon> (get_context());
	m_viewport.sampler = kobra::make_continuous_sampler(device);

	// Attach UI layers
	m_material_editor = std::make_shared <MaterialEditor>
                (*device, material_preview, &m_texture_loader, m_scene.system->material_daemon);

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

        m_inspector = make_inspector(m_scene.system.get());

        // TODO: UI attachemtn that shows frametime as little chunks per frame

        // EditorViewport
        m_editor_renderer = std::make_shared <EditorViewport> (get_context(),
                m_renderers.system,
                m_renderers.mesh_memory,
                m_scene.system->material_daemon);

        load_environment_map(&m_editor_renderer->environment_map,
                             &m_texture_loader, environment_map_path);

	// Load and set the icon
	std::string icon_path = KOBRA_DIR "/kobra_icon.png";
	std::cout << "Loading icon from " << icon_path << std::endl;

	GLFWimage icon;
	stbi_set_flip_vertically_on_load(false);
	icon.pixels = stbi_load(icon_path.c_str(), &icon.width, &icon.height, nullptr, 4);
	glfwSetWindowIcon(window->handle, 1, &icon);
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
                // viewport camera is not an system entity
                editor->m_viewport.camera_transform_dirty = true;
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

	auto renderables_transforms = m_scene.system->tuples <kobra::Renderable, kobra::Transform> ();
	auto lights_transforms = m_scene.system->tuples <kobra::Light, kobra::Transform> ();

	auto system = m_scene.system;
	for (int i = 0; i < system->size(); i++) {
		if (system->exists <kobra::Renderable> (i)) {
			const auto *renderable = &system->get <kobra::Renderable> (i);
			const auto *transform = &system->get <kobra::Transform> (i);

			renderables.push_back(renderable);
			renderable_transforms.push_back(transform);
		}

		if (system->exists <kobra::Light> (i)) {
			const auto *light = &system->get <kobra::Light> (i);
			const auto *transform = &system->get <kobra::Transform> (i);

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
                render_info.camera_transform_dirty = m_viewport.camera_transform_dirty;
                // render_info.cmd = &cmd;
                render_info.extent = m_editor_renderer->extent;
                // render_info.framebuffer = &m_viewport.framebuffer;
                render_info.highlighted_entities = m_highlighted_entities;

                // TODO: pass the system itself...
                std::vector <Entity> renderable_entities = m_scene.system->tuple_entities <Renderable> ();
                const MaterialDaemon *md = system->material_daemon;
                m_editor_renderer->render(render_info, renderable_entities, *transform_daemon, md);

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
                render_material_preview(*cmd, material_preview, m_scene.system->material_daemon);

                RenderContext rc {
                        .cmd = cmd,
                        .framebuffer = framebuffer,
                        .extent = window->extent,
                        .render_area = kobra::RenderArea::full(),
                };

                layers::start_user_interface(m_ui.get(), rc, true);

                for (auto &attachment : m_ui->attachments)
                        attachment->render();

                render(m_inspector);

                layers::end_user_interface(m_ui.get(), rc);
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

                        std::vector <Entity> renderable_entities = editor->m_scene.system->tuple_entities <Renderable> ();
                        auto indices = editor->m_editor_renderer->selection_query(renderable_entities, fixed);

                        // Update the material editor
                        if (indices.size() == 0) {
                                editor->m_material_editor->material_index = -1;
                                editor->m_ui_attachments.viewport->set_transform();
                                editor->m_highlighted_entities.clear();
                        } else {
                                auto selection = indices[0];
                                kobra::Renderable &renderable = editor->m_scene.system->get <kobra::Renderable> (selection.first);
                                editor->m_ui_attachments.viewport->set_transform(editor->m_scene.system->get_entity(selection.first));
                                uint32_t material_index = renderable.material_indices[selection.second];
                                editor->m_material_editor->material_index = material_index;
                                editor->m_highlighted_entities = {(int) material_index};
                        }
                }
        } else if (request.type == InputRequest::eRelease) {
		glfwSetInputMode(editor->window->handle, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
                viewport.dragging = false;
        } else if (request.type == InputRequest::eDrag) {
                bool started_within = within(request.start, input_context.viewport);
                bool currently_within = within(request.position, input_context.viewport);

                if (started_within && (currently_within || viewport.dragging)) {
                        if (!viewport.dragging)
                                glfwSetInputMode(editor->window->handle, GLFW_CURSOR, GLFW_CURSOR_DISABLED);

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
                        editor->m_viewport.camera_transform_dirty = true;
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
		glfwSetInputMode(editor->window->handle, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
                mp.dragging = false;
        } else if (request.type == InputRequest::eDrag) {
                bool started_within = within(request.start, input_context.material_preview);
                bool currently_within = within(request.position, input_context.material_preview);

                if (started_within && (currently_within || mp.dragging)) {
                        if (!mp.dragging)
                                glfwSetInputMode(editor->window->handle, GLFW_CURSOR, GLFW_CURSOR_DISABLED);

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

void handle_application_communications(Editor *editor)
{
        while (!g_application.packets.empty()) {
                Packet packet = g_application.packets.front();
                g_application.packets.pop();

                std::cout << "Packet received: " << packet.header  << std::endl;

                if (packet.header == "select_entity") {
                        int32_t entity_id = packet.data[0];
                        select(editor->m_inspector, entity_id);
                }
        }
}

void Editor::after_present()
{
        // Reset dirty flags here
        m_viewport.camera_transform_dirty = false;

        // TODO: handle all requests
	if (!input_context.requests.empty()) {
		// TODO: ideally should only be one type of request per after_present
		InputRequest request = input_context.requests.front();
		input_context.requests.pop();

                handle_viewport_input(this, request);
                handle_material_preview_input(this, request);

                // Reset the input context
                input_context.requests = std::queue <InputRequest> ();
	}

        // Handle application communications
        handle_application_communications(this);

	// Ping all systems using materials
        // TODO: formalize material daemon
	// kobra::Material::daemon.ping_all();

        // Daemon update cycle
        transform_daemon->update();
        update(m_scene.system->material_daemon);
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

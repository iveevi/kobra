#include "../include/app.hpp"
#include "../include/backend.hpp"
#include "../include/layers/common.hpp"
#include "../include/layers/forward_renderer.hpp"
#include "../include/layers/image_renderer.hpp"
#include "../include/layers/objectifier.hpp"
#include "../include/layers/ui.hpp"
#include "../include/project.hpp"
#include "../include/scene.hpp"
#include "../include/shader_program.hpp"
#include "../include/ui/attachment.hpp"
#include "../include/engine/irradiance_computer.hpp"
#include "../include/amadeus/armada.cuh"
#include "../include/amadeus/path_tracer.cuh"
#include "../include/layers/framer.hpp"
#include "../include/cuda/color.cuh"
#include "../include/layers/denoiser.cuh"

// Forward declarations
struct ProgressBar;
struct InfoTab;
struct MaterialEditor;
struct RTXRenderer;
struct Viewport;

// TODO: add updated (emissive) materials as lights...

// TODO: logging attachment
// TODO: info tab that shows logging and framerate...
// TODO: viewport attachment

struct Editor : public kobra::BaseApp {
	kobra::Scene m_scene;
	kobra::Entity m_camera;

	kobra::layers::ForwardRenderer m_forward_renderer;
	kobra::layers::Objectifier m_objectifier;

	std::shared_ptr <kobra::layers::UI> m_ui;

	std::shared_ptr <ProgressBar> m_progress_bar;
	std::shared_ptr <InfoTab> m_info_tab;
	std::shared_ptr <MaterialEditor> m_material_editor;

	kobra::engine::IrradianceComputer m_irradiance_computer;
	bool m_saved_irradiance = false;

	// Renderers
	struct {
		std::shared_ptr <kobra::amadeus::System> system;
		std::shared_ptr <kobra::layers::MeshMemory> mesh_memory;
		std::shared_ptr <kobra::amadeus::ArmadaRTX> armada_rtx;

		kobra::layers::Denoiser denoiser;
		kobra::layers::Framer framer;

		std::mutex movement_mutex;
		std::queue <uint32_t> movement;

		int mode = 0;
		bool denoise = true;
	} m_renderers;

	// Viewport
	struct {
		kobra::ImageData image = nullptr;
		vk::raii::Sampler sampler = nullptr;
		ImVec2 min = {1/0.0f, 1/0.0f};
		ImVec2 max = {-1.0f, -1.0f};
	} m_viewport;

	// Buffers
	struct {
		CUdeviceptr traced;
		std::vector <uint8_t> traced_cpu;
	} m_buffers;

	struct Request {
		float x;
		float y;
	};

	std::queue <Request> request_queue;
	std::pair <int, int> m_selection = {-1, -1};

	// Input state
	// TODO: bring all other related fields here
	struct {
		bool viewport_hovered = false;
		bool viewport_focused = false;
	} m_input;

	Editor(const vk::raii::PhysicalDevice &, const std::vector <const char *> &);
	~Editor();

	void record(const vk::raii::CommandBuffer &, const vk::raii::Framebuffer&) override;
	void resize(const vk::Extent2D &) override;
	void after_present() override;

	static void mouse_callback(void *, const kobra::io::MouseEvent &);
	static void keyboard_callback(void *, const kobra::io::KeyboardEvent &);

	// TODO: frustrum culling structure to cull once per pass (store status
	// in a map) and then is passed to other layers for rendering
};

int main()
{
	// Load Vulkan physical device
	auto predicate = [](const vk::raii::PhysicalDevice &dev) {
		return kobra::physical_device_able(dev,  {
			VK_KHR_SWAPCHAIN_EXTENSION_NAME,
			VK_KHR_GET_MEMORY_REQUIREMENTS_2_EXTENSION_NAME,
			VK_KHR_EXTERNAL_MEMORY_EXTENSION_NAME,
			VK_KHR_EXTERNAL_SEMAPHORE_EXTENSION_NAME,
		});
	};

	vk::raii::PhysicalDevice phdev = kobra::pick_physical_device(predicate);

	Editor editor {
		phdev, {
			VK_KHR_SWAPCHAIN_EXTENSION_NAME,
			VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME,
			VK_KHR_EXTERNAL_MEMORY_EXTENSION_NAME,
		},
	};

	editor.run();
}

// Progress bar UI Attachment
struct ProgressBar : public kobra::ui::ImGuiAttachment {
	std::string m_title;
	float m_progress = 0.0f;

	ProgressBar(const std::string &title)
		: m_title {title} {}

	void render() override {
		// Set font size
		ImGui::Begin(m_title.c_str());
		ImGui::SetWindowSize(ImVec2(500, 100), ImGuiCond_FirstUseEver);
		ImGui::ProgressBar(m_progress);
		ImGui::End();
	}
};

// Info UI Attachment
struct InfoTab : public kobra::ui::ImGuiAttachment {
	std::vector <std::string> m_lines;
	std::string m_message;

	InfoTab() {
		// Attach logger handler
		kobra::add_log_handler(this,
			[&](const char *str, std::streamsize n) {
				m_message += std::string(str, n);

				std::string message_remainder = m_message;
				for (size_t i = 0; i < m_message.size(); i++) {
					if (m_message[i] == '\n') {
						m_lines.push_back(m_message.substr(0, i));
						message_remainder = m_message.substr(i + 1);
					}
				}
			}
		);
	}

	~InfoTab() {
		kobra::remove_log_handler(this);
	}

	void render() override {
		// Output and performance tabs
		ImGui::Begin("Info");

		ImGui::SetWindowSize(ImVec2(500, 500), ImGuiCond_FirstUseEver);

		// TODO: dock for framerate and performance
		ImGui::Text("Output");
		ImGui::Separator();

		for (const auto &line : m_lines)
			ImGui::Text(line.c_str());

		ImGui::End();
	}
};

// Material editor UI attachment
class MaterialEditor : public kobra::ui::ImGuiAttachment {
	int m_prev_material_index = -1;

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
			: m_editor {editor}, m_texture_loader {texture_loader} {}

	void render() override {
		ImGui::Begin("Material Editor");
		if (material_index < 0) {
			ImGui::End();
			return;
		}

		// Check if it is a new material
		bool is_not_loaded = m_prev_material_index != material_index;
		m_prev_material_index = material_index;

		// For starters, print material data
		ImGui::Text("Material data:");
		ImGui::Separator();

		kobra::Material *material = &kobra::Material::all[material_index];

		glm::vec3 diffuse = material->diffuse;
		glm::vec3 specular = material->specular;
		glm::vec3 ambient = material->ambient;
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

// RTX Renderer UI attachment
class RTXRenderer : public kobra::ui::ImGuiAttachment {
	Editor *m_editor = nullptr;
	int m_path_depth = 0;
public:
	RTXRenderer() = delete;
	RTXRenderer(Editor *editor) : m_editor {editor}, m_path_depth {2} {
		m_editor->m_renderers.armada_rtx->set_depth(m_path_depth);
	}

	void render() override {
		ImGui::Begin("RTX Renderer");

		// Setting the path depth
		if (ImGui::SliderInt("Path Depth", &m_path_depth, 0, 10)) {
			m_editor->m_renderers.armada_rtx->set_depth(m_path_depth);
			std::lock_guard <std::mutex> lock_guard
				(m_editor->m_renderers.movement_mutex);
			m_editor->m_renderers.movement.push(0);
		}

		// Checkboxes for enabling/disabling denoising
		ImGui::Checkbox("Denoise", &m_editor->m_renderers.denoise);

		// TODO: roussian roulette, different integrators, and loading
		// RTX attachments

		ImGui::End();
	}
};

// Viewport UI attachment
class Viewport : public kobra::ui::ImGuiAttachment {
	Editor *m_editor = nullptr;
	vk::DescriptorSet m_dset;
	vk::Image m_old_image = nullptr;
	float m_old_aspect = 0.0f;
public:
	Viewport() = delete;
	Viewport(Editor *editor) : m_editor {editor} {
		m_old_aspect = m_editor->m_camera.get <kobra::Camera> ().aspect;
	}

	void render() override {
		ImGui::Begin("Viewport");

		vk::Image image = *m_editor->m_viewport.image.image;
		if (image == m_old_image) {
			// Get current window size
			ImVec2 window_size = ImGui::GetWindowSize();

			// Pad
			constexpr float padding = 20.0f;
			window_size.x -= padding;
			window_size.y -= padding;

			// TODO: set the window aspect ratio
			ImGui::Image(m_dset, window_size);

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
				m_editor->m_camera.get <kobra::Camera> ().aspect = aspect;
				m_old_aspect = aspect;
			}
		} else {
			m_dset = ImGui_ImplVulkan_AddTexture(
				static_cast <VkSampler>
				(*m_editor->m_viewport.sampler),

				static_cast <VkImageView>
				(*m_editor->m_viewport.image.view),

				static_cast <VkImageLayout>
				(vk::ImageLayout::eShaderReadOnlyOptimal)
			);
		}

		m_old_image = image;
		ImGui::End();
	}
};

// Editor implementation
Editor::Editor(const vk::raii::PhysicalDevice &phdev,
		const std::vector <const char *> &extensions)
		: kobra::BaseApp {
			phdev, "Stress Test",
			vk::Extent2D {1500, 1000},
			extensions
		}
{
	// TODO: constructor should be loaded very fast, everything else should
	// be loaded as needed...
	int MIP_LEVELS = 5;

	// Load environment map
	// TODO: load HDR...
	kobra::ImageData &environment_map = m_texture_loader
		.load_texture(KOBRA_DIR "/resources/skies/background_1.jpg");

	m_irradiance_computer = kobra::engine::IrradianceComputer(
		get_context(), environment_map,
		MIP_LEVELS, 128,
		"irradiance_maps"
	);

	KOBRA_LOG_FUNC(kobra::Log::WARN) << "Starting irradiance computations...\n";

	// Load all the layers
	m_forward_renderer = kobra::layers::ForwardRenderer(get_context());
	m_objectifier = kobra::layers::Objectifier(get_context());

	// Configure ImGui
	ImGui::CreateContext();
	ImGui_ImplGlfw_InitForVulkan(window.handle, true);

	ImGuiIO &imgui_io = ImGui::GetIO();
	imgui_io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;
	imgui_io.ConfigWindowsMoveFromTitleBarOnly = true;

	auto font = std::make_pair(KOBRA_DIR "/resources/fonts/NotoSans.ttf", 18);
	m_ui = std::make_shared <kobra::layers::UI> (
		get_context(), window,
		graphics_queue, font,
		vk::AttachmentLoadOp::eClear
	);

	// Load scene
	kobra::Project project = kobra::Project::load(".kobra/project");
	m_scene.load(get_context(), project.scene);

	// TODO: Create a camera somewhere outside...
	// plus icons for lights and cameras
	m_camera = m_scene.ecs.get_entity("Camera");
	m_camera.get <kobra::Camera> ().aspect = 1.5f;

	// IO callbacks
	io.mouse_events.subscribe(mouse_callback, this);
	io.keyboard_events.subscribe(keyboard_callback, this);

	/* Create the image viewer
	std::vector <const kobra::ImageData *> images;
	for (int i = 0; i < MIP_LEVELS; i++)
		images.emplace_back(m_irradiance_computer.irradiance_maps[i]); */

	// TODO: irradiance computer load from cache...

	// Configure the forward renderer
	m_forward_renderer.add_pipeline(
		"environment",
		KOBRA_DIR "/source/shaders/environment_lighter.frag",
		{
			kobra::DescriptorSetLayoutBinding {
				5, vk::DescriptorType::eCombinedImageSampler,
				5, vk::ShaderStageFlagBits::eFragment
			}
		},
		[&](const vk::raii::DescriptorSet &descriptor_set) {
			m_irradiance_computer.bind(device, descriptor_set, 5);
		}
	);

	// Load all the renderers
	m_renderers.system = std::make_shared <kobra::amadeus::System> ();
	m_renderers.mesh_memory = std::make_shared <kobra::layers::MeshMemory> (get_context());

	constexpr vk::Extent2D raytracing_extent = {1000, 1000};
	m_renderers.armada_rtx = std::make_shared <kobra::amadeus::ArmadaRTX> (
		get_context(), m_renderers.system,
		m_renderers.mesh_memory, raytracing_extent
	);

	m_renderers.armada_rtx->attach(
		"Path Tracer",
		std::make_shared <kobra::amadeus::PathTracer> ()
	);

	m_renderers.armada_rtx->set_envmap(KOBRA_DIR "/resources/skies/background_1.jpg");

	// Create the denoiser layer
	m_renderers.denoiser = kobra::layers::Denoiser::make(
		raytracing_extent,
		kobra::layers::Denoiser::eNone
		// kobra::layers::Denoiser::eNormal
		//	| kobra::layers::Denoiser::eAlbedo
	);

	m_renderers.framer = kobra::layers::Framer(get_context());

	// Allocate necessary buffers
	size_t size = m_renderers.armada_rtx->size();
	m_buffers.traced = kobra::cuda::alloc(size * sizeof(uint32_t));
	m_buffers.traced_cpu.resize(size);

	// Allocate the viewport resources
	m_viewport.image = kobra::ImageData(
		phdev, device,
		swapchain.format, window.extent,
		vk::ImageTiling::eOptimal,
		vk::ImageUsageFlagBits::eColorAttachment
			| vk::ImageUsageFlagBits::eSampled
			| vk::ImageUsageFlagBits::eTransferDst,
		vk::MemoryPropertyFlagBits::eDeviceLocal,
		vk::ImageAspectFlagBits::eColor
	);

	m_viewport.sampler = kobra::make_sampler(device, m_viewport.image);

	// Attach UI layers
	m_progress_bar = std::make_shared <ProgressBar> ("Irradiance Computation Progress");
	m_info_tab = std::make_shared <InfoTab> ();
	m_material_editor = std::make_shared <MaterialEditor> (this, &m_texture_loader);

	// m_ui->attach(m_image_viewer);
	m_ui->attach(m_progress_bar);
	m_ui->attach(m_info_tab);
	m_ui->attach(m_material_editor);
	m_ui->attach(std::make_shared <RTXRenderer> (this));
	m_ui->attach(std::make_shared <Viewport> (this));
}

Editor::~Editor()
{
	device.waitIdle();

	// TODO: method for total destruction
	ImGui_ImplVulkan_Shutdown();
	ImGui_ImplGlfw_Shutdown();
	ImGui::DestroyContext();
}

void Editor::record(const vk::raii::CommandBuffer &cmd,
		const vk::raii::Framebuffer &framebuffer)
{
	// Camera movement
	if (m_input.viewport_focused) {
		auto &transform = m_camera.get <kobra::Transform> ();

		float speed = 20.0f * frame_time;

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
			std::lock_guard <std::mutex> lock(m_renderers.movement_mutex);
			m_renderers.movement.push(0);
		}
	}

	std::vector <const kobra::Renderable *> renderables;
	std::vector <const kobra::Transform *> renderable_transforms;

	std::vector <const kobra::Light *> lights;
	std::vector <const kobra::Transform *> light_transforms;

	auto renderables_transforms = m_scene.ecs.tuples <kobra::Renderable, kobra::Transform> ();
	auto lights_transforms = m_scene.ecs.tuples <kobra::Light, kobra::Transform> ();

	auto ecs = m_scene.ecs;

	for (int i = 0; i < ecs.size(); i++) {
		if (ecs.exists <kobra::Renderable> (i)) {
			const auto *renderable = &ecs.get <kobra::Renderable> (i);
			const auto *transform = &ecs.get <kobra::Transform> (i);

			renderables.push_back(renderable);
			renderable_transforms.push_back(transform);
		}

		if (ecs.exists <kobra::Light> (i)) {
			const auto *light = &ecs.get <kobra::Light> (i);
			const auto *transform = &ecs.get <kobra::Transform> (i);

			lights.push_back(light);
			light_transforms.push_back(transform);
		}
	}

	kobra::layers::ForwardRenderer::Parameters params {
		.renderables = renderables_transforms,
		.lights = lights_transforms,
		.pipeline_package = "environment",
	};

	params.environment_map = KOBRA_DIR "/resources/skies/background_1.jpg";

	cmd.begin({});
		// TODO: also see the normal and albedo and depth buffers from
		// deferred renderer
		// TODO: drop down menu for selecting the renderer
		if (m_renderers.mode) {
			bool accumulate = m_renderers.movement.empty();

			{
				// Clear queue
				std::lock_guard <std::mutex> lock(m_renderers.movement_mutex);
				m_renderers.movement = std::queue <uint32_t> ();
			}

			m_renderers.armada_rtx->render(
				m_scene.ecs,
				m_camera.get <kobra::Camera> (),
				m_camera.get <kobra::Transform> (),
				accumulate
			);

			float4 *buffer = (float4 *) m_renderers.armada_rtx->color_buffer();
			if (m_renderers.denoise) {
				kobra::layers::denoise(m_renderers.denoiser, {
					.color = (CUdeviceptr) m_renderers.armada_rtx->color_buffer(),
					.normal = (CUdeviceptr) m_renderers.armada_rtx->normal_buffer(),
					.albedo = (CUdeviceptr) m_renderers.armada_rtx->albedo_buffer()
				});

				buffer = (float4 *) m_renderers.denoiser.result;
			}

			vk::Extent2D rtx_extent = m_renderers.armada_rtx->extent();

			kobra::cuda::hdr_to_ldr(
				buffer,
				(uint32_t *) m_buffers.traced,
				rtx_extent.width, rtx_extent.height,
				kobra::cuda::eTonemappingACES
			);

			kobra::cuda::copy(
				m_buffers.traced_cpu, m_buffers.traced,
				m_renderers.armada_rtx->size() * sizeof(uint32_t)
			);

			// TODO: import CUDA to Vulkan and render straight to the image
			m_renderers.framer.render(
				kobra::RawImage {
					.data = m_buffers.traced_cpu,
					.width = rtx_extent.width,
					.height = rtx_extent.height,
					.channels = 4
				},
				cmd, framebuffer, window.extent
			);
		} else {
			m_forward_renderer.render(
				params,
				m_camera.get <kobra::Camera> (),
				m_camera.get <kobra::Transform> (),
				cmd, framebuffer, window.extent
			);
		}

		m_irradiance_computer.sample(cmd);
		/* if (m_irradiance_computer.sample(cmd)
				&& !m_irradiance_computer.cached
				&& !m_saved_irradiance) {
			m_irradiance_computer.save_irradiance_maps(
				get_context(),
				"irradiance_maps"
			);

			m_saved_irradiance = true;
		} */

		// TODO: progress bar...
		// std::cout << "Sample count: " << m_irradiance_computer.samples << std::endl;
		m_progress_bar->m_progress = m_irradiance_computer.samples/128.0f;

		// Handle requests
		std::optional <Request> selection_request;
		while (!request_queue.empty()) {
			Request request = request_queue.front();
			request_queue.pop();

			selection_request = request;
		}

		if (selection_request) {
			m_objectifier.render(
				cmd,
				// TODO: pass extent...
				m_scene.ecs,
				m_camera.get <kobra::Camera> (),
				m_camera.get <kobra::Transform> ()
			);

			request_queue.push(*selection_request);
		}

		// If there is a selection, highlight it
		if (m_selection.first >= 0 && m_selection.second >= 0) {
			// TODO: only render the selected objetcs...
			// otherwise computtion becomes very wasted...
			m_objectifier.composite_highlight(
				cmd, framebuffer, window.extent,
				m_scene.ecs,
				m_camera.get <kobra::Camera> (),
				m_camera.get <kobra::Transform> (),
				m_selection
			);
		}

		// Copy framebuffer image to viewport image
		vk::ImageCopy copy_region {
			{vk::ImageAspectFlagBits::eColor, 0, 0, 1},
			{0, 0, 0},
			{vk::ImageAspectFlagBits::eColor, 0, 0, 1},
			{0, 0, 0},
			{window.extent.width, window.extent.height, 1}
		};

		// Transition layouts
		vk::ImageMemoryBarrier swapchain_barrier {
			{},
			vk::AccessFlagBits::eTransferWrite,
			vk::ImageLayout::ePresentSrcKHR,
			vk::ImageLayout::eTransferSrcOptimal,
			VK_QUEUE_FAMILY_IGNORED,
			VK_QUEUE_FAMILY_IGNORED,
			swapchain.images[frame_index],
			{vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1}
		};

		vk::ImageMemoryBarrier viewport_barrier {
			{},
			vk::AccessFlagBits::eTransferWrite,
			m_viewport.image.layout,
			vk::ImageLayout::eTransferDstOptimal,
			VK_QUEUE_FAMILY_IGNORED,
			VK_QUEUE_FAMILY_IGNORED,
			*m_viewport.image.image,
			{vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1}
		};

		cmd.pipelineBarrier(
			vk::PipelineStageFlagBits::eTopOfPipe,
			vk::PipelineStageFlagBits::eTransfer,
			{}, {}, {},
			{swapchain_barrier, viewport_barrier}
		);

		cmd.copyImage(
			swapchain.images[frame_index],
			vk::ImageLayout::eTransferSrcOptimal,
			*m_viewport.image.image,
			vk::ImageLayout::eTransferDstOptimal,
			copy_region
		);

		// Transition layouts back
		swapchain_barrier.srcAccessMask = vk::AccessFlagBits::eTransferWrite;
		swapchain_barrier.dstAccessMask = vk::AccessFlagBits::eMemoryRead;
		swapchain_barrier.oldLayout = vk::ImageLayout::eTransferSrcOptimal;
		swapchain_barrier.newLayout = vk::ImageLayout::ePresentSrcKHR;

		viewport_barrier.srcAccessMask = vk::AccessFlagBits::eTransferWrite;
		viewport_barrier.dstAccessMask = vk::AccessFlagBits::eShaderRead;
		viewport_barrier.oldLayout = vk::ImageLayout::eTransferDstOptimal;
		viewport_barrier.newLayout = vk::ImageLayout::eShaderReadOnlyOptimal;

		cmd.pipelineBarrier(
			vk::PipelineStageFlagBits::eTransfer,
			vk::PipelineStageFlagBits::eFragmentShader,
			{}, {}, {},
			{swapchain_barrier, viewport_barrier}
		);

		// Render the UI last
		m_ui->render(cmd,
			framebuffer, window.extent,
			kobra::RenderArea::full(), {true}
		);
	cmd.end();

	// TODO: after present actions...
}

void Editor::resize(const vk::Extent2D &extent)
{
	// m_camera.get <kobra::Camera> ().aspect = extent.width / (float) extent.height;
	// TODO: resize the objectifier...

	// Resize the viewport image
	// m_viewport.image.resize(extent); TODO: implement this

	m_viewport.image = kobra::ImageData(
		phdev, device,
		swapchain.format, extent,
		vk::ImageTiling::eOptimal,
		vk::ImageUsageFlagBits::eColorAttachment
			| vk::ImageUsageFlagBits::eSampled
			| vk::ImageUsageFlagBits::eTransferDst,
		vk::MemoryPropertyFlagBits::eDeviceLocal,
		vk::ImageAspectFlagBits::eColor
	);

	m_viewport.sampler = kobra::make_sampler(device, m_viewport.image);
}

void Editor::after_present()
{
	if (!request_queue.empty()) {
		// TODO: ideally should only be one type of request per after_present
		Request request = request_queue.front();
		request_queue.pop();

		ImVec2 min = m_viewport.min;
		ImVec2 max = m_viewport.max;

		ImVec2 fixed {
			(request.x - min.x) / (max.x - min.x),
			(request.y - min.y) / (max.y - min.y)
		};

		fixed.x *= window.extent.width;
		fixed.y *= window.extent.height;

		// TODO: get coordinates of the viewport image...
		auto ids = m_objectifier.query(fixed.x, fixed.y);
		m_selection = {int(ids.first) - 1, int(ids.second) - 1};

		// Update the material editor
		if (m_selection.first < 0 || m_selection.second < 0) {
			m_material_editor->material_index = -1;
		} else {
			kobra::Renderable &renderable = m_scene.ecs
				.get <kobra::Renderable> (m_selection.first);

			uint32_t material_index = renderable.material_indices[m_selection.second];
			m_material_editor->material_index = material_index;
		}
	}

	// Ping all systems using materials
	kobra::Material::daemon.ping_all();
}

void Editor::mouse_callback(void *us, const kobra::io::MouseEvent &event)
{
	static const int select_button = GLFW_MOUSE_BUTTON_LEFT;

	// Check if selecting
	if (event.action == GLFW_PRESS && event.button == select_button) {
		Editor *editor = static_cast <Editor *> (us);
		// TODO: this needs to query the position on the viewport
		// image...
		editor->request_queue.push({
			float(event.xpos),
			float(event.ypos)
		});
	}

	// Panning around
	static const int pan_button = GLFW_MOUSE_BUTTON_RIGHT;

	static const float sensitivity = 0.001f;

	static float px = 0.0f;
	static float py = 0.0f;

	static float yaw = 0.0f;
	static float pitch = 0.0f;

	// Deltas and directions
	float dx = event.xpos - px;
	float dy = event.ypos - py;

	// Check if panning
	static bool dragging = false;
	static bool alt_dragging = false;

	Editor *editor = static_cast <Editor *> (us);
	bool is_drag_button = (event.button == pan_button);
	if (event.action == GLFW_PRESS && is_drag_button && editor->m_input.viewport_hovered) {
		dragging = true;
		glfwSetInputMode(editor->window.handle, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
	} else if (event.action == GLFW_RELEASE && is_drag_button && !editor->m_input.viewport_hovered) {
		dragging = false;
		glfwSetInputMode(editor->window.handle, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
	}

	bool is_alt_down = editor->io.input->is_key_down(GLFW_KEY_LEFT_ALT);
	if (!alt_dragging && is_alt_down && editor->m_input.viewport_hovered) {
		alt_dragging = true;
		glfwSetInputMode(editor->window.handle, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
	} else if (alt_dragging && !is_alt_down && !editor->m_input.viewport_hovered) {
		alt_dragging = false;
		glfwSetInputMode(editor->window.handle, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
	}

	// Pan only when dragging
	if (dragging | alt_dragging) {
		yaw -= dx * sensitivity;
		pitch -= dy * sensitivity;

		if (pitch > 89.0f)
			pitch = 89.0f;
		if (pitch < -89.0f)
			pitch = -89.0f;

		kobra::Transform &transform = editor->m_camera.get <kobra::Transform> ();
		transform.rotation.x = pitch;
		transform.rotation.y = yaw;

		std::lock_guard <std::mutex> lock(editor->m_renderers.movement_mutex);
		editor->m_renderers.movement.push(0);
	}

	// Update previous position
	px = event.xpos;
	py = event.ypos;
}

void Editor::keyboard_callback(void *us, const kobra::io::KeyboardEvent &event)
{
	Editor *editor = static_cast <Editor *> (us);
	if (event.action == GLFW_PRESS) {
		if (event.key == GLFW_KEY_TAB)
			editor->m_renderers.mode = !editor->m_renderers.mode;
		if (event.key == GLFW_KEY_ESCAPE) {
			editor->m_selection = {-1, -1};
			editor->m_material_editor->material_index = -1;
		}
	}
}

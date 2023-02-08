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
struct ImageViewer;
struct ProgressBar;
struct InfoTab;
struct MaterialEditor;

// TODO: logging attachment
// TODO: info tab that shows logging and framerate...
// TODO: viewport attachment

struct Editor : public kobra::BaseApp {
	kobra::Scene m_scene;
	kobra::Entity m_camera;

	kobra::layers::ForwardRenderer m_forward_renderer;
	kobra::layers::Objectifier m_objectifier;

	std::shared_ptr <kobra::layers::UI> m_ui;

	std::shared_ptr <ImageViewer> m_image_viewer;
	std::shared_ptr <ProgressBar> m_progress_bar;
	std::shared_ptr <InfoTab> m_info_tab;
	std::shared_ptr <MaterialEditor> m_material_editor;

	kobra::engine::IrradianceComputer m_irradiance_computer;
	bool m_saved_irradiance = false;

	// Renderers
	struct {
		std::shared_ptr <kobra::amadeus::System> system;
		std::shared_ptr <kobra::layers::MeshMemory> mesh_memory;

		kobra::amadeus::ArmadaRTX armada_rtx;
		kobra::layers::Denoiser denoiser;
		kobra::layers::Framer framer;

		std::mutex movement_mutex;
		std::queue <uint32_t> movement;

		int mode = 0;
	} m_renderers;

	// Buffers
	struct {
		CUdeviceptr traced;
		std::vector <uint8_t> traced_cpu;
	} m_buffers;

	struct Request {
		double x;
		double y;
	};

	std::queue <Request> request_queue;
	std::pair <int, int> m_selection = {-1, -1};

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

// Image viewer UI attachment
struct ImageViewer : public kobra::ui::ImGuiAttachment {
	std::vector <kobra::ImageData *> m_images;
	std::vector <vk::DescriptorSet> m_descriptor_sets;
	int m_current_image = 0;

	ImageViewer(const kobra::Context &context, const std::vector <const kobra::ImageData *> &images)
	{
		for (const auto &image : images)
			m_images.push_back(const_cast <kobra::ImageData *> (image));

		for (const auto &image : m_images) {
			// Make sure layout is shader read only
			if (image->layout != vk::ImageLayout::eShaderReadOnlyOptimal) {
				vk::raii::Queue temp_queue {*context.device, 0, 0};
				kobra::submit_now(*context.device, temp_queue, *context.command_pool,
					[&](const vk::raii::CommandBuffer &cmd) {
						image->transition_layout(
							cmd,
							vk::ImageLayout::eShaderReadOnlyOptimal
						);
					}
				);
			}

			vk::raii::Sampler sampler = make_sampler(*context.device, *image);
			m_descriptor_sets.push_back(
				ImGui_ImplVulkan_AddTexture(
					static_cast <VkSampler> (*sampler),
					static_cast <VkImageView> (*image->view),
					static_cast <VkImageLayout> (image->layout)
				)
			);
		}
	}

	void render() override {
		ImGui::Begin("Image Viewer");
		ImGui::SetWindowSize(ImVec2(500, 500), ImGuiCond_FirstUseEver);
		ImGui::Image(m_descriptor_sets[m_current_image], ImVec2(500, 500), ImVec2(0, 1), ImVec2(1, 0));
		ImGui::SliderInt("Image", &m_current_image, 0, m_images.size() - 1);
		ImGui::End();
	}
};

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
	kobra::Material *m_prev_material = nullptr;

	vk::DescriptorSet m_diffuse_set;
	vk::DescriptorSet m_normal_set;

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
	kobra::Material *m_material = nullptr;

	MaterialEditor() = delete;
	MaterialEditor(const kobra::Context &context)
			: m_texture_loader {context.texture_loader} {}

	void render() override {
		if (!m_material)
			return;

		ImGui::Begin("Material Editor");
		ImGui::SetWindowSize(ImVec2(500, 500), ImGuiCond_FirstUseEver);

		// For starters, print material data
		ImGui::Text("Material data:");
		ImGui::Separator();

		glm::vec3 diffuse = m_material->diffuse;
		glm::vec3 specular = m_material->specular;
		glm::vec3 ambient = m_material->ambient;
		glm::vec3 emission = m_material->emission;

		ImGui::Text("Albedo: %f, %f, %f", diffuse.r, diffuse.g, diffuse.b);
		ImGui::Text("Specular: %f, %f, %f", specular.r, specular.g, specular.b);
		ImGui::Text("Ambient: %f, %f, %f", ambient.r, ambient.g, ambient.b);
		ImGui::Text("Emission: %f, %f, %f", emission.r, emission.g, emission.b);

		ImGui::Separator();

		bool is_not_loaded = m_prev_material != m_material;
		m_prev_material = m_material;

		if (m_material->has_albedo()) {
			ImGui::Text("Diffuse Texture:");

			std::string diffuse_path = m_material->albedo_texture;
			if (is_not_loaded)
				m_diffuse_set = imgui_allocate_image(diffuse_path);

			ImGui::Image(m_diffuse_set, ImVec2(256, 256));
			ImGui::Separator();
		}

		if (m_material->has_normal()) {
			ImGui::Text("Normal Texture:");

			std::string normal_path = m_material->normal_texture;
			if (is_not_loaded)
				m_normal_set = imgui_allocate_image(normal_path);

			ImGui::Image(m_normal_set, ImVec2(256, 256));
			ImGui::Separator();
		}

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

	ImGui::CreateContext();
	ImGui_ImplGlfw_InitForVulkan(window.handle, true);

	auto font = std::make_pair(KOBRA_DIR "/resources/fonts/NotoSans.ttf", 18);
	m_ui = std::make_shared <kobra::layers::UI> (
		get_context(), window,
		graphics_queue, font
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

	// Attach UI layers
	// m_image_viewer = std::make_shared <ImageViewer> (get_context(), images);
	m_progress_bar = std::make_shared <ProgressBar> ("Irradiance Computation Progress");
	m_info_tab = std::make_shared <InfoTab> ();
	m_material_editor = std::make_shared <MaterialEditor> (get_context());

	// m_ui->attach(m_image_viewer);
	m_ui->attach(m_progress_bar);
	m_ui->attach(m_info_tab);
	m_ui->attach(m_material_editor);

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
	m_renderers.armada_rtx = kobra::amadeus::ArmadaRTX(
		get_context(), m_renderers.system,
		m_renderers.mesh_memory, raytracing_extent
	);

	m_renderers.armada_rtx.attach(
		"Path Tracer",
		std::make_shared <kobra::amadeus::PathTracer> ()
	);

	m_renderers.armada_rtx.set_envmap(KOBRA_DIR "/resources/skies/background_1.jpg");

	// Create the denoiser layer
	m_renderers.denoiser = kobra::layers::Denoiser::make(
		raytracing_extent,
		kobra::layers::Denoiser::eNone
		// kobra::layers::Denoiser::eNormal
		//	| kobra::layers::Denoiser::eAlbedo
	);

	m_renderers.framer = kobra::layers::Framer(get_context());

	// Allocate necessary buffers
	size_t size = m_renderers.armada_rtx.size();
	m_buffers.traced = kobra::cuda::alloc(size * sizeof(uint32_t));
	m_buffers.traced_cpu.resize(size);
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

			m_renderers.armada_rtx.render(
				m_scene.ecs,
				m_camera.get <kobra::Camera> (),
				m_camera.get <kobra::Transform> (),
				accumulate
			);

			// TODO:enable/disable denoiser
			kobra::layers::denoise(m_renderers.denoiser, {
				.color = (CUdeviceptr) m_renderers.armada_rtx.color_buffer(),
				.normal = (CUdeviceptr) m_renderers.armada_rtx.normal_buffer(),
				.albedo = (CUdeviceptr) m_renderers.armada_rtx.albedo_buffer()
			});

			vk::Extent2D rtx_extent = m_renderers.armada_rtx.extent();

			kobra::cuda::hdr_to_ldr(
				(float4 *) m_renderers.denoiser.result,
				(uint32_t *) m_buffers.traced,
				rtx_extent.width, rtx_extent.height,
				kobra::cuda::eTonemappingACES
			);

			kobra::cuda::copy(
				m_buffers.traced_cpu, m_buffers.traced,
				m_renderers.armada_rtx.size() * sizeof(uint32_t)
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
			m_objectifier.composite_highlight(
				cmd, framebuffer, window.extent,
				m_scene.ecs,
				m_camera.get <kobra::Camera> (),
				m_camera.get <kobra::Transform> (),
				m_selection
			);
		}

		// Render the UI last
		m_ui->render(cmd, framebuffer, window.extent);
	cmd.end();

	// TODO: after present actions...
}

void Editor::resize(const vk::Extent2D &extent)
{
	m_camera.get <kobra::Camera> ().aspect = extent.width / (float) extent.height;
	// TODO: resize the objectifier...
}

void Editor::after_present()
{
	if (!request_queue.empty()) {
		// TODO: ideally should only be one type of request per after_present
		Request request = request_queue.front();
		request_queue.pop();

		auto ids = m_objectifier.query(request.x, request.y);
		m_selection = {int(ids.first) - 1, int(ids.second) - 1};

		// Update the material editor
		if (m_selection.first < 0 || m_selection.second < 0) {
			m_material_editor->m_material = nullptr;
		} else {
			kobra::Renderable &renderable = m_scene.ecs
				.get <kobra::Renderable> (m_selection.first);

			uint32_t material_index = renderable.material_indices[m_selection.second];
			m_material_editor->m_material = &kobra::Material::all[material_index];
		}
	}
}

void Editor::mouse_callback(void *us, const kobra::io::MouseEvent &event)
{
	static const int select_button = GLFW_MOUSE_BUTTON_LEFT;

	// Check if selecting
	if (event.action == GLFW_PRESS && event.button == select_button) {
		Editor *editor = static_cast <Editor *> (us);
		editor->request_queue.push({event.xpos, event.ypos});
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

	bool is_drag_button = (event.button == pan_button);
	if (event.action == GLFW_PRESS && is_drag_button)
		dragging = true;
	else if (event.action == GLFW_RELEASE && is_drag_button)
		dragging = false;

	Editor *editor = static_cast <Editor *> (us);
	bool is_alt_down = editor->io.input->is_key_down(GLFW_KEY_LEFT_ALT);
	if (!alt_dragging && is_alt_down)
		alt_dragging = true;
	else if (alt_dragging && !is_alt_down)
		alt_dragging = false;

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
	}
}

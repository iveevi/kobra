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

struct Editor : public kobra::BaseApp {
	kobra::Scene m_scene;
	kobra::Entity m_camera;
	kobra::layers::ForwardRenderer m_forward_renderer;
	kobra::layers::ImageRenderer m_image_renderer;
	kobra::layers::Objectifier m_objectifier;

	Editor(const vk::raii::PhysicalDevice &, const std::vector <const char *> &);

	void record(const vk::raii::CommandBuffer &, const vk::raii::Framebuffer &);
	void resize(const vk::Extent2D &);

	static void mouse_callback(void *, const kobra::io::MouseEvent &);
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
		phdev,
		{VK_KHR_SWAPCHAIN_EXTENSION_NAME},
	};

	editor.run();
}

constexpr const char *irradiance_computer_str = R"(
#version 450

layout (binding = 0) uniform sampler2D environment_map;
layout (binding = 1) writeonly uniform image2D irradiance_map;

// Push constants
layout (push_constant) uniform PushConstants {
	float roughness;
	int width;
	int height;
} push_constants;

void main()
{
	// Get the coordinates of the pixel
	ivec2 coords = ivec2(gl_GlobalInvocationID.xy);

	// Get the pixel's position in the texture
	vec2 uv = vec2(coords) / vec2(push_constants.width, push_constants.height);

	// Get the pixel's direction
	vec3 direction = normalize(vec3(uv, 1.0));

	// Get the pixel's color
	vec3 color = texture(environment_map, uv).rgb;

	// Write the color to the irradiance map
	imageStore(irradiance_map, coords, vec4(color, 1.0) + vec4(0.5, 0, 0, 0.0));
}
)";

// Editor implementation
Editor::Editor(const vk::raii::PhysicalDevice &phdev,
		const std::vector <const char *> &extensions)
		: kobra::BaseApp {
			phdev, "Stress Test",
			vk::Extent2D {1500, 1000},
			extensions
		}
{
	// Load all the layers
	m_forward_renderer = kobra::layers::ForwardRenderer(get_context());
	m_image_renderer = kobra::layers::ImageRenderer(get_context());

	kobra::Project project = kobra::Project::load(".kobra/project");
	m_scene.load(get_context(), project.scene);
	
	m_camera = m_scene.ecs.get_entity("Camera");
	m_camera.get <kobra::Camera> ().aspect = 1.5f;
	
	// Mouse callbacks
	io.mouse_events.subscribe(mouse_callback, this);

	// Load the compute shader
	kobra::ShaderProgram irradiance_computer {
		irradiance_computer_str,
		vk::ShaderStageFlagBits::eCompute
	};

	vk::raii::ShaderModule opt_irradiance_computer = std::move(*irradiance_computer.compile(device));

	// Load environment map
	m_texture_loader.load_texture(KOBRA_DIR "/resources/skies/background_1.jpg");
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

	if (io.input->is_key_down(GLFW_KEY_W))
		transform.move(forward * speed);
	else if (io.input->is_key_down(GLFW_KEY_S))
		transform.move(-forward * speed);

	if (io.input->is_key_down(GLFW_KEY_A))
		transform.move(-right * speed);
	else if (io.input->is_key_down(GLFW_KEY_D))
		transform.move(right * speed);

	if (io.input->is_key_down(GLFW_KEY_E))
		transform.move(up * speed);
	else if (io.input->is_key_down(GLFW_KEY_Q))
		transform.move(-up * speed);
	
	std::vector <const kobra::Renderable *> renderables;
	std::vector <const kobra::Transform *> renderable_transforms;

	std::vector <const kobra::Light *> lights;
	std::vector <const kobra::Transform *> light_transforms;

	auto renderables_transforms = m_scene.ecs.tuples <kobra::Renderable, kobra::Transform> ();
	auto lights_transforms = m_scene.ecs.tuples <kobra::Light, kobra::Transform> ();
	
	// std::cout << "renderables_transforms size: " << renderables_transforms.size() << std::endl;
	// std::cout << "lights_transforms size: " << lights_transforms.size() << std::endl;

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
	};

	cmd.begin({});
		m_forward_renderer.render(
			params,
			m_camera.get <kobra::Camera> (),
			m_camera.get <kobra::Transform> (),
			cmd, framebuffer, window.extent
		);

		m_image_renderer.render(
			m_texture_loader.load_texture(
				KOBRA_DIR "/resources/skies/background_1.jpg"
			),
			kobra::RenderContext {
				cmd, framebuffer,
				window.extent,
				{{100, 100}, {300, 300}}
			}
		);
	cmd.end();
}

void Editor::resize(const vk::Extent2D &extent)
{
	m_camera.get <kobra::Camera> ().aspect = extent.width / (float) extent.height;
}
		
void Editor::mouse_callback(void *us, const kobra::io::MouseEvent &event)
{
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
	}

	// Update previous position
	px = event.xpos;
	py = event.ypos;
}

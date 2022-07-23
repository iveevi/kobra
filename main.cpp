#include "include/app.hpp"
#include "include/backend.hpp"
#include "include/button.hpp"
#include "include/common.hpp"
#include "include/ecs.hpp"
#include "include/engine/ecs_panel.hpp"
#include "include/io/event.hpp"
#include "include/layers/font_renderer.hpp"
#include "include/layers/raster.hpp"
#include "include/layers/raytracer.hpp"
#include "include/layers/shape_renderer.hpp"
#include "include/logger.hpp"
#include "include/renderer.hpp"
#include "include/transform.hpp"
#include "include/types.hpp"
#include "tinyfiledialogs.h"

using namespace kobra;

// Scene path
std::string scene_path = "scenes/scene.kobra";

// Scene class
struct Scene {
	ECS ecs;

	// Other scene-local data
	std::string p_environment_map;

	// Saving and loading
	void save(const std::string &);
	void load(const Device &, const std::string &);
};

// Test app
struct ECSApp : public BaseApp {
	layers::Raster	rasterizer;
	layers::Raytracer raytracer;
	layers::FontRenderer font_renderer;
	layers::ShapeRenderer shape_renderer;
	engine::ECSPanel panel;

	::Scene scene;

	Entity camera;
	Button button;

	ECSApp(const vk::raii::PhysicalDevice &phdev, const std::vector <const char *> &extensions)
			: BaseApp(phdev, "ECSApp", {1000, 1000}, extensions, vk::AttachmentLoadOp::eLoad),
			rasterizer(get_context(), vk::AttachmentLoadOp::eClear),
			raytracer(get_context(), &sync_queue, vk::AttachmentLoadOp::eClear),
			font_renderer(get_context(), render_pass, "resources/fonts/noto_sans.ttf"),
			shape_renderer(get_context(), render_pass),
			panel(get_context(), scene.ecs, io) {
		scene.load(get_device(), scene_path);
		raytracer.environment_map(scene.p_environment_map);
		camera = scene.ecs.get_entity("Camera");

		// Input callbacks
		io.mouse_events.subscribe(mouse_callback, this);

		scene.ecs.info <Mesh> ();
	}

	int mode = 0;	// 0 for raster, 1 for raytracer
	bool tab_pressed = false;

	void active_input() {
		float speed = 20.0f * frame_time;

		// Camera movement
		// TODO: remove transform component from camera?
		auto &cam = camera.get <Camera> ();

		glm::vec3 forward = cam.transform.forward();
		glm::vec3 right = cam.transform.right();
		glm::vec3 up = cam.transform.up();

		if (io.input.is_key_down(GLFW_KEY_W))
			cam.transform.move(forward * speed);
		else if (io.input.is_key_down(GLFW_KEY_S))
			cam.transform.move(-forward * speed);

		if (io.input.is_key_down(GLFW_KEY_A))
			cam.transform.move(-right * speed);
		else if (io.input.is_key_down(GLFW_KEY_D))
			cam.transform.move(right * speed);

		if (io.input.is_key_down(GLFW_KEY_E))
			cam.transform.move(up * speed);
		else if (io.input.is_key_down(GLFW_KEY_Q))
			cam.transform.move(-up * speed);

		// Switch mode on tab
		if (io.input.is_key_down(GLFW_KEY_TAB)) {
			if (!tab_pressed) {
				tab_pressed = true;
				mode = (mode + 1) % 2;
			}
		} else {
			tab_pressed = false;
		}
	}

	float fps = 0;
	float time = 0;

	void record(const vk::raii::CommandBuffer &cmd,
			const vk::raii::Framebuffer &framebuffer) override {
		if (frame_time > 0)
			fps = (fps + 1.0f/frame_time) / 2.0f;

		std::vector <Text> texts {
			Text {
				.text = common::sprintf("%.2f fps", fps),
				.anchor = {10, 10},
				.size = 1.0f
			}
		};

		std::vector <Rect> rects {
			Rect {.min = {500, 500}, .max = {600, 600}, .color = {1, 0, 0}},
			button.shape()
		};

		time += frame_time;

		// Input
		active_input();

		// Begin command buffer
		cmd.begin({});

		if (mode == 1)
			raytracer.render(cmd, framebuffer, scene.ecs);
		else
			rasterizer.render(cmd, framebuffer, scene.ecs);

		// Start render pass
		std::array <vk::ClearValue, 2> clear_values = {
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
				*framebuffer,
				vk::Rect2D {
					vk::Offset2D {0, 0},
					extent,
				},
				static_cast <uint32_t> (clear_values.size()),
				clear_values.data()
			},
			vk::SubpassContents::eInline
		);

		font_renderer.render(cmd, texts);

		cmd.endRenderPass();

		cmd.end();
	}

	void terminate() override {
		if (io.input.is_key_down(GLFW_KEY_ESCAPE))
			glfwSetWindowShouldClose(window.handle, true);
	}

	// Mouse callback
	static void mouse_callback(void *us, const io::MouseEvent &event) {
		static const int pan_button = GLFW_MOUSE_BUTTON_MIDDLE;
		static const int alt_pan_button = GLFW_MOUSE_BUTTON_LEFT;
		static const int select_button = GLFW_MOUSE_BUTTON_LEFT;

		static const float sensitivity = 0.001f;

		static bool first_movement = true;
		static bool dragging = false;
		static bool dragging_select = false;
		static bool gizmo_dragging = false;

		static float px = 0.0f;
		static float py = 0.0f;

		static glm::vec2 previous_dir {0.0f, 0.0f};

		static float yaw = 0.0f;
		static float pitch = 0.0f;

		auto &app = *static_cast <ECSApp *> (us);
		auto &cam = app.camera.get <Camera> ();

		// Deltas and directions
		float dx = event.xpos - px;
		float dy = event.ypos - py;
		glm::vec2 dir {dx, dy};

		// Dragging only with the drag button
		// TODO: alt left dragging as ewll
		bool is_drag_button = (event.button == pan_button);
		if (event.action == GLFW_PRESS && is_drag_button)
			dragging = true;
		else if (event.action == GLFW_RELEASE && is_drag_button)
			dragging = false;

		// Pan only when draggign
		if (dragging) {
			yaw -= dx * sensitivity;
			pitch -= dy * sensitivity;

			if (pitch > 89.0f)
				pitch = 89.0f;
			if (pitch < -89.0f)
				pitch = -89.0f;

			cam.transform.rotation.x = pitch;
			cam.transform.rotation.y = yaw;
		}

		// Update previous position
		px = event.xpos;
		py = event.ypos;

		previous_dir = dir;
	}
};

int main()
{
	auto extensions = {
		VK_KHR_SWAPCHAIN_EXTENSION_NAME,
	};

	auto predicate = [&extensions](const vk::raii::PhysicalDevice &dev) {
		return physical_device_able(dev, extensions);
	};

	// Choose a physical device
	// TODO: static lambda (FIRST)
	auto phdev = pick_physical_device(predicate);

	/* auto camera = Camera {
		Transform { {2, 2, 6}, {-0.1, 0.3, 0} },
		Tunings { 45.0f, 800, 800 }
	}; */

	// Create the app and run it
	ECSApp app(phdev, extensions);
	// RTApp app(phdev, extensions);
	// engine::RTCapture app(phdev, {1000, 1000}, extensions, scene_path, camera);

	// Run the app
	app.run();
}

// Scene saving functions and loading functions
static constexpr char transform_format[]
= R"(position: %f %f %f
rotation: %f %f %f
scale: %f %f %f
)";

static constexpr char material_format[]
= R"(diffuse: %f %f %f
specular: %f %f %f
emission: %f %f %f
ambient: %f %f %f
shininess: %f
roughness: %f
refraction: %f
albedo_texture: %s
normal_texture: %s
)";

static constexpr char rasterizer_format[]
= R"(mode: %s
)";

static const std::array <std::string, 4> rasterizer_modes {
	"Phong",
	"Normal",
	"Albedo",
	"Wireframe"
};

static constexpr char light_format[]
= R"(color: %f %f %f
power: %f
type: %s
)";

static const std::array <std::string, 4> light_types {
	"Point",
	"Spot",
	"Directional",
	"Area"
};

// TODO: get rid of the transform portion
static constexpr char camera_format[]
= R"(position: %f %f %f
rotation: %f %f %f
scale: %f %f %f
fov: %f
scale: %f
aspect: %f
)";

static void save_transform(const Transform &transform, std::ofstream &fout)
{
	fout << "\n[TRANSFORM]\n";
	fout << common::sprintf(transform_format,
		transform.position.x, transform.position.y, transform.position.z,
		transform.rotation.x, transform.rotation.y, transform.rotation.z,
		transform.scale.x, transform.scale.y, transform.scale.z
	);
}

static void save_material(const Material &mat, std::ofstream &fout)
{
	fout << "\n[MATERIAL]\n";
	fout << common::sprintf(material_format,
		mat.diffuse.r, mat.diffuse.g, mat.diffuse.b,
		mat.specular.r, mat.specular.g, mat.specular.b,
		mat.emission.r, mat.emission.g, mat.emission.b,
		mat.ambient.r, mat.ambient.g, mat.ambient.b,
		mat.shininess,
		mat.roughness,
		mat.refraction,
		mat.albedo_texture.empty() ? "0" : mat.albedo_texture.c_str(),
		mat.normal_texture.empty() ? "0" : mat.normal_texture.c_str()
	);
	fout << "shading_type: " << shading_str(mat.type) << "\n";
}

static void save_rasterizer(const Rasterizer &rasterizer, std::ofstream &fout)
{
	fout << "\n[RASTERIZER]\n";
	fout << common::sprintf(rasterizer_format,
		rasterizer_modes[rasterizer.mode].c_str()
	);
}

static void save_raytracer(const Raytracer &raytracer, std::ofstream &fout)
{
	fout << "\n[RAYTRACER]\n";
}

static void save_light(const Light &light, std::ofstream &fout)
{
	fout << "\n[LIGHT]\n";
	fout << common::sprintf(light_format,
		light.color.r, light.color.g, light.color.b,
		light.power,
		light_types[light.type].c_str()
	);
}

static void save_camera(const Camera &cam, std::ofstream &fout)
{
	fout << "\n[CAMERA]\n";
	fout << common::sprintf(camera_format,
		cam.transform.position.x, cam.transform.position.y, cam.transform.position.z,
		cam.transform.rotation.x, cam.transform.rotation.y, cam.transform.rotation.z,
		cam.transform.scale.x, cam.transform.scale.y, cam.transform.scale.z,
		cam.tunings.fov,
		cam.tunings.scale,
		cam.tunings.aspect
	);
}

static void save_mesh(const Mesh &mesh, std::ofstream &fout)
{
	fout << "\n[MESH]\n";
	fout << "source: " << (mesh.source().empty() ? "0" : mesh.source().c_str()) << "\n";

	if (mesh.source().empty()) {
		// No source, raw data
		for (const auto &submesh : mesh.submeshes) {
			fout << "submesh {\n";
			for (const auto &vert : submesh.vertices) {
				fout << common::sprintf("\tv %.2f %.2f %.2f\n",
					vert.position.x, vert.position.y, vert.position.z
				);
			}

			fout << "\n";
			for (int i = 0; i < submesh.indices.size(); i += 3) {
				fout << common::sprintf("\tf %d %d %d\n",
					submesh.indices[i], submesh.indices[i + 1], submesh.indices[i + 2]
				);
			}

			fout << "}\n";
		}
	}
}

static void save_components(const Entity &e, std::ofstream &fout)
{
	// Case by case...
	save_transform(e.get <Transform> (), fout);

	if (e.exists <Material> ())
		save_material(e.get <Material> (), fout);

	if (e.exists <Mesh> ())
		save_mesh(e.get <Mesh> (), fout);

	if (e.exists <Rasterizer> ())
		save_rasterizer(e.get <Rasterizer> (), fout);

	if (e.exists <Raytracer> ())
		save_raytracer(e.get <Raytracer> (), fout);

	if (e.exists <Light> ())
		save_light(e.get <Light> (), fout);

	if (e.exists <Camera> ())
		save_camera(e.get <Camera> (), fout);
}

void ::Scene::save(const std::string &path)
{
	std::ofstream fout(path);
	if (!fout.is_open()) {
		KOBRA_LOG_FUNC(error) << "Failed to open file: " << path << std::endl;
		return;
	}

	fout << "[PROPERTIES]" << std::endl;
	fout << "environment_map: " << p_environment_map << std::endl;

	for (int i = 0; i < ecs.size(); i++) {
		const auto &entity = ecs.get_entity(i);
		fout << "\n[ENTITY]" << std::endl;
		fout << "name: " << entity.name << std::endl;

		save_components(entity, fout);
	}
}

// Scene loading functions and helpers
inline std::string get_header(std::ifstream &fin)
{
	std::string line;

	do {
		std::getline(fin, line);
	} while (line.empty() && !fin.eof());

	return line;
}

// Field reading utility
template <class ... Args>
struct field_reader {
	static void read(std::ifstream &fin, const char *format, Args ... args) {
		std::string line;
		std::getline(fin, line);
		int read = sscanf(line.c_str(), format, std::forward <Args> (args)...);

		if (read != sizeof...(Args))
			KOBRA_LOG_FUNC(warn) << "Failed to read fields: " << line << std::endl;
	}
};

// Specializations
template <class ... Args>
struct field_reader <std::string, Args...> {
	static void read(std::ifstream &fin, const char *format, std::string &str, Args ... args) {
		static char buf[1024];
		field_reader <char *, Args...> ::read(fin, format, buf, args...);
		str = buf;
	}
};

template <class ... Args>
struct field_reader <glm::vec3, Args...> {
	static void read(std::ifstream &fin, const char *format, glm::vec3 &vec, Args &... args) {
		field_reader <float *, float *, float *, Args...>
			::read(fin, format, &vec.x, &vec.y, &vec.z, args...);
	}
};

// TODO: if none other than the following simple types are parsed, then remove
// the over templatization
inline void read_string(std::ifstream &fin, const char *fmt, std::string &str)
{
	field_reader <std::string> ::read(fin, fmt, str);
}

inline void read_float(std::ifstream &fin, const char *fmt, float &f)
{
	field_reader <float *> ::read(fin, fmt, &f);
}

inline void read_vec3(std::ifstream &fin, const char *fmt, glm::vec3 &vec)
{
	field_reader <glm::vec3> ::read(fin, fmt, vec);
}

template <class ... Args>
void read_fmt(std::ifstream &fin, const char *fmt, Args ... args)
{
	// Count number of lines to read
	int count = 0;
	for (const char *c = fmt; *c != '\0'; c++) {
		if (*c == '\n')
			count++;
	}

	// Read lines
	std::string line;
	std::string whole;
	for (int i = 0; i < count; i++) {
		std::getline(fin, line);
		whole += line + "\n";
	}

	// Parse
	int read = sscanf(whole.c_str(), fmt, std::forward <Args> (args)...);
	if (read != sizeof...(Args))
		KOBRA_LOG_FUNC(warn) << "Failed to read field after #" << read << " fields\n";
}

// Component basis
void load_transform(Entity &e, std::ifstream &fin)
{
	// TODO: eventually just use the format
	e.add <Transform> ();

	Transform &transform = e.get <Transform> ();

	read_fmt(fin, transform_format,
		&transform.position.x, &transform.position.y, &transform.position.z,
		&transform.rotation.x, &transform.rotation.y, &transform.rotation.z,
		&transform.scale.x, &transform.scale.y, &transform.scale.z
	);
}

void load_material(Entity &e, std::ifstream &fin)
{
	static char buf_albedo[1024];
	static char buf_normal[1024];

	e.add <Material> ();

	Material &material = e.get <Material> ();

	read_fmt(fin, material_format,
		&material.diffuse.x, &material.diffuse.y, &material.diffuse.z,
		&material.specular.x, &material.specular.y, &material.specular.z,
		&material.emission.x, &material.emission.y, &material.emission.z,
		&material.ambient.x, &material.ambient.y, &material.ambient.z,
		&material.shininess,
		&material.roughness,
		&material.refraction,
		buf_albedo, buf_normal
	);

	std::string line;
	std::getline(fin, line);

	std::string field = line.substr(0, 14);
	std::string value = line.substr(14);

	if (field != "shading_type: ") {
		KOBRA_LOG_FUNC(warn) << "Failed to read shading type: field = \""
			<< field  << "\": value = \"" << value << "\"" << std::endl;
		return;
	}

	if (std::string(buf_albedo) != "0")
		material.albedo_texture = buf_albedo;

	if (std::string(buf_normal) != "0")
		material.normal_texture = buf_normal;

	material.type = *shading_from_str(value);
}

void load_mesh(Entity &e, std::ifstream &fin)
{
	static char buf_source[1024];


	std::string line;
	std::getline(fin, line);

	sscanf(line.c_str(), "source: %s", buf_source);

	if (std::string(buf_source) != "0") {
		auto mptr = Mesh::load(buf_source);

		if (!mptr.has_value()) {
			KOBRA_LOG_FUNC(warn) << "Failed to load mesh: " << buf_source << std::endl;
			return;
		}

		e.add <Mesh> (*mptr);
	} else {
		// Raw mesh
		std::vector <Submesh> submeshes;
		while (true) {
			// Iterate over submeshes
			std::getline(fin, line);
			if (line.empty())
				break;

			if (line != "submesh {") {
				KOBRA_LOG_FUNC(warn) << "Expected submesh, got: " << line << std::endl;
				return;
			}

			// Submesh in creation
			std::vector <Vertex> vertices;
			std::vector <uint32_t> indices;

			// Read submesh vertices
			while (true) {
				std::getline(fin, line);
				if (line.empty())
					break;

				glm::vec3 vertex;
				sscanf(line.c_str(), "\tv %f %f %f", &vertex.x, &vertex.y, &vertex.z);
				vertices.push_back(vertex);
			}

			// Read faces
			while (true) {
				std::getline(fin, line);

				// TODO: check amount read
				int v0, v1, v2;
				int read = sscanf(line.c_str(), "\tf %d %d %d", &v0, &v1, &v2);

				if (read == 0)
					break;

				indices.push_back(v0);
				indices.push_back(v1);
				indices.push_back(v2);
			}

			// Assert closure
			if (line != "}") {
				KOBRA_LOG_FUNC(warn) << "Expected closure, got: " << line << std::endl;
				return;
			}

			// Append submesh
			submeshes.push_back({vertices, indices});
		}

		// Create mesh
		e.add <Mesh> (submeshes);
	}
}

void load_rasterizer(Entity &e, std::ifstream &fin, const Device &dev)
{
	static char buf_mode[1024];

	// Mkae sure we have a mesh and material
	if (!e.exists <Mesh> ()) {
		KOBRA_LOG_FUNC(warn) << "No mesh for rasterizer" << std::endl;
		return;
	}

	if (!e.exists <Material> ()) {
		KOBRA_LOG_FUNC(warn) << "No material for rasterizer" << std::endl;
		return;
	}

	e.add <Rasterizer> (dev, e.get <Mesh> (), &e.get <Material> ());

	// Read mode
	std::string line;
	std::getline(fin, line);

	sscanf(line.c_str(), "mode: %s", buf_mode);

	// Get index
	int index = 0;
	while (rasterizer_modes[index] != buf_mode)
		index++;

	if (index >= rasterizer_modes.size()) {
		KOBRA_LOG_FUNC(warn) << "Unknown rasterizer mode: " << buf_mode << std::endl;
		return;
	}

	// Set mode
	e.get <Rasterizer> ().mode = RasterMode(index);
}

void load_raytracer(Entity &e, std::ifstream &fin)
{
	// Make sure we have a mesh and material
	if (!e.exists <Mesh> ()) {
		KOBRA_LOG_FUNC(warn) << "No mesh for raytracer" << std::endl;
		return;
	}

	if (!e.exists <Material> ()) {
		KOBRA_LOG_FUNC(warn) << "No material for raytracer" << std::endl;
		return;
	}

	e.add <Raytracer> (&e.get <Mesh> (), &e.get <Material> ());
}

void load_camera(Entity &e, std::ifstream &fin)
{
	e.add <Camera> ();

	Camera &camera = e.get <Camera> ();
	read_fmt(fin, camera_format,
		&camera.transform.position.x, &camera.transform.position.y, &camera.transform.position.z,
		&camera.transform.rotation.x, &camera.transform.rotation.y, &camera.transform.rotation.z,
		&camera.transform.scale.x, &camera.transform.scale.y, &camera.transform.scale.z,
		&camera.tunings.fov, &camera.tunings.scale, &camera.tunings.aspect
	);
}

void load_light(Entity &e, std::ifstream &fin)
{
	static char buf_type[1024];

	e.add <Light> ();

	Light &light = e.get <Light> ();
	read_fmt(fin, light_format,
		&light.color.x, &light.color.y, &light.color.z,
		&light.power, buf_type
	);

	// Get light type
	int index = 0;
	while (light_types[index] != buf_type)
		index++;

	if (index >= light_types.size()) {
		KOBRA_LOG_FUNC(warn) << "Unknown light type: " << buf_type << std::endl;
		return;
	}

	light.type = Light::Type(index);
}

std::string load_components(Entity &e, std::ifstream &fin, const Device &dev)
{
	std::string header;

	// Go in order of the components
	while (true) {
		header = get_header(fin);
		if (header == "[TRANSFORM]") {
			load_transform(e, fin);
			continue;
		}

		if (header == "[MATERIAL]") {
			load_material(e, fin);
			continue;
		}

		if (header == "[MESH]") {
			load_mesh(e, fin);
			continue;
		}

		if (header == "[RASTERIZER]") {
			load_rasterizer(e, fin, dev);
			continue;
		}

		if (header == "[RAYTRACER]") {
			load_raytracer(e, fin);
			continue;
		}

		if (header == "[CAMERA]") {
			load_camera(e, fin);
			continue;
		}

		if (header == "[LIGHT]") {
			load_light(e, fin);
			continue;
		}

		break;
	}

	return header;
}

void ::Scene::load(const Device &dev, const std::string &path)
{
	std::ifstream fin(path);
	if (!fin.is_open()) {
		KOBRA_LOG_FUNC(error) << "Failed to open file: " << path << std::endl;
		return;
	}

	// Load properties
	if (get_header(fin) != "[PROPERTIES]") {
		KOBRA_LOG_FUNC(error) << "Failed to load properties" << std::endl;
		return;
	}

	field_reader <std::string> ::read(fin, "environment_map: %s", p_environment_map);

	// Load entities
	std::string header = get_header(fin);
	while (fin.good()) {
		if (header != "[ENTITY]") {
			KOBRA_LOG_FUNC(error) << "Invalid header: " << header << std::endl;
			return;
		}

		std::string name;
		field_reader <std::string> ::read(fin, "name: %s", name);
		Entity &e = ecs.make_entity(name);

		header = load_components(e, fin, dev);
	}
}

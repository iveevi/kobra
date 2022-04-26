#ifndef GLOBAL_H_
#define GLOBAL_H_

// Standard headers
#include <cstring>
#include <iostream>
#include <thread>
#include <vulkan/vulkan_core.h>

// Engine macros
// #define KOBRA_VALIDATION_LAYERS
// #define KOBRA_ERROR_ONLY
// #define KOBRA_THROW_ERROR

// Engine headers
#include "include/app.hpp"
#include "include/backend.hpp"
#include "include/capture.hpp"
#include "include/engine/gizmo.hpp"
#include "include/engine/rt_capture.hpp"
#include "include/gui/gui.hpp"
#include "include/gui/layer.hpp"
#include "include/gui/rect.hpp"
#include "include/gui/sprite.hpp"
#include "include/io/event.hpp"
#include "include/model.hpp"
#include "include/profiler.hpp"
#include "include/raster/layer.hpp"
#include "include/raytracing/layer.hpp"
#include "include/raytracing/mesh.hpp"
#include "include/raytracing/sphere.hpp"
#include "include/scene.hpp"
#include "include/types.hpp"
#include "profiler.hpp"

using namespace kobra;

// Scene path
extern std::string scene_path;

// TODO: focus on adding objects to the scene (spheres, boxes, models)
// TODO: RT sampling sphere lights
// TODO: input window for RT Capture (etner # of samples, ect)

// Main class
class RTApp :  public BaseApp {
	// Application camera
	Camera		camera;

	// RT or Raster
	bool			raster = true;
	bool			modified = false;
	bool			show_mouse = false;

	// Layers
	rt::Layer		rt_layer;
	raster::Layer		raster_layer;
	gui::Layer		gui_layer;

	// Capturer
	engine::RTCapture	*capturer = nullptr;
	std::thread		*capturer_thread = nullptr;

	// Current scene
	Scene			scene;

	// Raster state
	int			highlight = -1;
	bool			edit_mode = false;

	// Editting state
	struct {
		// 1 = translate, 2 = rotate, 3 = scale
		int			gizmo_mode = 1;
		raster::Layer::ptr	selected = nullptr;

		int			rot_axis = -1;
		raster::Mesh *gizmo_x = nullptr;
		raster::Mesh *gizmo_y = nullptr;
		raster::Mesh *gizmo_z = nullptr;

		raster::Mesh *p0 = nullptr;
		raster::Mesh *p1 = nullptr;
	} edit;

	// Raytracing state
	rt::Batch		batch;
	rt::BatchIndex		batch_index;

	// GUI state
	// TODO: gui window abstraction (uses screen info returns as well)
	struct {
		// Statistics
		struct {
			gui::Text	*frame_rate;
			gui::Text	*counts;
			gui::Rect	*bg;
		} stats;

		// Help
		gui::Text	*text_help;
		gui::Rect	*help_bg;

		// Editing
		gui::Text	*text_position;
		gui::Text	*text_rotation;
		gui::Text	*text_scale;
		gui::Rect	*edit_bg;
	} gui;

	// Update the scene with the raster_layer's current state
	void update_scene() {
		for (auto &obj : raster_layer) {
			std::string name = obj->name();
			scene[name]->transform() = obj->transform();
		}
	}

	// Add a mesh object
	void add_mesh(const std::string &path) {
		// Load model, then all its meshes
		auto model = Model::load(path);

		for (int i = 0; i < model.mesh_count(); i++) {
			auto mesh = model[i];

			// Add to raster and scene
			raster_layer.add(new raster::Mesh(context, mesh));
			scene.add(ObjectPtr(new Mesh(mesh)));
		}
	}

	// Duplicate an object
	std::string duplicate_object(const std::string &name) {
		// TODO: better naming scheme
		static int count = 1;

		// Do for both raster and scene
		auto raster_obj = scene[name];

		// For now, just these types are allowed to be duplicated
		std::string type = raster_obj->type();

		auto new_name = name + "_" + std::to_string(count++);

		KOBRA_LOG_FILE(notify) << "Duplicating " << name << " as " << new_name << " (type = " << type << ")\n";
		if (type == Mesh::object_type) {
			auto ptr_mesh = std::dynamic_pointer_cast <Mesh> (raster_obj);

			auto scene_mesh = new Mesh(*ptr_mesh);
			auto raster_mesh = new raster::Mesh(context, *ptr_mesh);

			scene_mesh->set_name(new_name);
			raster_mesh->set_name(new_name);

			scene.add(ObjectPtr(scene_mesh));
			raster_layer.add(raster_mesh);
		} else if (type == Sphere::object_type) {
			auto ptr_sphere = std::dynamic_pointer_cast <Sphere> (raster_obj);
			auto scene_sphere = new Sphere(*ptr_sphere);

			auto sphere_mesh = Mesh::make_sphere(ptr_sphere->center(), ptr_sphere->radius());
			auto raster_sphere = new raster::Mesh(context, sphere_mesh);

			scene_sphere->set_name(new_name);
			raster_sphere->set_name(new_name);

			scene.add(ObjectPtr(scene_sphere));
			raster_layer.add(raster_sphere);
		} else {
			KOBRA_LOG_FILE(warn) << "Cannot duplicate object of type " << type << "\n";
		}

		return new_name;
	}

	// Initialize GUI elements
	void initialize_gui() {
		// TODO: rounded corners
		// TODO: color type

		// TODO: method to add gui elements
		// TODO: add_scene for gui layer

		// Fonts
		gui_layer.load_font("default", "resources/fonts/noto_sans.ttf");

		// Statistics
		gui.stats.frame_rate = gui_layer.text_render("default")->text(
			"",
			window.coordinates(10, 10),
			{1, 1, 1, 1}
		);

		gui_layer.add(gui.stats.frame_rate);
	}

	// Update GUI elements
	void update_gui() {
		static char buffer[1024];

		// Statistics
		// TODO: statistics should be made a standalone layer
		std::sprintf(buffer, "time: %.2f ms, fps: %.2f",
			1000 * frame_time,
			1.0f/frame_time
		);

		gui.stats.frame_rate->str = buffer;
	}

	// Input handling for edit mode
	// TODO: text to indicate mode and name of selected object
	int transform_row = 0;
	int transform_col = 0;

	void handle_edit_mode() {
		// Assume edit mode
		auto eptr = raster_layer[highlight];
		auto &transform = eptr->transform();
		float speed = 0.01f;

		// TODO: determine a reference to a float
		float *fptr = nullptr;
		if (transform_row == 0) {
			fptr = &transform.position[transform_col];
		} else if (transform_row == 1) {
			fptr = &transform.rotation[transform_col];
			speed = 0.1f;
		} else if (transform_row == 2) {
			fptr = &transform.scale[transform_col];
		}

		if (input.is_key_down('=')) {
			*fptr += speed;
			modified = true;
		} else if (input.is_key_down('-')) {
			*fptr -= speed;
			modified = true;
		}

		// Save with corresponding scene object
		std::string name = eptr->name();
		scene[name]->transform() = transform;
	}

	// Save element currently in raster_layer
	void save(const std::string &path) {
		for (auto &e : raster_layer) {
			scene[e->name()]->transform() = e->transform();
			// TODO: later will need to copy material, etc
		}

		scene.save(path);
	}

	// Keyboard handler
	static void keyboard_handler(void *, const io::KeyboardEvent &);

	// Mouve camera
	static void mouse_movement(void *, const io::MouseEvent &);

	// Gizmo things
	engine::Gizmo::Handle		gizmo_handle;
	engine::Gizmo			gizmo_set;

	// Gizmo pipeline
	VkRenderPass			gizmo_render_pass;
	Vulkan::Pipeline		gizmo_pipeline;

	void init_gizmo() {
		// Create render pass
		gizmo_render_pass = context.make_render_pass(swapchain,
			VK_ATTACHMENT_LOAD_OP_LOAD,
			VK_ATTACHMENT_STORE_OP_STORE
		);

		// Load shaders
		auto shaders = context.make_shaders({
			"shaders/bin/raster/vertex.spv",
			"shaders/bin/raster/plain_color_frag.spv"
		});

		// Push constants
		VkPushConstantRange pcr {
			.stageFlags = VK_SHADER_STAGE_VERTEX_BIT,
			.offset = 0,
			.size = sizeof(typename raster::Mesh::MVP)
		};

		// Creation info
		Vulkan::PipelineInfo info {
			.swapchain = swapchain,
			.render_pass = render_pass,

			.vert = shaders[0],
			.frag = shaders[1],

			.dsls = {},

			.vertex_binding = Vertex::vertex_binding(),
			.vertex_attributes = Vertex::vertex_attributes(),

			.push_consts = 1,
			.push_consts_range = &pcr,

			.depth_test = false,

			.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,

			// TODO: make a swapchain method to get viewport
			.viewport {
				.width = (int) swapchain.extent.width,
				.height = (int) swapchain.extent.height,
				.x = 0,
				.y = 0
			}
		};

		gizmo_pipeline = context.make_pipeline(info);
	}
public:
	// TODO: just remove the option of no depth buffer (always use depth buffer)
	RTApp(Vulkan *vk) : BaseApp({
		vk,
		1000, 1000, 2,
		"RT App"
	}) {
		// Construct camera
		camera = Camera {
			Transform { {0, 6, 18}, {-0.2, 0, 0} },
			Tunings { 45.0f, 800, 800 }
		};

		// Load scene
		// create_scene();
		Profiler::one().frame("Loading scene");
		scene = Scene(context, window.command_pool, scene_path);
		Profiler::one().end();

		for (auto &obj : scene)
			std::cout << "Scene object: " << obj->name() << std::endl;

		///////////////////////
		// Initialize layers //
		///////////////////////

		// Ray tracing layer
		rt_layer = rt::Layer(window);
		rt_layer.add_scene(scene);

		// Initialize batch and batch index
		batch = rt::Batch(1000, 1000, 100, 100, 16);
		batch_index = batch.make_batch_index(0, 0);
		batch_index.light_samples = 1;

		// TODO: m8 gotta really fix auto channels
		rt_layer.set_environment_map(
			load_image_texture("resources/skies/background_1.jpg", 4)
		);

		// Rasterization layer
		raster_layer = raster::Layer(window, VK_ATTACHMENT_LOAD_OP_CLEAR);
		raster_layer.set_mode(raster::Layer::Mode::BLINN_PHONG);
		raster_layer.add_scene(scene);

		// Rotation gizmo
		auto ring_x = Mesh::make_ring({0, 0, 0}, 1, 0.05, 0.05);
		auto ring_y = Mesh::make_ring({0, 0, 0}, 1, 0.05, 0.05);
		auto ring_z = Mesh::make_ring({0, 0, 0}, 1, 0.05, 0.05);

		ring_x.transform().rotation = {90, 0, 0};
		ring_z.transform().rotation = {0, 0, 90};

		ring_x.material().albedo = {1, 0, 0};
		ring_y.material().albedo = {0, 1, 0};
		ring_z.material().albedo = {0, 0, 1};

		edit.gizmo_x = new raster::Mesh(context, ring_x);
		edit.gizmo_y = new raster::Mesh(context, ring_y);
		edit.gizmo_z = new raster::Mesh(context, ring_z);

		auto s0 = Mesh::make_sphere({0, 0, 0}, 0.05);

		edit.p0 = new raster::Mesh(context, s0);
		edit.p0->material().albedo = {0, 0, 0};
		
		edit.p1 = new raster::Mesh(context, s0);
		edit.p1->material().albedo = {1, 1, 1};

		init_gizmo();

		// GUI layer
		// TODO: be able to load gui elements from scene
		gui_layer = gui::Layer(window, VK_ATTACHMENT_LOAD_OP_LOAD);
		initialize_gui();

		// Gizmo layer
		// TODO: all layer type constructor should take a mandatory load
		// operation
		gizmo_set = engine::Gizmo(window, VK_ATTACHMENT_LOAD_OP_LOAD);
		gizmo_handle = gizmo_set.transform_gizmo();

		// Add event listeners
		window.keyboard_events->subscribe(keyboard_handler, this);
		window.mouse_events->subscribe(mouse_movement, this);

		// Show results of profiling
		auto frame = Profiler::one().pop();
		// std::cout << Profiler::pretty(frame);
	}

	// Destructor
	~RTApp() {
		if (capturer) {
			capturer->terminate_now();
			capturer_thread->join();
			delete capturer_thread;
			delete capturer;
		}
	}

	// Override record method
	// TODO: preview raytraced scene with a very low resolution
	void record(const VkCommandBuffer &cmd, const VkFramebuffer &framebuffer) override {
		static float time = 0.0f;

		// Start recording command buffer
		Vulkan::begin(cmd);

		// WASDEQ movement
		// TODO: method
		float speed = 20.0f * frame_time;

		glm::vec3 forward = camera.transform.forward();
		glm::vec3 right = camera.transform.right();
		glm::vec3 up = camera.transform.up();

		if (input.is_key_down(GLFW_KEY_W))
			camera.transform.move(forward * speed);
		else if (input.is_key_down(GLFW_KEY_S))
			camera.transform.move(-forward * speed);

		if (input.is_key_down(GLFW_KEY_A))
			camera.transform.move(-right * speed);
		else if (input.is_key_down(GLFW_KEY_D))
			camera.transform.move(right * speed);

		if (input.is_key_down(GLFW_KEY_E))
			camera.transform.move(up * speed);
		else if (input.is_key_down(GLFW_KEY_Q))
			camera.transform.move(-up * speed);

		// Input
		if (edit_mode)
			handle_edit_mode();

		// Copy camera contents to each relevant layer
		rt_layer.set_active_camera(camera);
		raster_layer.set_active_camera(camera);
		gizmo_set.set_camera(camera);

		/* Highlight appropriate object
		raster_layer.clear_highlight();
		if (edit_mode)
			raster_layer.set_highlight(highlight, true); */

		// Render appropriate layer
		if (raster) {
			raster_layer.render(cmd, framebuffer);
		} else {
			if (modified) {
				// TODO: fix clear method
				// rt_layer.clear();
				// rt_layer.add_scene(scene);
				KOBRA_LOG_FILE(notify) << "Reconstructing RT layer\n";
				scene.save(scene_path);
				scene = Scene(context, window.command_pool, scene_path);

				rt_layer = rt::Layer(window);
				rt_layer.add_scene(scene);
				rt_layer.set_active_camera(camera);
				modified = false;
			}

			rt_layer.render(cmd, framebuffer, batch, batch_index);

			batch.increment(batch_index);
			if (batch.completed())
				batch.reset();
		}

		// Render GUI
		update_gui();
		gui_layer.render(cmd, framebuffer);

		// Render gizmo
		if (gizmo_handle->get_object() != nullptr && edit.gizmo_mode == 1)
			gizmo_set.render(cmd, framebuffer);
		
		// Start gizmo render pass
		VkRect2D render_area {
			.offset = {0, 0},
			.extent = swapchain.extent
		};

		std::vector <VkClearValue> clear_colors {
			{.color = {0.0f, 0.0f, 0.0f, 1.0f}},
			{.depthStencil = {1.0f, 0}}
		};
		
		Vulkan::begin_render_pass(cmd,
			gizmo_render_pass,
			framebuffer,
			render_area,
			clear_colors
		);

		// Bind pipeline
		vkCmdBindPipeline(cmd,
			VK_PIPELINE_BIND_POINT_GRAPHICS,
			gizmo_pipeline.pipeline
		);

		// Initialize render packet
		raster::RenderPacket packet {
			.cmd = cmd,

			.pipeline_layout = gizmo_pipeline.layout,

			// TODO: warn on null camera
			.view = camera.view(),
			.proj = camera.perspective()
		};

		// Render gizmos
		if (edit.gizmo_mode == 2 && edit.selected != nullptr) {
			auto name = edit.selected->name();
			auto ptr = raster_layer[name];
			auto pos = ptr->transform().position;

			edit.gizmo_x->transform().position = pos;
			edit.gizmo_y->transform().position = pos;
			edit.gizmo_z->transform().position = pos;

			edit.gizmo_x->draw(packet);
			edit.gizmo_y->draw(packet);
			edit.gizmo_z->draw(packet);
		}

		// End render pass
		Vulkan::end_render_pass(cmd);

		// End recording command buffer
		Vulkan::end(cmd);

		// Update time
		time += frame_time;
	}

	// Termination method
	void terminate() override {
		// Check input
		if (window.input->is_key_down(GLFW_KEY_ESCAPE))
			glfwSetWindowShouldClose(surface.window, true);
	}
};

#endif

// Engine headers
#include "kobra.hpp"
#include "global.hpp"

// Keyboard callback
// TODO: in class
bool mouse_tracking = true;

void key_callback(GLFWwindow *window, int key, int scancode, int action, int mods)
{
        if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS)
                glfwSetWindowShouldClose(window, GL_TRUE);

	// Tab to toggle cursor visibility
	static bool cursor_visible = false;
	if (key == GLFW_KEY_TAB && action == GLFW_PRESS) {
		cursor_visible = !cursor_visible;
		mouse_tracking = !mouse_tracking;
		glfwSetInputMode(
			window,
			GLFW_CURSOR,
			cursor_visible ?
				GLFW_CURSOR_NORMAL
				: GLFW_CURSOR_DISABLED
		);
	}
}

// Mouse movement callback
// TODO: in class
void mouse_callback(GLFWwindow *window, double xpos, double ypos)
{
	static bool first_mouse = true;
	static float last_x = WIDTH / 2.0f;
	static float last_y = HEIGHT / 2.0f;
	static const float sensitivity = 0.001f;

	if (!mouse_tracking)
		return;

	if (first_mouse) {
		first_mouse = false;
		last_x = xpos;
		last_y = ypos;
		return;
	}

	// Store pitch and yaw
	static float pitch = 0.0f;
	static float yaw = 0.0f;

	float xoffset = xpos - last_x;
	float yoffset = ypos - last_y;

	xoffset *= sensitivity;
	yoffset *= sensitivity;

	yaw -= xoffset;
	pitch -= yoffset;

	if (pitch > 89.0f)
		pitch = 89.0f;
	else if (pitch < -89.0f)
		pitch = -89.0f;

	world.camera.transform.rotation.x = pitch;
	world.camera.transform.rotation.y = yaw;

	last_x = xpos;
	last_y = ypos;
}

// List of objet materials
Material materials[] {
	{.albedo = glm::vec3 {0.1f, 0.5f, 0.2f}},
	{
		.albedo = glm::vec3 {1.0f},
		.reflectance = 1.0f
	},
	{
		.albedo = glm::vec3 {1.0f, 1.0f, 1.0f},
		.specular = 1.0,
		.reflectance = 0.01,
		.refractance = 0.0
	},
	{.albedo = glm::vec3 {0.5f, 0.1f, 0.6f}},
	{.albedo = glm::vec3 {0.6f, 0.5f, 0.3f}},
	{.albedo = glm::vec3 {1.0f, 0.5f, 1.0f}},
	{
		.albedo = glm::vec3 {1.0f, 1.0f, 1.0f},
		.shading = SHADING_TYPE_LIGHT
	}
};

using kobra::rt::World;
using kobra::raytracing::Sphere;

World world {
	// Camera
	Camera {
		Transform {
			glm::vec3(-1.46986, 8.05425, 10.311),
			glm::vec3(-0.158, 0.139, 0)
		},

		Tunings {
			45.0f, 800, 600
		}
	},

	// Primitives
	// TODO: later read from file
	std::vector <World::PrimitivePtr> {
		World::PrimitivePtr(new Sphere(0.25f, glm::vec3 {0, 0, 0}, materials[6])),

		World::PrimitivePtr(new Sphere(1.0f, glm::vec3 {0, 3, 4},
			{
				.albedo = glm::vec3 {1.0f},
				.refractance = 1.5f
			}
		)),

		World::PrimitivePtr(new Sphere(1.0f, glm::vec3 {-3.0f, 6, 1.0f},
			{
				.albedo = glm::vec3 {1.0f},
				.reflectance = 1.0f
			}
		))
	},

	// Lights
	std::vector <World::LightPtr> {
		// TODO: objects with emmision
		World::LightPtr(new PointLight(glm::vec3 {0, 10, -10}, 0.0f))
	}
};

using RTMesh = kobra::raytracing::Mesh;

int main()
{
	kobra::Model model1("/home/venki/downloads/quixel/column/ud4pbimfa_LOD0.fbx");
	kobra::Model model2("/home/venki/downloads/quixel/statue/vd4ibgcva_LOD0.fbx");
	kobra::Model model3("/home/venki/downloads/quixel/rock/vivveardw_LOD0.fbx");

	Material mat1 = materials[1];
	Material mat2 = materials[1];
	Material mat3 {
		.albedo = glm::vec3 {0.8, 0.5, 0.5},
		.reflectance = 0.0f,
		.refractance = 0.0f
	};

	auto m1 = new RTMesh(model1[0]);
	auto m2 = new RTMesh(model2[0]);
	auto m3 = new RTMesh(model3[0]);

	m1->transform().scale = glm::vec3(0.08f);
	m2->transform().scale = glm::vec3(0.03f);
	m3->transform().scale = glm::vec3(0.03f);

	m1->set_material(mat1);
	m2->set_material(mat2);
	m3->set_material(mat3);

	m1->transform().position = glm::vec3(0.0f, 1.5f, 0.0f);
	m1->transform().rotation = glm::vec3(0.0f, 0.0f, -10);

	m2->transform().position = glm::vec3(-3.0f, 3.0f, -3.0f);
	m2->transform().rotation = glm::vec3(0.0f, 0.0f, 10);

	auto m4 = new RTMesh(*m3);
	auto m5 = new RTMesh(*m3);

	m4->transform().scale = glm::vec3(0.10f);
	m5->transform().scale = glm::vec3(0.05f);

	m4->transform().rotation = glm::vec3(13, 45, 0.0);
	m5->transform().rotation = glm::vec3(0, 0, 0);

	m3->transform().position = glm::vec3(-6.0f, 1.0f, -3.0f);
	m4->transform().position = glm::vec3(-3.0f, -5.0f, -3.0f);
	m5->transform().position = glm::vec3(0.0f, 1.0f, -3.0f);

	world.objects.push_back(std::shared_ptr <RTMesh> (m1));
	world.objects.push_back(std::shared_ptr <RTMesh> (m2));
	world.objects.push_back(std::shared_ptr <RTMesh> (m3));
	world.objects.push_back(std::shared_ptr <RTMesh> (m4));
	world.objects.push_back(std::shared_ptr <RTMesh> (m5));

	Logger::ok() << "[main] Loaded model with "
		<< model1.mesh_count() << " meshe(s), "
		<< model1[0].vertex_count() << " vertices, "
		<< model1[0].triangle_count() << " triangles" << std::endl;

	// Plane mesh
	float width = 10.0f;
	float length = 10.0f;

	RTMesh plane_mesh {
		VertexList {
			glm::vec3(-width/2, 0.27f, -length/2),
			glm::vec3(width/2, 0.27f, -length/2),
			glm::vec3(width/2, 0.27f, length/2),
			glm::vec3(-width/2, 0.27f, length/2)
		},
		IndexList {
			0, 1, 2,
			2, 3, 0
		},
		Material {
			.albedo = glm::vec3(1.0f, 1.0f, 0.7f),
			// .reflectance = 0.0f,
			.refractance = 0.0f,
			.extinction = 0.28176f
		}
	};

	/* Add plane to world
	world.objects.push_back(std::shared_ptr <RTMesh> (
		new RTMesh(plane_mesh)
	)); */

	Logger::notify("Transforms (model matrices) of all objects:");
	for (auto &object : world.objects) {
		glm::mat4 model = object->transform().matrix();
		glm::vec4 pos = model[3];
		Logger::notify() << "\t" << pos.x << ", " << pos.y << ", " << pos.z << std::endl;
	}

	// Save world into scene
	kobra::Scene scene("default_world", world);
	scene.save("resources/default_world.hg");

	// Initialize Vulkan
	Vulkan *vulkan = new Vulkan();
	vulkan->init_imgui();

	// Create sample scene
	RTApp app(vulkan);

	app.update_descriptor_set();
	app.update_command_buffers();
	app.run();

	delete vulkan;
}

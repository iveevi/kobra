// Standard headers
#include <cstring>
#include <iostream>
#include <thread>
#include <vulkan/vulkan_core.h>
// #include <vulkan/vulkan_core.h>

// Local headers
#include "global.hpp"
#include "mercury.hpp"

using namespace mercury;

// List of objet materials
Material materials[] {
	{.albedo = glm::vec3 {0.1f, 0.5f, 0.2f}},
	{
		.albedo = glm::vec3 {1.0f, 1.0f, 1.0f},
		.specular = 32.0f,
		.reflectance = 0.9f,
		.refractance = 0.0f	// TODO: refractance should be a complex number
	},
	{
		.albedo = glm::vec3 {1.0f, 1.0f, 1.0f},
		.specular = 1.0,
		.reflectance = 0.01,
		.refractance = 1.5
	},
	{.albedo = glm::vec3 {0.5f, 0.1f, 0.6f}},
	{.albedo = glm::vec3 {0.6f, 0.5f, 0.3f}},
	{.albedo = glm::vec3 {1.0f, 0.5f, 1.0f}},
	{
		.albedo = glm::vec3 {1.0f, 1.0f, 1.0f},
		.shading = SHADING_TYPE_LIGHT
	}
};

// List of object transforms
Transform transforms[] {
	glm::vec3 {-1.0f, 0.0f, 4.0f},
	glm::vec3 {0.5f, 5.0f, 3.5f},
	glm::vec3 {6.0f, -2.0f, 5.0f},
	glm::vec3 {6.0f, 3.0f, 11.5f},
	glm::vec3 {6.0f, 3.0f, -2.0f},
	glm::vec3 {0.0f, 0.0f, 0.0f},
	glm::vec3 {0.0f, 0.0f, -1.0f}
};

World world {
	// Camera
	Camera {
		Transform {
			glm::vec3(0.0f, 0.0f, -4.0f)
		},
	 
		Tunings {
			45.0f, 800, 600
		}
	},

	// Primitives
	// TODO: later read from file
	std::vector <World::PrimitivePtr> {
		World::PrimitivePtr(new Sphere(0.25f, transforms[0], materials[6])),

		// World::PrimitivePtr(new Sphere(1.0f, transforms[0], materials[0])),
		// World::PrimitivePtr(new Sphere(3.0f, transforms[1], materials[1])),
		/* World::PrimitivePtr(new Sphere(6.0f, transforms[2], materials[2])),
		World::PrimitivePtr(new Sphere(2.0f, transforms[3], materials[3])),
		World::PrimitivePtr(new Sphere(2.0f, transforms[4], materials[4])),

		// Cube mesh
		World::PrimitivePtr(new mercury::Mesh <mercury::VERTEX_TYPE_POSITION> (
			{
				glm::vec3(0.0f, 6.0f, -1.5f),
				glm::vec3(1.0f, 6.0f, -1.5f),
				glm::vec3(1.0f, 7.0f, -1.5f),
				glm::vec3(0.0f, 7.0f, -1.5),
				glm::vec3(0.0f, 6.0f, 0.5f),
				glm::vec3(1.0f, 6.0f, 0.5f),
				glm::vec3(1.0f, 7.0f, 0.5f),
				glm::vec3(0.0f, 7.0f, 0.5f)
			},
			{
				0, 1, 2,	0, 2, 3,
				4, 5, 6,	4, 6, 7,
				0, 4, 7,	0, 7, 3,
				1, 5, 6,	1, 6, 2,
				0, 1, 4,	1, 4, 5,
				2, 6, 7,	2, 7, 3
			},
			transforms[6],
			materials[1]
		)), */
	},

	// Lights
	std::vector <World::LightPtr> {
		// TODO: objects with emmision
		World::LightPtr(new PointLight(transforms[0], 0.0f))
	}
};

int main()
{
	// Redirect logger to file
	// Logger::switch_file("mercury.log");

	mercury::Model <mercury::VERTEX_TYPE_POSITION> model("resources/debug.obj");
	model[0].material = materials[1];
	model[0].transform.scale = glm::vec3(0.5f);

	auto m1 = model[0];
	auto m2 = model[0];
	auto m3 = model[0];

	m2.transform.position = glm::vec3(-3.0f, 0.0f, -3.0f);
	m3.transform.position = glm::vec3(3.0f, 0.0f, 3.0f);

	world.objects.push_back(std::shared_ptr <mercury::Mesh <mercury::VERTEX_TYPE_POSITION>> (
		new mercury::Mesh <mercury::VERTEX_TYPE_POSITION> (m1)
	));
	
	world.objects.push_back(std::shared_ptr <mercury::Mesh <mercury::VERTEX_TYPE_POSITION>> (
		new mercury::Mesh <mercury::VERTEX_TYPE_POSITION> (m2)
	));
	
	world.objects.push_back(std::shared_ptr <mercury::Mesh <mercury::VERTEX_TYPE_POSITION>> (
		new mercury::Mesh <mercury::VERTEX_TYPE_POSITION> (m3)
	));

	Logger::ok() << "[main] Loaded model with "
		<< model.mesh_count() << " meshe(s), "
		<< model[0].vertex_count() << " vertices, "
		<< model[0].triangle_count() << " triangles" << std::endl;

	// Plane mesh
	float width = 10.0f;
	float length = 10.0f;

	mercury::Mesh <mercury::VERTEX_TYPE_POSITION> plane_mesh {
		{
			glm::vec3(-width/2, -0.1f, -length/2),
			glm::vec3(width/2, -0.1f, -length/2),
			glm::vec3(width/2, -0.1f, length/2),
			glm::vec3(-width/2, -0.1f, length/2)
		},
		{
			0, 1, 2,
			2, 3, 0
		},
		{
			.albedo = glm::vec3(1.0f, 1.0f, 0.7f),
			.reflectance = 0.0f,
			.refractance = 1.5f
		}
	};

	// Add plane to world
	world.objects.push_back(std::shared_ptr <mercury::Mesh <mercury::VERTEX_TYPE_POSITION>> (
		new mercury::Mesh <mercury::VERTEX_TYPE_POSITION> (plane_mesh)
	));

	Logger::notify("Transforms (model matrices) of all objects:");
	for (auto &object : world.objects) {
		glm::mat4 model = object->transform.model();
		glm::vec4 pos = model[3];
		Logger::notify() << "\t" << pos.x << ", " << pos.y << ", " << pos.z << std::endl;
	}

	// Save world into scene
	mercury::Scene scene("default_world", world);
	scene.save("resources/default_world.hg");

	// Initialize Vulkan
	Vulkan *vulkan = new Vulkan();
	vulkan->init_imgui();

	// Create sample scene
	MercuryApplication app(vulkan);

	app.update_descriptor_set();
	app.update_command_buffers();
	app.run();

	delete vulkan;
}
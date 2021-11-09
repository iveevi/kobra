#include <iostream>

#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include "include/camera.hpp"
#include "include/shader.hpp"
#include "include/model.hpp"
#include "include/init.hpp"
#include "include/logger.hpp"

// Using declarations
using namespace mercury;

// Forward declarations
void mouse_callback(GLFWwindow* window, double xpos, double ypos);
void scroll_callback(GLFWwindow* window, double xoffset, double yoffset);
void processInput(GLFWwindow *window);

void add_triangle(Mesh::AVertex &, Mesh::AIndices &,
	const glm::vec3 &, const glm::vec3 &, const glm::vec3 &);

Mesh cuboid(const glm::vec3 &center, float w, float h, float d);

// settings
const unsigned int SCR_WIDTH = 800;
const unsigned int SCR_HEIGHT = 600;

// camera
mercury::Camera camera(glm::vec3(0.0f, 0.0f, 3.0f));

float lastX = SCR_WIDTH / 2.0f;
float lastY = SCR_HEIGHT / 2.0f;
bool firstMouse = true;

// timing
float deltaTime = 0.0f;
float lastFrame = 0.0f;

// Shader source
const char *vertex = R"(
#version 330 core

layout (location = 0) in vec3 v_pos;
layout (location = 1) in vec3 v_normal;

out vec3 normal;
out vec3 frag_pos;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

void main()
{
	vec4 pos = vec4(v_pos, 1.0);
	gl_Position = projection * view * model * pos;
	frag_pos = vec3(model * pos);

	// TODO: this should be done on the cpu because its intensive
	normal = mat3(transpose(inverse(model))) * v_normal;
}
)";

const char *fragment_source = R"(
#version 330 core

out vec4 frag_color;

uniform vec3 color;

void main()
{
	frag_color = vec4(color, 1.0);
}
)";

const char *fragment_hit = R"(
#version 330 core

in vec3 normal;
in vec3 frag_pos;

out vec4 frag_color;

uniform vec3 color;
uniform vec3 light_color;
uniform vec3 light_pos;
uniform vec3 view_pos;

void main()
{
	// ambient
	float ambient_strength = 0.25;
	vec3 ambient = ambient_strength * light_color;

	// diffuse
	vec3 norm = normalize(normal);
	if (!gl_FrontFacing)
		norm *= -1;

	vec3 light_dir = normalize(light_pos - frag_pos);
	float diff = max(dot(norm, light_dir), 0.0);
	vec3 diffuse = diff * light_color;

	// specular
	float shine = 32;
	float specular_strength = 0.5;
	vec3 view_dir = normalize(view_pos - frag_pos);
	vec3 reflect_dir = reflect(-light_dir, norm);
	float spec = pow(max(dot(view_dir, reflect_dir), 0.0), shine);
	vec3 specular = specular_strength * spec * light_color;

	vec3 result = (ambient + diffuse + specular) * color;
	vec3 rgb_normal = 0.5 * norm + 0.5;

	frag_color = vec4(result, 1.0);
}
)";

float randm()
{
	return rand()/((float) RAND_MAX) - 0.5;
}

// TODO: put into common (.cpp/.hpp) file
// Overload printing glm::vec3
std::ostream &operator<<(std::ostream &os, const glm::vec3 &vec)
{
	return os << "<" << vec.x << ", " << vec.y
		<< ", " << vec.z << ">";
}

// TODO: these go into a namespace like "tree"
using Ring = std::vector <glm::vec3>;

void add_ring(Mesh::AVertex &vertices, Mesh::AIndices &indices,
		const Ring &ring1, const Ring &ring2)
{
	int divs = ring1.size();
	for (int i = 0; i < divs; i++) {
		int n = (i + 1) % divs;

		add_triangle(vertices, indices,
				ring1[i], ring2[i], ring1[n]);
		add_triangle(vertices, indices,
				ring2[i], ring2[n], ring1[n]);
	}
}

// TODO: Generate fewer vertices, and use more normal maps
void add_branch(Mesh::AVertex &vertices, Mesh::AIndices &indices,
		const glm::vec3 &p1, const glm::vec3 &p2,
		float rad_i, float rad_f,
		int nrings = 10, int nslices = 10)
{
	// Constants
	const float k_exp = std::log(rad_f/rad_i) / nrings;
	const float slice = 2 * acos(-1) / nslices;

	// Radius function
	auto radius = [k_exp, rad_i](float x) -> float {
		return rad_i * std::exp(k_exp * x);
	};

	// TODO: need some way to draw vectors
	const glm::vec3 axis = p2 - p1;

	// TODO: why dont we do cross product?
	// glm::vec3 perp = {1, 0, 0};

	// Generate the rotation matrix
	glm::mat4 transform(1);
	transform = glm::rotate(transform, -slice, axis);

	// List of rings
	std::vector <Ring> rings;

	// Spacial traverser
	glm::vec3 point = p1;

	// Adding the vertices
	for (int i = 0; i < nrings; i++) {
		// Next ring
		Ring ring;

		// Save perpendicular
		glm::vec3 c_perp = {1, 0, 0};

		float rad = radius(i);
		for (int j = 0; j < nslices; j++) {
			glm::vec3 v1 = point + (rad + 0.02f * randm()) * c_perp;
			c_perp = glm::vec3(transform * glm::vec4(c_perp, 1.0));
			ring.push_back(v1);
		}

		// Add the ring
		rings.push_back(ring);

		// Move along the branch
		point += axis/(float) nrings;
	}

	for (int i = 0; i < nrings - 1; i++)
		add_ring(vertices, indices, rings[i], rings[i + 1]);
}

int main()
{
	// TODO: do in init
	srand(clock());
	mercury::init();

	// tell stb_image.h to flip loaded texture's on the y-axis (before loading model).
	stbi_set_flip_vertically_on_load(true);

	// configure global opengl state
	glEnable(GL_DEPTH_TEST);

	// Draw in wireframe -- TODO: should be an init option (or live option)
	// glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);

	/* Shader meshShader(
		"resources/shaders/mesh_shader.vs",
		"resources/shaders/mesh_shader.fs"
	); */

	// Position of the light
	glm::vec3 lpos = {2, 1.6, 1.6};

	// Meshes
	Mesh hit_cube1 = cuboid({0.5, 0.5, 0.5}, 1, 1, 1);
	Mesh hit_cube2 = cuboid({3, -0.5, 0.0}, 1, 2, 1);
	Mesh source_cube = cuboid(lpos, 0.5, 0.5, 0.5);

	// Create shader and set base properties
	Shader source = Shader::from_source(vertex, fragment_source);
	Shader hit = Shader::from_source(vertex, fragment_hit);

	source.use();
	source.set_vec3("color", {1.0, 1.0, 1.0});

	hit.use();
	hit.set_vec3("light_color", {1.0, 1.0, 1.0});
	hit.set_vec3("light_pos", lpos);		// TODO: add a centroid method for meshes

	// Another branch alg
	Mesh::AVertex vertices;
	Mesh::AIndices indices;

	add_branch(vertices, indices, {4, 0, 2}, {4, 3, 2}, 0.7, 0.3, 25, 25);
	add_branch(vertices, indices, {4.1, 2.9, 2}, {5, 5, 2}, 0.25, 0.1);

	Mesh tree(vertices, {}, indices);

	// render loop
	// -----------
	while (!glfwWindowShouldClose(mercury::cwin.window))
	{
		// per-frame time logic
		// --------------------
		float currentFrame = glfwGetTime();
		deltaTime = currentFrame - lastFrame;
		lastFrame = currentFrame;

		// input
		// -----
		processInput(mercury::cwin.window);

		// render
		// ------
		glClearColor(0.05f, 1.00f, 0.05f, 1.0f);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

		// Create model, view, projection
		glm::mat4 model = glm::mat4(1.0f);
		model = glm::translate(model, glm::vec3(0.0f, 0.0f, 0.0f)); // translate it down so it's at the center of the scene
		model = glm::scale(model, glm::vec3(1.0f, 1.0f, 1.0f));	// it's a bit too big for our scene, so scale it down
		glm::mat4 view = camera.get_view();
		glm::mat4 projection = glm::perspective(glm::radians(camera.zoom),
				(float) SCR_WIDTH / (float) SCR_HEIGHT, 0.1f, 100.0f);

		// Modify the shader properties
		source.use();
		source.set_mat4("model", model);
		source.set_mat4("view", view);
		source.set_mat4("projection", projection);

		hit.use();
		hit.set_mat4("model", model);
		hit.set_mat4("view", view);
		hit.set_mat4("projection", projection);
		hit.set_vec3("view_pos", camera.position);

		// Draw the cubes
		source_cube.draw(source);

		hit.use();
		hit.set_vec3("color", {0.5, 1.0, 0.5});
		hit_cube1.draw(hit);

		hit.use();
		hit.set_vec3("color", {1.0, 0.5, 0.5});
		hit_cube2.draw(hit);

		hit.use();
		hit.set_vec3("color", {1.0, 0.8, 0.5});
		tree.draw(hit);

		// glfw: swap buffers and poll IO events (keys pressed/released, mouse moved etc.)
		// -------------------------------------------------------------------------------
		glfwSwapBuffers(mercury::cwin.window);
		glfwPollEvents();
	}

	// glfw: terminate, clearing all previously allocated GLFW resources.
	// ------------------------------------------------------------------
	glfwTerminate();
	return 0;
}

// process all input: query GLFW whether relevant keys are pressed/released this frame and react accordingly
// ---------------------------------------------------------------------------------------------------------
void processInput(GLFWwindow *window)
{
	if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
		glfwSetWindowShouldClose(window, true);

	float cameraSpeed = 5 * deltaTime;

	// Forward motion
	if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS)
		camera.move(cameraSpeed * camera.front);
	if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS)
		camera.move(-cameraSpeed * camera.front);

	// Lateral motion
	if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS)
		camera.move(-cameraSpeed * camera.right);
	if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS)
		camera.move(cameraSpeed * camera.right);

	// Vertical motion
	if (glfwGetKey(window, GLFW_KEY_E) == GLFW_PRESS)
		camera.move(-cameraSpeed * camera.up);
	if (glfwGetKey(window, GLFW_KEY_Q) == GLFW_PRESS)
		camera.move(cameraSpeed * camera.up);
}

// glfw: whenever the mouse moves, this callback is called
// -------------------------------------------------------
void mouse_callback(GLFWwindow* window, double xpos, double ypos)
{
	static const float sensitivity = 0.1f;

	if (firstMouse) {
		lastX = xpos;
		lastY = ypos;
		firstMouse = false;
	}

	float xoffset = xpos - lastX;
	float yoffset = lastY - ypos; // reversed since y-coordinates go from bottom to top

	lastX = xpos;
	lastY = ypos;

	// camera.ProcessMouseMovement(xoffset, yoffset);
	camera.add_yaw(xoffset * sensitivity);
	camera.add_pitch(yoffset * sensitivity);
}

// glfw: whenever the mouse scroll wheel scrolls, this callback is called
// ----------------------------------------------------------------------
void scroll_callback(GLFWwindow* window, double xoffset, double yoffset)
{
	// TODO: method to change zoom
	camera.zoom -= (float)yoffset;
	if (camera.zoom < 1.0f)
		camera.zoom = 1.0f;
	if (camera.zoom > 45.0f)
		camera.zoom = 45.0f;
}

void add_triangle(Mesh::AVertex &vertices, Mesh::AIndices &indices,
		const glm::vec3 &p1,
		const glm::vec3 &p2,
		const glm::vec3 &p3)
{
	glm::vec3 v1 = p2 - p1;
	glm::vec3 v2 = p3 - p1;
	glm::vec3 normal = glm::cross(v1, v2);

	unsigned int base = vertices.size();

	// Triangle coordinates
	Mesh::AVertex tmp {
		Vertex {.position = p1, .normal = normal},
		Vertex {.position = p2, .normal = normal},
		Vertex {.position = p3, .normal = normal}
	};

	// Add vertices
	vertices.insert(vertices.begin(), tmp.begin(), tmp.end());

	// Add indices
	indices.push_back(base);
	indices.push_back(base + 1);
	indices.push_back(base + 2);
}

void add_face(Mesh::AVertex &vertices, Mesh::AIndices &indices,
		const glm::vec3 &p1,
		const glm::vec3 &p2,
		const glm::vec3 &p3,
		const glm::vec3 &p4,
		const glm::vec3 &normal)
{
	add_triangle(vertices, indices, p1, p3, p2);
	add_triangle(vertices, indices, p1, p4, p3);
}

// Generates a cuboid mesh
Mesh cuboid(const glm::vec3 &center, float w, float h, float d)
{
	float w2 = w/2;
	float h2 = h/2;
	float d2 = d/2;

	// 6 faces
	Mesh::AVertex vertices;
	Mesh::AIndices indices;

	add_face(vertices, indices,
		{center.x - w2, center.y - h2, center.z - d2},
		{center.x + w2, center.y - h2, center.z - d2},
		{center.x + w2, center.y + h2, center.z - d2},
		{center.x - w2, center.y + h2, center.z - d2},
		{0, 0, -1}
	);

	add_face(vertices, indices,
		{center.x - w2, center.y - h2, center.z + d2},
		{center.x - w2, center.y + h2, center.z + d2},
		{center.x + w2, center.y + h2, center.z + d2},
		{center.x + w2, center.y - h2, center.z + d2},
		{0, 0, 1}
	);

	add_face(vertices, indices,
		{center.x - w2, center.y - h2, center.z - d2},
		{center.x - w2, center.y - h2, center.z + d2},
		{center.x + w2, center.y - h2, center.z + d2},
		{center.x + w2, center.y - h2, center.z - d2},
		{0, -1, 0}
	);

	add_face(vertices, indices,
		{center.x - w2, center.y + h2, center.z - d2},
		{center.x + w2, center.y + h2, center.z - d2},
		{center.x + w2, center.y + h2, center.z + d2},
		{center.x - w2, center.y + h2, center.z + d2},
		{0, 1, 0}
	);

	add_face(vertices, indices,
		{center.x - w2, center.y - h2, center.z - d2},
		{center.x - w2, center.y + h2, center.z - d2},
		{center.x - w2, center.y + h2, center.z + d2},
		{center.x - w2, center.y - h2, center.z + d2},
		{-1, 0, 0}
	);

	add_face(vertices, indices,
		{center.x + w2, center.y - h2, center.z - d2},
		{center.x + w2, center.y - h2, center.z + d2},
		{center.x + w2, center.y + h2, center.z + d2},
		{center.x + w2, center.y + h2, center.z - d2},
		{1, 0, 0}
	);

	return Mesh {vertices, {}, indices};
}

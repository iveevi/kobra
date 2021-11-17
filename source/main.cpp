#include <iostream>
#include <unordered_map>

// GLFW headers
#include <glad/glad.h>
#include <GLFW/glfw3.h>

// GLM headers
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

// Engine headers
#include "include/shader.hpp"
#include "include/model.hpp"
#include "include/init.hpp"
#include "include/logger.hpp"
#include "include/lighting.hpp"
#include "include/rendering.hpp"
#include "include/varray.hpp"

#include "include/engine/camera.hpp"
#include "include/engine/monitors.hpp"
#include "include/engine/skybox.hpp"

#include "include/math/linalg.hpp"

#include "include/mesh/basic.hpp"
#include "include/mesh/sphere.hpp"
#include "include/mesh/cube.hpp"

#include "include/ui/text.hpp"
#include "include/ui/ui_layer.hpp"
#include "include/ui/line.hpp"

// Using declarations
using namespace mercury;

// Forward declarations
void mouse_callback(GLFWwindow* window, double xpos, double ypos);
void scroll_callback(GLFWwindow* window, double xoffset, double yoffset);
void process_input(GLFWwindow *window, float);

// Camera
mercury::Camera camera(glm::vec3(0.0f, 0.0f, 3.0f));

// Daemons
lighting::Daemon ldam;
rendering::Daemon rdam;

// Annotations
std::vector <Drawable *> annotations;

Shader sphere_shader;		// TODO: change to annotation shader

void add_annotation(SVA3 *sva, const glm::vec3 &color)
{
	sva->color = color;
	size_t index = annotations.size();
	annotations.push_back(sva);
	rdam.add(annotations[index], &sphere_shader);
}

void add_annotation(ui::Line *line, const glm::vec3 &color)
{
	line->color = color;
	size_t index = annotations.size();
	// lines.push_back(*line);
	annotations.push_back(line);
	rdam.add(annotations[index], winman.cres.line_shader);
}

// TODO: these go into a namespace like "tree"

// Random [0, 1]
inline float runit()
{
	return rand()/((float) RAND_MAX);
}

using Ring = std::vector <glm::vec3>;
using Branch = std::vector <Ring>;		// Should be a queue so poping is easier

void add_ring(Mesh::AVertex &vertices, Mesh::AIndices &indices,
		const Ring &ring1, const Ring &ring2)
{
	int divs = ring1.size();
	for (int i = 0; i < divs; i++) {
		int n = (i + 1) % divs;

		mesh::add_triangle(vertices, indices,
				ring1[i], ring2[i], ring1[n]);
		mesh::add_triangle(vertices, indices,
				ring2[i], ring2[n], ring1[n]);
	}
}

// TODO: return the vector of rings
Branch generate_branch(
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

	// Add the branch line
	add_annotation(new ui::Line(p1, p2), {0.5f, 0.5f, 0.5f});

	// TODO: global color variables
	add_annotation(new SVA3(mesh::wireframe_sphere(p1, rad_i)), {0.0, 0.0, 1.0});
	add_annotation(new SVA3(mesh::wireframe_sphere(p2, rad_f)), {0.0, 0.0, 1.0});

	// Tree axis
	const glm::vec3 axis = p2 - p1;

	// Gram-Schmidt process to create an orthonormal vector
	glm::vec3 v2 = p1 + glm::vec3 {1.0f, 0.0f, 0.0f};
	glm::vec3 perp = glm::normalize(v2 - math::project(v2, axis));

	// Generate the rotation matrix
	glm::mat4 transform(1);
	transform = glm::rotate(transform, -slice, axis);

	// List of rings
	Branch rings;

	// Spacial traverser
	glm::vec3 point = p1;

	// Adding the vertices
	for (int i = 0; i <= nrings; i++) {
		// Next ring
		Ring ring;

		// Save perpendicular
		glm::vec3 c_perp = perp;// {1, 0, 0};

		float rad = radius(i);
		for (int j = 0; j < nslices; j++) {
			glm::vec3 v1 = point + (rad + 0.02f * runit()) * c_perp;
			c_perp = glm::vec3(transform * glm::vec4(c_perp, 1.0));
			ring.push_back(v1);
		}

		// Add the ring
		rings.push_back(ring);

		// Move along the branch
		point += axis/(float) nrings;
	}

	return rings;
}

// Add branch to vertices
void add_branch(Mesh::AVertex &vertices, Mesh::AIndices &indices,
		const Branch &branch)
{
	// Add the rings
	for (int i = 0; i < (int) branch.size() - 1; i++)
		add_ring(vertices, indices, branch[i], branch[i + 1]);
}

// Generate the tree skeleton
// TODO: later implement tropisms, etc
// NOTE: only draws the branchs that steam from the current trunk
Branch generate_tree(Mesh::AVertex &vertices, Mesh::AIndices &indices,
		const glm::vec3 &base,
		const glm::vec3 &up,
		float height,
		float tradius,
		int max_iterations = 2)
{
	// At least two branches
	size_t nbranches = (rand() % 2) + 2;

	// Draw the base
	add_annotation(new SVA3(mesh::wireframe_sphere(base, tradius)), {1.0, 1.0, 0.5});
	
	// Stopping point
	if (height < 0.2 || max_iterations <= 0) {
		// TODO: these are actually leaves
		return {};
	}

	const glm::vec3 color = glm::vec3 {1.0f, 0.5f, 1.0f};
	const glm::vec3 end = base + height * up;

	// lines.push_back(ui::Line(base, end, color));
	add_annotation(new ui::Line(base, end), color);

	Branch branch = generate_branch(base, end, tradius, 0.6 * tradius);
	// annotate_ring(annotations, branch[10]);

	float sliceh = height/10.0f;

	// TODO: will need to cull branches that are too close to each other
	float pi = acos(-1);

	// Store xy angle offset
	float xy = glm::dot(up, {0, 1, 0});
	for (size_t i = 0; i < nbranches; i++) {
		// Recurse
		float xy_angle = pi * runit() / 2;
		float xz_angle = 2 * pi * runit();

		glm::vec3 nup = math::sph_to_cart(xy_angle, xz_angle);

		// TODO: store min scale
		float hk = 0.6 * runit() + 0.4;
		float nheight = hk * height;

		Branch br = generate_tree(
			vertices, indices,
			end, nup,
			nheight, tradius * 0.6,
			max_iterations - 1
		);

		if (max_iterations <= 1)
			break;

		// Ensure that the bounding box for the branch is not intersecting
		float nsliceh = nheight/10.0f;
		float kheight = nsliceh/2.0f;

		add_branch(vertices, indices, br);
	}

	return branch;
}

Mesh make_tree(const glm::vec3 &base, const glm::vec3 &up, float height, float tradius)
{
	Mesh::AVertex vertices;
	Mesh::AIndices indices;

	std::vector <Ring> rings = generate_tree(vertices, indices, base, up, height, tradius);
	Logger::notify() << "Finished generating tree\n";

	// Draw the last branch
	add_branch(vertices, indices, rings);

	return Mesh(vertices, {}, indices);
}

// Render function for main window
Mesh hit_cube1;
Mesh hit_cube2;
Mesh hit_cube3;
Mesh tree;

// Skybox
Skybox sb;

// Lights
const glm::vec3 lpos1 {2, 1.6, 1.6};
const glm::vec3 lpos2 {0.2, 1.6, 1.6};

lighting::DirLight dirlight {
	{-0.2f, -1.0f, -0.3f},
	{0.2f, 0.2f, 0.2f},
        {0.9f, 0.9f, 0.9f},
	{1.0f, 1.0f, 1.0f}
};

// TODO: color constants and hex strings

// Wireframe sphere: TODO: is there a better way than to manually set vertices?
const glm::vec3 center {1.0, 1.0, 1.0};
const float radius = 0.2f;

void main_initializer()
{
	// Uncap FPS
	glfwSwapInterval(0);

	// TODO: put in init or smthing
	// stbi_set_flip_vertically_on_load(true);

	// Configure global opengl state
	glEnable(GL_DEPTH_TEST);
	glEnable(GL_MULTISAMPLE);

	// Hide cursor
	glfwSetInputMode(winman.cwin,
		GLFW_CURSOR, GLFW_CURSOR_DISABLED);

	// TODO: do in init
	srand(clock());

	// Draw in wireframe -- TODO: should be an init option (or live option)
	// glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);

	// Load resources
	winman.load_font(0);
	winman.load_skybox(0);
	winman.load_lines(0);

	// Meshes
	hit_cube1 = mesh::cuboid({0.5, 0, 0.5}, 1, 1, 1);
	hit_cube2 = mesh::cuboid({3, 0.5, 0.0}, 1, 2, 1);
	hit_cube3 = mesh::cuboid({3, -1.0, 0.0}, 10, 1, 10);

	// Construct the tree
	tree = make_tree({0, -0.5, -2}, {0, 1, 0}, 3.0f, 0.5f);
	
	// Set the materials
	hit_cube1.set_material({.color = {0.5, 1.0, 0.5}});
	hit_cube2.set_material({.color = {1.0, 0.5, 0.5}});
	hit_cube3.set_material({.color = {0.9, 0.9, 0.9}});

	tree.set_material({.color = {1.0, 0.8, 0.5}});

	// Set line width
	glLineWidth(5.0f);

	// Create the sphere
	sphere_shader = Shader(
		_shader("basic3d.vert"),
		_shader("basic.frag")
	);

	sphere_shader.set_name("sphere_shader");

	// TODO: some way to check that the resources being used in render are in another context

	// Skybox stuff
	sb = Skybox({
		"resources/textures/skybox/uv_4.png",
		"resources/textures/skybox/uv_2.png",
		"resources/textures/skybox/uv_1.png",
		"resources/textures/skybox/uv_6.png",
		"resources/textures/skybox/uv_3.png",
		"resources/textures/skybox/uv_5.png"
	});

	// Add objects and lights to the ldam system
	ldam = lighting::Daemon(&rdam);

	ldam.add_light(dirlight);

	ldam.add_object(&hit_cube1);
	ldam.add_object(&hit_cube2);
	ldam.add_object(&hit_cube3);

	// ldam.add_object(&tree);

	// Add objects to the render daemon
	rdam.add(&sb, winman.cres.sb_shader);
}

void main_renderer()
{
	// Total time
	static float time = 0;
	static float delta_t = 0;
	static float last_t = 0;

	// Get time stuff
	float current_frame = glfwGetTime();
	delta_t = current_frame - last_t;
	last_t = current_frame;

	// Process input
	process_input(mercury::winman.cwin, delta_t);	// TODO: return movement of camera

	// render
	glClearColor(0.05f, 1.00f, 0.05f, 1.0f);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	// TODO: rerender all this only if the camera has moved

	// Create model, view, projection matrices
	glm::mat4 model = glm::mat4(1.0f);
	model = glm::translate(model, glm::vec3(0.0f, 0.0f, 0.0f));
	model = glm::scale(model, glm::vec3(1.0f, 1.0f, 1.0f));		// TODO: what is the basis of this computation?

	glm::mat4 view = camera.get_view();
	glm::mat4 projection = glm::perspective(
		glm::radians(camera.zoom),
		winman.width / winman.height,
		0.1f, 100.0f
	);

	// Set lighting daemon uniforms
	ldam.uniforms = {
		model,
		view,
		projection,
		camera.position
	};

	// Lighut and render scene
	ldam.light();
	rdam.render();

	// Draw sphere		TODO: seriously some way to check that uniforms have been set
	sphere_shader.use();	// TODO: make into a common shader
	sphere_shader.set_mat4("model", model);
	sphere_shader.set_mat4("view", view);
	sphere_shader.set_mat4("projection", projection);

	// TODO: make a set_mvp function and helper in winman
	Shader *lshader = winman.cres.line_shader;
	lshader->use();
	lshader->set_mat4("model", model);
	lshader->set_mat4("view", view);
	lshader->set_mat4("projection", projection);

	// Draw skybox
	view = glm::mat4(glm::mat3(camera.get_view()));

	// Set skybox shader properties
	Shader *sshader = winman.cres.sb_shader;
	sshader->use();
	sshader->set_mat4("projection", projection);
	sshader->set_mat4("view", view);
}

// Program render loop condition
bool rcondition()
{
	return !glfwWindowShouldClose(winman[0]);
}

int main()
{
	// Initialize mercury
	init();

	// Set winman bindings
	winman.set_condition(rcondition);

	winman.set_initializer(0, main_initializer);
	winman.set_renderer(0, main_renderer);

	// Render loop
	winman.run();

	// Terminate GLFW
	glfwTerminate();

	return 0;
}

// process all input: query GLFW whether relevant keys are pressed/released this frame and react accordingly
// ---------------------------------------------------------------------------------------------------------
void process_input(GLFWwindow *window, float delta_t)
{
	if (glfwGetKey(window, GLFW_KEY_BACKSPACE) == GLFW_PRESS)
		glfwSetWindowShouldClose(window, true);

	if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS) {
		// glfwSetWindowShouldClose(window, true);
		glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
	}

	float cameraSpeed = 5 * delta_t;

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

// Variables for mouse movement
float lastX = 0.0; // SCR_WIDTH / 2.0f;
float lastY = 0.0; // SCR_HEIGHT / 2.0f;
bool firstMouse = true;

// Mouse callback
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

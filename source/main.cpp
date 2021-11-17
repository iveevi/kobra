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
#include "include/mesh/cuboid.hpp"

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

// Annotation colors
glm::vec3 spheres {0.6f, 0.6f, 1.0f};
glm::vec3 lines {1.0f, 0.6f, 1.0f};

// TODO: these go into a namespace like "tree"

// Random [0, 1]
inline float runit()
{
	return rand()/((float) RAND_MAX);
}

// Maximum number of branches
const int MAX_BRANCHES = 4;

// Tree skeleton structure
struct Skeleton {
	glm::vec3 base;
	glm::vec3 tip;
	glm::vec3 saddle;	// Offset to saddle point

	float rad_i;
	float rad_f;

	size_t nbranches = 0;

	Skeleton *branches[MAX_BRANCHES];

	// TODO: static free method or deconstructor?
};

// Generate the tree skeleton
// TODO: Later, base the parameters of creation
//	on real life studies (e.g. height, age, tropisms, etc.)
Skeleton *generate_skeleton(const glm::vec3 &base,
		const glm::vec3 &up,
		float length,
		float rad_i,
		float rad_f,
		int max_iterations = 2)
{
	// Constants
	static const float pi = acos(-1);	// TODO: put in math header
	static const float sk = 1.5f;		// Saddle point weight on base branch
	
	// Stopping point, branch turns into leaves:
	//	- max number of iterations is reached
	//	- height is too small
	//	- radius is too small
	if (max_iterations <= 0
			|| length <= 0.2f
			|| rad_i <= 0.01f) {
		return new Skeleton {
			.base = base,
			.tip = base,
			.rad_i = 0.0f,
			.rad_f = 0.0f
		};
	}

	// Compute new values
	glm::vec3 tip = base + length * up;
	size_t nbranches = (rand() % MAX_BRANCHES) + 1;	// At least 1 branch

	// Initialize
	Skeleton *out = new Skeleton{
		.base = base,
		.tip = tip,
		.saddle = {0.0, 0.0, 0.0},
		.rad_i = rad_i,
		.rad_f = rad_f,
		.nbranches = nbranches
	};

	// Iterate through branches
	for (size_t i = 0; i < nbranches; i++) {
		// New parameters
		// TODO: need to space the branches out more
		float xy_angle = pi * runit() / 2;
		float xz_angle = 2 * pi * runit();
		
		glm::vec3 nup = math::sph_to_cart(xy_angle, xz_angle);
		
		// Height and radius factors
		float hk = 0.3 * runit() + 0.4;
		float rk = 0.3 * runit() + 0.4;

		// Create the next branch and append it
		Skeleton *br = generate_skeleton(
			tip, nup,
			hk * length,
			rad_f, rad_f * 0.6,
			max_iterations - 1
		);

		out->branches[i] = br;
		out->saddle += br->tip - tip;
	}

	// Average the saddle
	out->saddle /= sk * (nbranches + 1);

	return out;
}

// Annotate the skeleton
void annotate_skeleton(Skeleton *sk)
{
	// Constants
	static const glm::vec3 saddles = {1.0f, 0.7f, 0.5f};

	// Annotate current branch
	add_annotation(new ui::Line(sk->base, sk->tip), lines);
	add_annotation(new SVA3(mesh::wireframe_sphere(sk->tip, sk->rad_f)), spheres);

	// Only draw saddle if there are branches
	if (sk->nbranches > 0) {
		add_annotation(new ui::Line(sk->tip, sk->tip + sk->saddle), saddles);
		add_annotation(new SVA3(mesh::wireframe_sphere(sk->tip + sk->saddle, 0.1f)), saddles);
	}

	// Annotate branches
	for (int i = 0; i < sk->nbranches; i++) {
		// Draw line to saddle
		add_annotation(new ui::Line(sk->branches[i]->tip, sk->tip + sk->saddle), saddles);

		annotate_skeleton(sk->branches[i]);
	}
}

Mesh make_tree(const glm::vec3 &base, const glm::vec3 &up, float length, float rad_i, float rad_f)
{
	Mesh::AVertex vertices;
	Mesh::AIndices indices;

	Skeleton *skeleton = generate_skeleton(base, up, length, rad_i, rad_f);
	annotate_skeleton(skeleton);

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
	tree = make_tree({0, -0.5, -2}, {0, 1, 0}, 3.0f, 0.5f, 0.2f);
	
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

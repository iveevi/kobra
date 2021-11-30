#include <iostream>
#include <unordered_map>

// GLFW headers
#include <glad/glad.h>
#include <GLFW/glfw3.h>

// GLM headers
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

// Engine headers
#include "include/mercury.hpp"

// Using declarations
using namespace mercury;

// Forward declarations
void mouse_callback(GLFWwindow* window, double xpos, double ypos);
void scroll_callback(GLFWwindow* window, double xoffset, double yoffset);
void process_input(GLFWwindow *window, float);

// Camera
mercury::Camera camera(glm::vec3(0.0f, 1.0f, 7.5f));

// Daemons
lighting::Daemon	ldam;
rendering::Daemon	rdam;
physics::Daemon		pdam;

btDiscreteDynamicsWorld* dynamicsWorld;

// Annotations
std::vector <Drawable *> annotations;

Shader sphere_shader;		// TODO: change to annotation shader

void add_annotation(SVA3 *sva, const glm::vec3 &color, Transform *transform = nullptr)
{
	static Transform default_transform;

	sva->color = color;
	size_t index = annotations.size();
	annotations.push_back(sva);
	rdam.add(
		annotations[index],
		&sphere_shader,
		(transform ? transform : &default_transform)
	);
}

void add_annotation(ui::Line *line, const glm::vec3 &color)
{
	line->color = color;
	size_t index = annotations.size();
	// lines.push_back(*line);
	annotations.push_back(line);
	rdam.add(annotations[index], winman.cres.line_shader);
}

// Render function for main window
Mesh hit_cube1;
Mesh hit_cube2;
Mesh hit_cube3;

// TODO: need to fix bug when t1 is at low y (~0.32)
Transform t1({-4, 10, 0}, glm::radians(glm::vec3 {30, 30, 30}));
Transform t2({5.5, 14, 0}, glm::radians(glm::vec3 {0, 0, 90}));
Transform t3({0, -1, 0}, glm::radians(glm::vec3(0, 0, 10)));

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
physics::CollisionObject co1 {
	new btBoxShape(vec3 {1, 1, 1}/2),
	&t1,
	1
};

physics::CollisionObject co2 {
	new btBoxShape(vec3 {1, 2, 1}/2),
	&t2,
	1
};

physics::CollisionObject co3 {
	new btBoxShape(vec3 {10, 1, 10}/2),
	&t3,
	0
};

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
	hit_cube1 = mesh::cuboid({0, 0, 0}, 1, 1, 1);
	hit_cube2 = mesh::cuboid({0, 0, 0}, 1, 2, 1);
	hit_cube3 = mesh::cuboid({0, 0, 0}, 10, 1, 10);	// TODO: size only function
	
	// Set the materials
	hit_cube1.set_material({.color = {0.5, 1.0, 0.5}});
	hit_cube2.set_material({.color = {1.0, 0.5, 0.5}});
	hit_cube3.set_material({.color = {0.9, 0.9, 0.9}});

	// hit_cube1.set_wireframe();
	// hit_cube2.set_wireframe();

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

	ldam.add_object(&hit_cube1, &t1);
	ldam.add_object(&hit_cube2, &t2);
	ldam.add_object(&hit_cube3, &t3);

	// ldam.add_object(&tree);

	// Add objects to the render daemon
	rdam.add(&sb, winman.cres.sb_shader);

	pdam.add(co1);
	pdam.add(co2);
	pdam.add(co3);
}

// TODO: into linalg
glm::mat4 _mk_model(const glm::vec3 &translation = {0, 0, 0}, const glm::vec3 &scale = {1, 1, 1})
{
	glm::mat4 model = glm::mat4(1.0f);
	model = glm::translate(model, translation);
	model = glm::scale(model, scale);		// TODO: what is the basis of this computation?
	return model;
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

	/* Update the monitor
	tui::tui.update();
	tui::tui.update_fps(delta_t); */

	// Process input
	process_input(mercury::winman.cwin, delta_t);	// TODO: return movement of camera

	// render
	glClearColor(0.05f, 1.00f, 0.05f, 1.0f);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	// TODO: rerender all this only if the camera has moved

	// View and projection matrices
	glm::mat4 view = camera.get_view();
	glm::mat4 projection = glm::perspective(
		glm::radians(camera.zoom),
		winman.width / winman.height,
		0.1f, 100.0f
	);

	// Set lighting daemon uniforms
	ldam.uniforms = {
		_mk_model(),
		view,
		projection,
		camera.position
	};

	// Lighut and render scene
	/* dynamicsWorld->stepSimulation(delta_t, 10);	// TODO: update transforms in the physics daemon
	dynamicsWorld->debugDrawWorld(); */
	pdam.update(delta_t);
	ldam.light();
	rdam.render();

	// Draw sphere		TODO: seriously some way to check that uniforms have been set
	sphere_shader.use();	// TODO: make into a common shader
	sphere_shader.set_mat4("model", _mk_model());
	sphere_shader.set_mat4("view", view);
	sphere_shader.set_mat4("projection", projection);

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
	// tui::tui.init();

	/* collision configuration contains default setup for memory, collision setup. Advanced users can create their own configuration.
	btDefaultCollisionConfiguration* collisionConfiguration = new btDefaultCollisionConfiguration();

	///use the default collision dispatcher. For parallel processing you can use a diffent dispatcher (see Extras/BulletMultiThreaded)
	btCollisionDispatcher* dispatcher = new btCollisionDispatcher(collisionConfiguration);

	///btDbvtBroadphase is a good general purpose broadphase. You can also try out btAxis3Sweep.
	btBroadphaseInterface* overlappingPairCache = new btDbvtBroadphase();

	///the default constraint solver. For parallel processing you can use a different solver (see Extras/BulletMultiThreaded)
	btSequentialImpulseConstraintSolver* solver = new btSequentialImpulseConstraintSolver;

	dynamicsWorld = new btDiscreteDynamicsWorld(dispatcher, overlappingPairCache, solver, collisionConfiguration);
	// dynamicsWorld->setGravity(btVector3(0, -10, 0));
	
	physics::BulletDebugger *debugger = new physics::BulletDebugger();
	debugger->setDebugMode(btIDebugDraw::DBG_DrawWireframe);
	dynamicsWorld->setDebugDrawer(debugger); */

	// Set winman bindings
	winman.set_condition(rcondition);

	winman.set_initializer(0, main_initializer);
	winman.set_renderer(0, main_renderer);

	// Render loop
	winman.run();

	// Terminate GLFW
	// tui::tui.deinit();
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

	// Rotating a box
	if (glfwGetKey(window, GLFW_KEY_LEFT) == GLFW_PRESS)
		t2.rotate(0.05f * glm::vec3(0, 0, 1));

	if (glfwGetKey(window, GLFW_KEY_RIGHT) == GLFW_PRESS)
		t2.rotate(-0.05f * glm::vec3(0, 0, 1));
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

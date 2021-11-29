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
#include "include/math/compat.hpp"

#include <btBulletDynamicsCommon.h>

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

// Collision object components
// TODO: need to fix bug when t1 is at low y (~0.32)
Transform t1({-4, 10, 0}, glm::radians(glm::vec3 {30, 30, 30}));
Transform t2({5.5, 14, 0}, glm::radians(glm::vec3 {0, 0, 90}));
Transform t3({0, -1, 0});

/* physics::BoxCollider t1_collider({1, 1, 1}, &t1);
physics::BoxCollider t2_collider({1, 2, 1}, &t2);
physics::BoxCollider t3_collider({10, 1, 10}, &t3);

physics::CollisionObject t1_co(&t1_collider, physics::CollisionObject::Type::DYNAMIC);
physics::CollisionObject t2_co(&t2_collider, physics::CollisionObject::Type::DYNAMIC);
physics::CollisionObject t3_co(&t3_collider); */

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

struct BulletDebugger : public btIDebugDraw {
	int dbmode;

	void drawLine(const btVector3 &from, const btVector3 &to, const btVector3 &color) override {
		glm::vec3 from_glm {from.x(), from.y(), from.z()};
		glm::vec3 to_glm {to.x(), to.y(), to.z()};
		glm::vec3 color_glm {color.x(), color.y(), color.z()};

		ui::Line line(from_glm, to_glm, color_glm);
		line.draw(&sphere_shader);
	}

	void drawContactPoint(const btVector3 &PointOnB, const btVector3 &normalOnB, btScalar distance, int lifeTime, const btVector3 &color) override {
		glm::vec3 point_glm {PointOnB.x(), PointOnB.y(), PointOnB.z()};
		glm::vec3 normal_glm {normalOnB.x(), normalOnB.y(), normalOnB.z()};
		glm::vec3 color_glm {color.x(), color.y(), color.z()};

		ui::Line line(point_glm, point_glm + normal_glm, color_glm);
		line.draw(winman.cres.line_shader);
	}

	void reportErrorWarning(const char *warningString) override {
		Logger::error() << " BULLET: " << warningString << std::endl;
	}

	void draw3dText(const btVector3 &location, const char *textString) override {}

	void setDebugMode(int debugMode) override {
		dbmode = debugMode;
	}

	int getDebugMode() const override {
		return dbmode;
	}
};

// TODO: add custom vector class for casting between glm vec and btVector3
// and also quaternions
btAlignedObjectArray<btCollisionShape*> collisionShapes;
btCollisionObject *add_body(float m, const vec3 &dim, Transform *transform)
{
	// TODO: multiply size by scale?
	btCollisionShape* groundShape = new btBoxShape(dim/2);

	collisionShapes.push_back(groundShape);

	btTransform groundTransform;
	groundTransform.setIdentity();

	vec3 t = transform->translation;
	quat q = transform->orient;
	groundTransform.setOrigin(t);
	groundTransform.setRotation(q);

	btScalar mass(m);

	//rigidbody is dynamic if and only if mass is non zero, otherwise static
	bool isDynamic = (mass != 0.f);

	btVector3 localInertia(0, 0, 0);
	if (isDynamic)
		groundShape->calculateLocalInertia(mass, localInertia);

	//using motionstate is optional, it provides interpolation capabilities, and only synchronizes 'active' objects
	btDefaultMotionState* myMotionState = new btDefaultMotionState(groundTransform);
	btRigidBody::btRigidBodyConstructionInfo rbInfo(mass, myMotionState, groundShape, localInertia);
	btRigidBody* body = new btRigidBody(rbInfo);

	//add the body to the dynamics world
	dynamicsWorld->addRigidBody(body);
	return body;
}

btCollisionObject *co1;
btCollisionObject *co2;
btCollisionObject *co3;

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

	co1 = add_body(1, {1, 1, 1}, &t1);
	co2 = add_body(1, {1, 2, 1}, &t2);
	co3 = add_body(0, {10, 1, 10}, &t3);
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
	// pdam.update(delta_t, &rdam, &sphere_shader);
	dynamicsWorld->stepSimulation(delta_t, 10);	// TODO: update transforms in the physics daemon
	dynamicsWorld->debugDrawWorld();
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

	/* Draw bounding boxes
	physics::AABB ab;
	SVA3 box;
	
	ab = t1_collider.aabb();
	box = mesh::wireframe_cuboid(ab.center, ab.size);
	box.color = {1.0, 1.0, 0.5};
	box.draw(&sphere_shader);
	
	ab = t2_collider.aabb();
	box = mesh::wireframe_cuboid(ab.center, ab.size);
	box.color = {1.0, 1.0, 0.5};
	box.draw(&sphere_shader);
	
	ab = t3_collider.aabb();
	box = mesh::wireframe_cuboid(ab.center, ab.size);
	box.color = {1.0, 1.0, 0.5};
	box.draw(&sphere_shader); */
	btVector3 p;
	btQuaternion q;
	
	p = co1->getWorldTransform().getOrigin();
	q = co1->getWorldTransform().getRotation();
	t1.translation = {p.getX(), p.getY(), p.getZ()};
	t1.orient = {q.getW(), q.getX(), q.getY(), q.getZ()};

	p = co2->getWorldTransform().getOrigin();
	q = co2->getWorldTransform().getRotation();
	t2.translation = {p.getX(), p.getY(), p.getZ()};
	t2.orient = {q.getW(), q.getX(), q.getY(), q.getZ()};
	
	p = co3->getWorldTransform().getOrigin();
	q = co3->getWorldTransform().getRotation();
	t3.translation = {p.getX(), p.getY(), p.getZ()};
	t3.orient = {q.getW(), q.getX(), q.getY(), q.getZ()};
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

	///collision configuration contains default setup for memory, collision setup. Advanced users can create their own configuration.
	btDefaultCollisionConfiguration* collisionConfiguration = new btDefaultCollisionConfiguration();

	///use the default collision dispatcher. For parallel processing you can use a diffent dispatcher (see Extras/BulletMultiThreaded)
	btCollisionDispatcher* dispatcher = new btCollisionDispatcher(collisionConfiguration);

	///btDbvtBroadphase is a good general purpose broadphase. You can also try out btAxis3Sweep.
	btBroadphaseInterface* overlappingPairCache = new btDbvtBroadphase();

	///the default constraint solver. For parallel processing you can use a different solver (see Extras/BulletMultiThreaded)
	btSequentialImpulseConstraintSolver* solver = new btSequentialImpulseConstraintSolver;

	dynamicsWorld = new btDiscreteDynamicsWorld(dispatcher, overlappingPairCache, solver, collisionConfiguration);
	dynamicsWorld->setGravity(btVector3(0, -10, 0));
	
	BulletDebugger *debugger = new BulletDebugger();
	debugger->setDebugMode(btIDebugDraw::DBG_DrawWireframe);
	dynamicsWorld->setDebugDrawer(debugger);

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

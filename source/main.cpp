#include <iostream>
#include <unordered_map>

// GLFW headers
#include <glad/glad.h>
#include <GLFW/glfw3.h>

// GLM headers
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

// Engine headers
#include "include/camera.hpp"
#include "include/shader.hpp"
#include "include/model.hpp"
#include "include/init.hpp"
#include "include/logger.hpp"
#include "include/lighting.hpp"
#include "include/varray.hpp"

#include "include/engine/skybox.hpp"

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
void process_input(GLFWwindow *window);

// Settings
const unsigned int SCR_WIDTH = 800;
const unsigned int SCR_HEIGHT = 600;

// Camera
mercury::Camera camera(glm::vec3(0.0f, 0.0f, 3.0f));

// Variables for mouse movement
float lastX = SCR_WIDTH / 2.0f;
float lastY = SCR_HEIGHT / 2.0f;
bool firstMouse = true;

// Variables for timing
float delta_time = 0.0f;
float last_frame = 0.0f;

// Random [0, 1]
float runit()
{
	return rand()/((float) RAND_MAX);
}

// TODO: these go into a namespace like "tree"
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

// TODO: Generate fewer vertices, and use more normal maps
std::vector <ui::Line> lines;

std::vector <SVA3> spheres;		// NOTE: wireframe objects and lines are not considered in lighting...
std::vector <SVA3> annotations;

// Radial spacing constants
const float rk_max = 2.5f;
const float rk_min = 1.5f;

// TODO: move this to a glm extra header

// Project v over u
glm::vec3 project(const glm::vec3 &v, const glm::vec3 &u)
{
	return u * glm::dot(v, u)/glm::dot(u, u);
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
	// const float rk = 2.0;					// Radial spacing constant

	// Radius function
	auto radius = [k_exp, rad_i](float x) -> float {
		return rad_i * std::exp(k_exp * x);
	};

	// Add the branch line
	ui::Line line({
		p1.x, p1.y, p1.z,	// TODO: Add a better constructor (with just two vec3s)
		p2.x, p2.y, p2.z
	});
	line.color = {0.5, 0.5, 1.0};

	lines.push_back(line);

	spheres.push_back(mesh::wireframe_sphere(p1, rad_i));
	spheres.push_back(mesh::wireframe_sphere(p2, rad_f));
	
	// spheres2.push_back(mesh::wireframe_sphere(p1, rk_max * rad_i));
	// spheres2.push_back(mesh::wireframe_sphere(p2, rk_max * rad_f));

	// Tree axis
	const glm::vec3 axis = p2 - p1;

	// Gram-Schmidt process to create an orthonormal vector
	glm::vec3 v2 = p1 + glm::vec3 {1.0f, 0.0f, 0.0f};
	glm::vec3 perp = glm::normalize(v2 - project(v2, axis));

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

// Spherical coordinates to cartesian
glm::vec3 sph_to_cart(float xy, float xz)
{
	float xr_rad = cos(xy);
	return glm::vec3(
		xr_rad * cos(xz),
		sin(xy),
		xr_rad * sin(xz)
	);
}

// Add branch to vertices
void add_branch(Mesh::AVertex &vertices, Mesh::AIndices &indices,
		const Branch &branch)
{
	// Add the rings
	for (int i = 0; i < (int) branch.size() - 1; i++)
		add_ring(vertices, indices, branch[i], branch[i + 1]);
}

using Interval = std::pair <float, float>;

// Bounding box structure
struct BoundingBox {
	glm::vec3 center;
	glm::vec3 size;
	glm::vec3 up;
	glm::vec3 right;

	void annotate() {
		annotations.push_back(mesh::wireframe_cuboid(
			center, size, up
		));
	}

	// TODO: put both these methods outside bbox, into header in physics namespace
	// Return min and max of projection on axis
	float proj_len(const glm::vec3 &pt, const glm::vec3 &naxis) const
	{
		return glm::length(::project(pt, naxis));
	}

	Interval project(const glm::vec3 &axis) const
	{
		float min = std::numeric_limits <float> ::max();
		float max = -std::numeric_limits <float> ::max();
        
		glm::vec3 nup = glm::normalize(up);
		glm::vec3 nright = glm::normalize(right);
		glm::vec3 nforward = glm::normalize(glm::cross(nup, nright));

		glm::vec3 xdir = nright * size.x/2.0f;
		glm::vec3 ydir = nup * size.y/2.0f;
		glm::vec3 zdir = nforward * size.z/2.0f;

		std::vector <glm::vec3> pts = {
			center + xdir + ydir + zdir,
			center - xdir + ydir + zdir,
			center + xdir - ydir + zdir,
			center - xdir - ydir + zdir,
			center + xdir + ydir - zdir,
			center - xdir + ydir - zdir,
			center + xdir - ydir - zdir,
			center - xdir - ydir - zdir
		};

		for (const glm::vec3 &pt : pts) {
			float len = proj_len(pt, axis);
			// Logger::notify() << "\t\t\tlen = " << len << ", pt = " << pt << ", axis = " << axis << ", projction = " << ::project(pt, axis) << "\n";
			if (len < min)
				min = len;
			if (len > max)
				max = len;
		}

		return {min, max};
	}
};

bool intersecting(const BoundingBox &b1, const BoundingBox &b2)
{
	glm::vec3 foward1 = glm::normalize(glm::cross(b1.up, b1.right));
	glm::vec3 foward2 = glm::normalize(glm::cross(b2.up, b2.right));

	std::vector <glm::vec3> axes {
		// b1 normals
		b1.right,
		b1.up,
		foward1,

		// b2 normals
		b2.right,
		b2.up,
		foward2,

		// cross products
		glm::cross(b1.right, b2.right),
		glm::cross(b1.right, b2.up),
		glm::cross(b1.right, foward2),

		glm::cross(b1.up, b2.right),
		glm::cross(b1.up, b2.up),
		glm::cross(b1.up, foward2),

		glm::cross(foward1, b2.right),
		glm::cross(foward1, b2.up),
		glm::cross(foward1, foward2)
	};

	// Iterate over all axes
	Logger::notify() << "\tIntersection function\n";
	for (const glm::vec3 &axis : axes) {
		if (axis == glm::vec3(0, 0, 0)) {
			Logger::notify() << "\t\tAxis is zero\n";
			continue;
		}

		Interval i1 = b1.project(axis);
		Interval i2 = b2.project(axis);

		Logger::notify() << "\t\ti1 = " << i1.first << ", " << i1.second << "\ti2 = " << i2.first << ", " << i2.second << "\n";

		if (i1.first > i2.second || i2.first > i1.second)
			return false;
	}

	return true;
}

// Get the bounding box
BoundingBox get_bounding_box(const Ring &ring1, const Ring &ring2,
		const glm::vec3 &center, const glm::vec3 &up,
		float rheight)
{
	// TODO: add a helper method for this...
	glm::vec3 c1 = center - rheight * up/2.0f;
	glm::vec3 c2 = center + rheight * up/2.0f;

	// First ring
	float max1 = 0.0f;

	// Logger::notify() << "\tcenter = " << center << ", c1 = " << c1 << ", c2 = " << c2 << std::endl;
	for (int i = 0; i < (int) ring1.size(); i++) {
		float d = glm::length(ring1[i] - c1);
		if (d > max1)
			max1 = d;
	}

	// Second ring
	float max2 = 0.0f;

	for (int i = 0; i < (int) ring2.size(); i++) {
		float d = glm::length(ring2[i] - c2);
		if (d > max2)
			max2 = d;
	}

	// Add the bounding box
	float radius = 2 * std::fmax(max1, max2);
	// Logger::notify() << "\tradius = " << radius << "\n";

	return {
		center,
		{radius, rheight, radius},
		up,
		{radius, 0, 0}
	};
}

// Generate the tree skeleton
// TODO: later implement tropisms, etc
// NOTE: only draws the branchs that steam from the current trunk
Branch generate_tree(Mesh::AVertex &vertices, Mesh::AIndices &indices,
		const glm::vec3 &base,
		const glm::vec3 &up,
		float height,
		float tradius,
		int max_iterations = 4)
{
	// At least two branches
	size_t nbranches = (rand() % 2) + 2;

	// Stopping point
	spheres.push_back(mesh::wireframe_sphere(base, tradius));
	if (height < 0.2 || max_iterations <= 0) {
		// TODO: these are actually leaves
		return {};
	}

	const glm::vec3 color = glm::vec3 {1.0f, 0.5f, 1.0f};
	const glm::vec3 end = base + height * up;

	lines.push_back(ui::Line(base, end, color));

	Branch branch = generate_branch(base, end, tradius, 0.6 * tradius);

	float sliceh = height/10.0f;
	Logger::notify() << "max_iterations = " << max_iterations << std::endl;
	// get_bounding_box(branch[0], branch[1], base + (sliceh/2.0f) * up, up, sliceh);

	// Main bounding box
	BoundingBox mbbox = get_bounding_box(
		branch[9], branch[10],
		base + (height - sliceh/2.0f) * up,
		up, sliceh
	);
	mbbox.annotate();

	// Logger::notify() << "bbox test.\n";
	// bool b1 = mbbox.is_inside(mbbox.center);
	// bool b2 = mbbox.is_inside(mbbox.center + mbbox.size/2.0f);

	// Logger::notify() << "\tb1 = " << b1 << ", b2 = " << b2 << "\n";

	// TODO: will need to cull branches that are too close to each other
	float pi = acos(-1);

	/* Simple bounding box test:
	 * if the ring's bounding box is
	 * intersecting the main bounding box,
	 * (or two branch's bounding boxes are
	 * intersecting each other), then
	 * remove the ring from the branch
	 */

	std::vector <Branch> branches;		// TODO: why do I need this?
	for (size_t i = 0; i < nbranches; i++) {
		// Recurse
		float xy_angle = pi * runit() / 2;
		float xz_angle = 2 * pi * runit();

		glm::vec3 nup = sph_to_cart(xy_angle, xz_angle);

		// TODO: store min scale
		float hk = 0.5 * runit() + 0.5;
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

		int len = br.size();

		BoundingBox bbox;
		for (int j = 0; j < len; j++, kheight += nsliceh) {
			bbox = get_bounding_box(
				br[j], br[j + 1],
				end + kheight * nup,
				nup, nsliceh
			);

			bool inter = intersecting(mbbox, bbox);
			Logger::notify() << "Index is [" << i << ", " << j << "]: intersect? " << inter << std::endl;
			if (intersecting(mbbox, bbox)) {
				br.erase(br.begin());
			} else {
				break;
			}
		}

		add_branch(vertices, indices, br);
		bbox.annotate();
		// branches.push_back(br);
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

// Render function for FPS monitor
Shader basic;

// TODO: basic_2d.vert
const char *basic_vert = R"(
#version 330 core

layout (location = 0) in vec3 vertex;

uniform mat4 projection;

void main()
{
	gl_Position = projection * vec4(vertex, 1.0);
}
)";

const char *basic_frag = R"(
#version 330 core

out vec4 color;

uniform vec3 ecolor;

void main()
{
	color = vec4(ecolor, 1.0);
}
)";

// FPS monitor

// Graph vertices: normalize for [0, 100] for each coordinate
std::vector <float> fps_vertices(3 * 10);

template <unsigned int fields>
class DynamicVA : public VertexArray <fields> {
	std::vector <float> _vertices;

	// TODO: nead indexing operations,
	// maybe even iterators...
};

// Graph points
unsigned int vbo;
unsigned int vao;

// Axes
SVA3 axes;

// Text
ui::Text text_fps;

// Reload the buffer
void generate_arrays()
{
	// Load graph buffer
	glBindBuffer(GL_ARRAY_BUFFER, vbo);
	glCheckError();

	glBufferData(GL_ARRAY_BUFFER, sizeof(float) * fps_vertices.size(),
		&fps_vertices[0], GL_STATIC_DRAW);
	glCheckError();

	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE,
		3 * sizeof(float), (void *) 0);
	glCheckError();

	glEnableVertexAttribArray(0);
	glCheckError();
}

// TODO: once this is done, should go into separate file
//	to create a fps monitor on the fly
// TODO: and graphs in general
void fps_monitor_initializer()
{
	// TODO: display fps counter here

	// Uncap FPS
	glfwSwapInterval(0);

	// For text rendering
        glEnable(GL_BLEND);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

	// Load font
	winman.load_font(1);

	// Set line width
	glLineWidth(5.0f);

	// Fill vertices with 0
	for (size_t i = 0; i < 3 * 10; i++)
		fps_vertices[i] = 0;

	// Create axes
	axes = SVA3({
		0,	100,	0,
		0,	0,	0,
		100,	0,	0,
	});

	// Allocate graph buffer
	glGenVertexArrays(1, &vao);
	glGenBuffers(1, &vbo);
	glBindVertexArray(vao);

	generate_arrays();

	// Create and configure base graphing shader
	basic = Shader::from_source(basic_vert, basic_frag);

	basic.use();
	//basic.set_vec3("ecolor", {1.0, 1.0, 1.0});
	basic.set_mat4("projection", glm::ortho(-10.0f, 110.0f, -10.0f, 110.0f));

	// Set text
	text_fps = ui::Text("FPS", 100, 10, 0.9, {1.0, 0.5, 1.0});

	// TODO: move to init or smthing
}

void fps_monitor_renderer()
{
	// Static timer
	static float time = 0.0f;
	static float totfps = 0.0f;
	static float iters = 0.0f;
	static float avgfps = 0.0f;

	// Clear the color
	glClearColor(0.05f, 0.05f, 0.05f, 1.0f);
	glClear(GL_COLOR_BUFFER_BIT);

	// Get time stuff
	float current_frame = glfwGetTime();
	delta_time = current_frame - last_frame;
	last_frame = current_frame;
	time += delta_time;

	int fps = 1/delta_time;
	totfps += fps;
	iters++;

	if (time > 0.5f) {
		// Delete the first point
		fps_vertices.erase(fps_vertices.begin(), fps_vertices.begin() + 3);

		// 0 -> 600 fps
		avgfps = totfps/iters;

		float normalized = 100.0f * avgfps/600.0f;
		fps_vertices.push_back(100.0f);
		fps_vertices.push_back(normalized);
		fps_vertices.push_back(0.0f);

		// Shift other points

		// TODO: += fields (getter)
		for (size_t i = 0; i < fps_vertices.size(); i += 3) {
			float px = fps_vertices[i];
			fps_vertices[i] = std::max(px - 10.0f, 0.0f);
		}

		// Regenerate buffer data
		glBindBuffer(GL_ARRAY_BUFFER, vbo);
		glCheckError();

		glBufferData(GL_ARRAY_BUFFER, sizeof(float) * fps_vertices.size(),
			&fps_vertices[0], GL_STATIC_DRAW);

		// Reset time
		time = 0.0f;
	}

	// Draw the graph
	basic.use();
	basic.set_vec3("ecolor", {1.0, 1.0, 1.0});

	glBindVertexArray(vao);
	glCheckError();

	glDrawArrays(GL_LINE_STRIP, 0, fps_vertices.size()/3);
	glCheckError();

	// Draw axes
	basic.set_vec3("ecolor", {0.6, 0.6, 0.6});

	axes.draw(GL_LINE_STRIP);

	// Draw text
	text_fps.set_str(std::to_string(delta_time).substr(0, 6)
		+ "s delta, " + std::to_string(avgfps).substr(0, 6) + " fps");
	text_fps.draw(*winman.cres.text_shader);
}

// Render function for main window
Mesh hit_cube1;
Mesh hit_cube2;
Mesh hit_cube3;
Mesh global_hit;
Mesh source_cube1;
Mesh source_cube2;
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

lighting::PointLight light1 {
	lpos1,
	{1.0, 1.0, 1.0}		// TODO: color constants and hex strings
};

lighting::PointLight light2 {
	lpos2,
	{1.0, 1.0, 1.0}
};

// Lighting daemon
lighting::Daemon ldam;

// Wireframe sphere: TODO: is there a better way than to manually set vertices?
const glm::vec3 center {1.0, 1.0, 1.0};
const float radius = 0.2f;

SVA3 wsphere;
Shader sphere_shader;

// TODO: basic_3d.vert
const char *sphere_vert = R"(
#version 330 core

layout (location = 0) in vec3 vertex;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

void main()
{
	gl_Position = projection * view * model * vec4(vertex, 1.0);
}
)";

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

	global_hit = mesh::cuboid({3, -1.0, 12.0}, 10, 1, 10);

	source_cube1 = mesh::cuboid(lpos1, 0.5, 0.5, 0.5);
	source_cube2 = mesh::cuboid(lpos2, 0.5, 0.5, 0.5);

	// Set line width
	glLineWidth(5.0f);

	// Create the sphere
	wsphere = mesh::wireframe_sphere(center, radius);

	sphere_shader = Shader::from_source(sphere_vert, basic_frag);
	sphere_shader.set_name("sphere_shader");

	// Construct the tree
	tree = make_tree({0, -0.5, -2}, {0, 1, 0}, 3.0f, 0.5f);

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

	Logger::ok("Finished constructing skybox.");

	// Add objects and lights to the ldam system
	ldam.add_light(dirlight);

	// ldam.add_light(light1);
	// ldam.add_light(light2);

	// ldam.add_object(&source_cube1, {.color = {1.0, 1.0, 1.0}}, lighting::COLOR_ONLY);
	// ldam.add_object(&source_cube2, {.color = {1.0, 1.0, 1.0}}, lighting::COLOR_ONLY);

	ldam.add_object(&hit_cube1, {.color = {0.5, 1.0, 0.5}});
	ldam.add_object(&hit_cube2, {.color = {1.0, 0.5, 0.5}});
	ldam.add_object(&hit_cube3, {.color = {0.9, 0.9, 0.9}});

	ldam.add_object(&global_hit, {.color = {0.9, 0.9, 0.9}});

	ldam.add_object(&tree, {.color = {1.0, 0.8, 0.5}});
}

void main_renderer()
{
	// Get time stuff
	float current_frame = glfwGetTime();
	delta_time = current_frame - last_frame;
	last_frame = current_frame;

	// Process input
	process_input(mercury::winman.cwin);

	// render
	glClearColor(0.05f, 1.00f, 0.05f, 1.0f);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	// Create model, view, projection matrices
	glm::mat4 model = glm::mat4(1.0f);
	model = glm::translate(model, glm::vec3(0.0f, 0.0f, 0.0f));
	model = glm::scale(model, glm::vec3(1.0f, 1.0f, 1.0f));		// TODO: what is the basis of this computation?

	glm::mat4 view = camera.get_view();
	glm::mat4 projection = glm::perspective(
		glm::radians(camera.zoom),
		(float) SCR_WIDTH / (float) SCR_HEIGHT,
		0.1f, 100.0f
	);

	// Set lighting daemon uniforms
	ldam.uniforms = {
		model,
		view,
		projection,
		camera.position
	};

	// Render scene
	ldam.render();

	// Draw sphere		TODO: seriously some way to check that uniforms have been set
	sphere_shader.use();	// TODO: make into a common shader
	sphere_shader.set_mat4("model", model);
	sphere_shader.set_mat4("view", view);
	sphere_shader.set_mat4("projection", projection);
	// wsphere.draw(GL_LINE_STRIP);

	sphere_shader.use();
	sphere_shader.set_vec3("ecolor", 0.1, 0.5, 1.0);
	for (const SVA3 &sva : spheres)
		sva.draw(GL_LINE_STRIP);
	
	sphere_shader.use();
	sphere_shader.set_vec3("ecolor", 0.9, 0.9, 0.5);
	for (const SVA3 &sva : annotations)
		sva.draw(GL_LINE_STRIP);

	// Draw tree skeleton
	lines[0].set_mvp(model, view, projection);
	for (const ui::Line &line : lines)
		line.draw();

	// Draw skybox
	view = glm::mat4(glm::mat3(camera.get_view()));

	// Set skybox shader properties
	winman.cres.sb_shader->use();
	winman.cres.sb_shader->set_mat4("projection", projection);
	winman.cres.sb_shader->set_mat4("view", view);

	// TODO: use the current shader in the skybox draw method
	sb.draw(*winman.cres.sb_shader);
}

// Program render loop condition
bool rcondition()
{
	return !glfwWindowShouldClose(winman[0]);
}

int main()
{
	// Initialize mercury
	mercury::init();

	// Add windows
	// winman.add_win("FPS Monitor");

	// Set winman bindings
	winman.set_condition(rcondition);

	winman.set_initializer(0, main_initializer);
	// winman.set_initializer(1, fps_monitor_initializer);

	winman.set_renderer(0, main_renderer);
	// winman.set_renderer(1, fps_monitor_renderer);

	// Render loop
	winman.run();

	// TODO: mercury deinit function?
	// Terminate GLFW
	glfwTerminate();

	return 0;
}

// process all input: query GLFW whether relevant keys are pressed/released this frame and react accordingly
// ---------------------------------------------------------------------------------------------------------
void process_input(GLFWwindow *window)
{
	if (glfwGetKey(window, GLFW_KEY_BACKSPACE) == GLFW_PRESS)
		glfwSetWindowShouldClose(window, true);

	if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS) {
		// glfwSetWindowShouldClose(window, true);
		glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
	}

	float cameraSpeed = 5 * delta_time;

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

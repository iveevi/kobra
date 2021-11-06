#include <iostream>

#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include "include/camera.hpp"
#include "include/shader.hpp"
#include "include/model.hpp"
#include "include/init.hpp"

// Using declarations
using namespace mercury;

/* #include <glad/glad.h>
// #include <GLFW/glfw3.h>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <learnopengl/shader_m.h>
#include <learnopengl/camera.h>
#include <learnopengl/model.h> */

// #include <iostream>

void framebuffer_size_callback(GLFWwindow* window, int width, int height);
void mouse_callback(GLFWwindow* window, double xpos, double ypos);
void scroll_callback(GLFWwindow* window, double xoffset, double yoffset);
void processInput(GLFWwindow *window);

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

Mesh cuboid(const glm::vec3 &center, float w, float h, float d)
{
	float w2 = w/2;
	float h2 = h/2;
	float d2 = d/2;

	// 8 faces
	Mesh::AVertex vertices {
		Vertex {.position = {center.x + w2, center.y + h2, center.z - d2}},
		Vertex {.position = {center.x - w2, center.y + h2, center.z - d2}},
		Vertex {.position = {center.x + w2, center.y - h2, center.z - d2}},
		Vertex {.position = {center.x - w2, center.y - h2, center.z - d2}},
		Vertex {.position = {center.x + w2, center.y + h2, center.z + d2}},
		Vertex {.position = {center.x - w2, center.y + h2, center.z + d2}},
		Vertex {.position = {center.x + w2, center.y - h2, center.z + d2}},
		Vertex {.position = {center.x - w2, center.y - h2, center.z + d2}}
	};

	// Each row is a square constiting of two triangles
	Mesh::AIndices indices {
		0, 1, 2,	1, 2, 3,
		4, 5, 6,	5, 6, 7,
		0, 4, 2,	4, 2, 6,
		2, 3, 6,	3, 6, 7,
		0, 1, 4,	1, 4, 5,
		1, 3, 7,	1, 5, 7
	};

	return Mesh {vertices, {}, indices};
}

int main()
{
	mercury::init();

	// tell stb_image.h to flip loaded texture's on the y-axis (before loading model).
	stbi_set_flip_vertically_on_load(true);

	// configure global opengl state
	// -----------------------------
	glEnable(GL_DEPTH_TEST);

	// Shader
	Shader ourShader(
		"resources/backpack/shader.vs",
		"resources/backpack/shader.fs"
	);

	Shader meshShader(
		"resources/shaders/mesh_shader.vs",
		"resources/shaders/mesh_shader.fs"
	);

	auto vert = [](const glm::vec3 &pos) {
		return Vertex {
			.position = pos
		};
	};

	Mesh::AVertex vertices {
		vert({0, 0, 0}),
		vert({0, 1, 0}),
		vert({1, 0, 0}),
		vert({1, 1, 0}),
		vert({0, 0, 1}),
		vert({0, 1, 1}),
		vert({1, 0, 1}),
		vert({1, 1, 1})
	};

	Mesh::AIndices indices = {
		0, 1, 2,
		1, 2, 3,
		4, 5, 6,
		5, 6, 7,
		0, 4, 2,
		4, 2, 6,
		2, 3, 6,
		3, 6, 7,
		0, 1, 4,
		1, 4, 5,
		1, 3, 7,
		1, 5, 7
	};

	Mesh mesh {vertices, {}, indices};
	Mesh cuboid1 = cuboid({2, 1, 0}, 0.5, 0.5, 0.5);

	// draw in wireframe
	//glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);

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
		glClearColor(0.05f, 0.05f, 0.05f, 1.0f);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

		// Create model, view, projection
		glm::mat4 model = glm::mat4(1.0f);
		model = glm::translate(model, glm::vec3(0.0f, 0.0f, 0.0f)); // translate it down so it's at the center of the scene
		model = glm::scale(model, glm::vec3(1.0f, 1.0f, 1.0f));	// it's a bit too big for our scene, so scale it down
		glm::mat4 view = camera.get_view();
		glm::mat4 projection = glm::perspective(glm::radians(camera.zoom),
				(float) SCR_WIDTH / (float) SCR_HEIGHT, 0.1f, 100.0f);

		meshShader.use();
		meshShader.set_mat4("model", model);
		meshShader.set_mat4("view", view);
		meshShader.set_mat4("projection", projection);
		meshShader.set_vec3("color", {0.5, 1.0, 0.5});

		// Draw the cube
		mesh.draw(meshShader);
		
		meshShader.use();
		meshShader.set_vec3("color", {1.0, 0.2, 0.2});

		cuboid1.draw(meshShader);

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
	if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS)
		camera.move(cameraSpeed * camera.front);
	if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS)
		camera.move(-cameraSpeed * camera.front);
	if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS)
		camera.move(-cameraSpeed * camera.right);
	if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS)
		camera.move(cameraSpeed * camera.right);
}

// glfw: whenever the window size changed (by OS or user resize) this callback function executes
// ---------------------------------------------------------------------------------------------
void framebuffer_size_callback(GLFWwindow* window, int width, int height)
{
	// make sure the viewport matches the new window dimensions; note that width and
	// height will be significantly larger than specified on retina displays.
	glViewport(0, 0, width, height);
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

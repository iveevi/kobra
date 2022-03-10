// Engine headers
#include "global.hpp"

// Keyboard callback
// TODO: in class
bool mouse_tracking = true;

void key_callback(GLFWwindow *window, int key, int scancode, int action, int mods)
{
        if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS)
                glfwSetWindowShouldClose(window, GL_TRUE);

        // Camera movement
        float speed = 0.5f;
        if (key == GLFW_KEY_W)
                world.camera.transform.position += world.camera.transform.forward * speed;
        else if (key == GLFW_KEY_S)
                world.camera.transform.position -= world.camera.transform.forward * speed;

        if (key == GLFW_KEY_A)
                world.camera.transform.position -= world.camera.transform.right * speed;
        else if (key == GLFW_KEY_D)
                world.camera.transform.position += world.camera.transform.right * speed;

	if (key == GLFW_KEY_E)
		world.camera.transform.position += world.camera.transform.up * speed;
	else if (key == GLFW_KEY_Q)
		world.camera.transform.position -= world.camera.transform.up * speed;

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

	yaw += xoffset;
	pitch += yoffset;

	if (pitch > 89.0f)
		pitch = 89.0f;
	else if (pitch < -89.0f)
		pitch = -89.0f;

	world.camera.transform.set_euler(pitch, yaw);

	last_x = xpos;
	last_y = ypos;
}

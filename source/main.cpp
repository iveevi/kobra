#include "../include/init.hpp"
#include "../include/text.hpp"
#include "../include/ui/rect.hpp"
#include "../include/ui/pure_rect.hpp"

#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

void processInput(GLFWwindow *window);
void RenderText(mercury::Shader &shader, std::string text, float x, float y, float scale, glm::vec3 color);

int main()
{
	// Initialize mercury
	GLFWwindow *window = mercury::init();

	// Setup the shader
	// TODO: put 2d project into win struct...
	glm::mat4 projection = glm::ortho(0.0f, mercury::win_width, 0.0f, mercury::win_height);
	mercury::Char::shader.use();
	mercury::Char::shader.set_mat4("projection", projection);

	// Texts
	mercury::Text title("Mercury Engine", 10.0,
		mercury::win_height - 50.0, 1.0,
		glm::vec3(0.5, 0.5, 0.5));

	// Shapes
	mercury::ui::Rect rect_files(
		{50.0, 50.0},
		{700.0, 500.0},
		{0.1, 0.1, 0.1, 1.0},
		5.0,
		{0.5, 0.5, 0.5, 1.0}
	);

	while (!glfwWindowShouldClose(window)) {
		processInput(window);

		glClearColor(0.1f, 0.1f, 0.1f, 1.0f);
		glClear(GL_COLOR_BUFFER_BIT);

		title.draw();
		rect_files.draw();

		glfwSwapBuffers(window);
		glfwPollEvents();
	}

	glfwTerminate();
	return 0;
}

void processInput(GLFWwindow *window)
{
	if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
		glfwSetWindowShouldClose(window, true);
}

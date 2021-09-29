#include "include/init.hpp"
#include "include/ui/text.hpp"
#include "include/ui/rect.hpp"
#include "include/ui/pure_rect.hpp"
#include "include/ui/button.hpp"
#include "include/ui/ui_layer.hpp"

#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

void process_input(GLFWwindow *window)
{
	if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
		glfwSetWindowShouldClose(window, true);
}

void mouse_button_callback(GLFWwindow* window, int button, int action, int mods)
{
	if (button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_PRESS) {
		double xpos, ypos;

		glfwGetCursorPos(window, &xpos, &ypos);
		mercury::win_mhandler.publish(
			{mercury::MouseBus::MOUSE_PRESSED, {xpos, ypos}}
		);
	}
}

// Handlers
class MyHandler : public mercury::Handler {
public:
	virtual void call(size_t *data) const override {
		std::cout << "MyHandler ftn..." << std::endl;

		glm::vec2 mpos = *((glm::vec2 *) data);
		std::cout << "\tposition = " << mpos.x << ", " << mpos.y << std::endl;
	}
};

int main()
{
	// Initialize mercury
	GLFWwindow *window = mercury::init();
	glfwSetMouseButtonCallback(window, mouse_button_callback);

	// Setup the shader
	// TODO: put 2d project into win struct...
	glm::mat4 projection = glm::ortho(0.0f, mercury::win_width, 0.0f, mercury::win_height);
	mercury::Char::shader.use();
	mercury::Char::shader.set_mat4("projection", projection);

	// Texts
	mercury::ui::Text title("Mercury Engine", 25.0,
		mercury::win_height - 50.0, 1.0,
		glm::vec3(0.5, 0.5, 0.5));

	// Shapes
	mercury::ui::Rect rect_files(
		{25.0, 75.0},
		{575.0, 575.0},
		{0.1, 0.1, 0.1, 1.0},
		5.0,
		{0.5, 0.5, 0.5, 1.0}
	);

	mercury::ui::Rect *rect_button = new mercury::ui::Rect(
		{600.0, 100.0},
		{700.0, 200.0},
		{1.0, 0.1, 0.1, 1.0},
		5.0,
		{0.5, 1.0, 0.5, 1.0}
	);

	// TODO: new UI layer
	mercury::ui::Button button(rect_button, new MyHandler());

	mercury::ui::UILayer ui_layer;
	ui_layer.add_element(&title);
	ui_layer.add_element(&rect_files);
	ui_layer.add_element(&button);

	while (!glfwWindowShouldClose(window)) {
		process_input(window);

		glClearColor(0.1f, 0.1f, 0.1f, 1.0f);
		glClear(GL_COLOR_BUFFER_BIT);

		// NOTE: UI Layer is always drawn last
		ui_layer.draw();

		glfwSwapBuffers(window);
		glfwPollEvents();
	}

	glfwTerminate();
	return 0;
}

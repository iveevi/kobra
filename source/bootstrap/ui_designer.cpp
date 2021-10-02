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
	if (button == GLFW_MOUSE_BUTTON_LEFT) {
		// TODO: put into function in init.hpp
		double xpos, ypos;

		glfwGetCursorPos(window, &xpos, &ypos);
		mercury::cwin.mouse_handler.publish(
			{(mercury::MouseBus::Type) action, {xpos, ypos}}
		);
	}
}

class DragNode : public mercury::ui::Button {
	glm::vec2	_ppos;
	bool		_drag = false;
public:
	DragNode() {}
	DragNode(mercury::ui::Shape *shape, mercury::ui::UILayer *layer = nullptr)
		: mercury::ui::Button(shape, nullptr, nullptr, layer) {}

	void on_pressed(const glm::vec2 &mpos) override {
		_ppos = mpos;
		_drag = true;
	}

	void on_released(const glm::vec2 &mpos) override {
		_drag = false;
	}

	void draw() override {
		Button::draw();

		if (_drag) {
			double xpos, ypos;
			glfwGetCursorPos(mercury::cwin.window, &xpos, &ypos);

			glm::vec2 dpos = glm::vec2 {xpos, ypos} - _ppos;
			_ppos = {xpos, ypos};
			move(dpos);
		}
	}
};

class UIDesigner {
	// Text
	mercury::ui::Text *title;
	mercury::ui::Text *button_text;

	// Shapes
	mercury::ui::Rect *rect_dragb;

	// Buttons
	DragNode *dnode;

	// UILayer
	mercury::ui::UILayer *ui_layer;
	mercury::ui::UILayer *dnode_layer;

	void _init() {
		// Initialize mercury
		mercury::init();

		// TODO: do in init
		glfwSetMouseButtonCallback(mercury::cwin.window, mouse_button_callback);

		// Setup the shader
		// TODO: put 2d project into win struct...
		glm::mat4 projection = glm::ortho(0.0f, mercury::cwin.width,
				0.0f, mercury::cwin.height);
		mercury::Char::shader.use();
		mercury::Char::shader.set_mat4("projection", projection);
	}
public:
	UIDesigner() {
		_init();

		// Texts
		title = new mercury::ui::Text("UI Designer",
			10.0, 10.0, 1.0,
			glm::vec3(0.5, 0.5, 0.5)
		);

		button_text = new mercury::ui::Text("Save",
			10.0, 10.0, 0.75,
			glm::vec3(0.5, 0.5, 0.5)
		);

		// Shapes
		rect_dragb = new mercury::ui::Rect(
			{100.0, 100.0},
			{200.0, 200.0},
			{1.0, 0.1, 0.1, 1.0},
			5.0,
			{0.5, 1.0, 0.5, 1.0}
		);

		// Layers
		dnode_layer = new mercury::ui::UILayer();
		ui_layer = new mercury::ui::UILayer();
	}

	~UIDesigner() {
		delete title;
		delete button_text;
		delete rect_dragb;
		delete dnode_layer;
		delete dnode;
		delete ui_layer;
	}

	void build() {
		// Building UI layers
		button_text->center_within(rect_dragb->get_bounds(), true);
		dnode_layer->add_element(button_text);

		dnode = new DragNode(rect_dragb, dnode_layer);

		ui_layer->add_element(title);
		ui_layer->add_element(dnode);
	}

	int run() {
		// TODO: loop function in cwin?
		while (!glfwWindowShouldClose(mercury::cwin.window)) {
			// std::cout << "----------------------> " << i++ << std::endl;

			process_input(mercury::cwin.window);

			glClearColor(0.1f, 0.1f, 0.1f, 1.0f);
			glClear(GL_COLOR_BUFFER_BIT);

			// NOTE: UI Layer is always drawn last
			ui_layer->draw();

			glfwSwapBuffers(mercury::cwin.window);
			glfwPollEvents();
		}

		glfwTerminate();
		return 0;
	};
};

int main()
{
	UIDesigner UID;
	UID.build();
	return UID.run();
}

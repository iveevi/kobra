// Engine headers
#include "include/init.hpp"
#include "include/common.hpp"
#include "include/ui/text.hpp"
#include "include/ui/rect.hpp"
#include "include/ui/pure_rect.hpp"
#include "include/ui/button.hpp"
#include "include/ui/ui_layer.hpp"

// Extra headers
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

// Using declarations
using namespace mercury;

void process_input(GLFWwindow *window)
{
	if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
		glfwSetWindowShouldClose(window, true);
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

// TODO: Abstract from an application class that
// provides tools for "garbage collection"
// - has its own allocator for faster allocation
class UIDesigner {
	// Text
	ui::Text *title;
	ui::Text *save_button_text;
	ui::Text *add_button_text;

	// Shapes
	ui::Rect *border;
	ui::Rect *vp_bounds;
	ui::Rect *save_button_box;
	ui::Rect *add_button_box;

	// Buttons
	ui::Button *save_button;
	ui::Button *add_button;

	// UILayer
	ui::UILayer *ui_layer;
	ui::UILayer *viewport_layer;
	ui::UILayer *add_button_layer;
	ui::UILayer *save_button_layer;
public:
	UIDesigner() {
		// Initialize mercury
		mercury::init();

		// Texts
		title = new ui::Text("UI Designer",
			10.0, 10.0, 1.0,
			glm::vec3(0.5, 0.5, 0.5)
		);

		save_button_text = new ui::Text("Save",
			10.0, 10.0, 0.6,
			glm::vec3(0.5, 0.5, 0.5)
		);

		add_button_text = new ui::Text("Add",
			10.0, 10.0, 0.6,
			glm::vec3(0.5, 0.5, 0.5)
		);

		// Shapes
		border = new ui::Rect(
			{25.0, 75.0},
			{775.0, 575.0},
			{.1, 0.1, 0.1, 1.0},
			5.0,
			{0.5, 0.5, 0.5, 1.0}
		);

		vp_bounds = new ui::Rect(
			{0.0, 0.0},
			{800.0, 600.0},
			{0.1, 0.1, 0.1, 1.0},
			5.0,
			{0.5, 0.7, 0.5, 1.0}
		);

		save_button_box = new ui::Rect(
			{650.0, 10.0},
			{750.0, 50.0},
			{0.1, 0.1, 0.1, 1.0},
			5.0,
			{0.5, 0.5, 0.5, 1.0}
		);

		add_button_box = new ui::Rect(
			{500.0, 10.0},
			{600.0, 50.0},
			{0.1, 0.1, 0.1, 1.0},
			5.0,
			{0.5, 0.5, 0.5, 1.0}
		);

		// Layers
		ui_layer = new ui::UILayer();
		viewport_layer = new ui::UILayer();
		save_button_layer = new ui::UILayer();
		add_button_layer = new ui::UILayer();
	}

	~UIDesigner() {
	}

	void build() {
		// Building UI layers
		save_button_text->center_within(save_button_box->get_bounds(), true);
		save_button_layer->add_element(save_button_text);

		add_button_text->center_within(add_button_box->get_bounds(), true);

		// TODO: add element needs to check for nullptr
		add_button_layer->add_element(add_button_text);

		save_button = new ui::Button(save_button_box, nullptr, nullptr, save_button_layer);
		add_button = new ui::Button(add_button_box, nullptr, nullptr, add_button_layer);

		ui_layer->add_element(title);
		ui_layer->add_element(save_button);
		ui_layer->add_element(add_button);
		ui_layer->add_element(border);

		viewport_layer->add_element(vp_bounds);
		viewport_layer->add_element(save_button);
	}

	int run() {
		// TODO: loop function in cwin?
		while (!glfwWindowShouldClose(mercury::cwin.window)) {
			process_input(mercury::cwin.window);

			glClearColor(0.1f, 0.1f, 0.1f, 1.0f);
			glClear(GL_COLOR_BUFFER_BIT);

			// NOTE: UI Layer is always drawn last
			// glViewport(0.0f, 0.0f, 1600.0f, 1200.0f);
			glViewport(0.0f, 0.0f, 800.0f, 600.0f);
			glm::mat4 proj = glm::ortho(0.0f, 800.0f, 0.0f, 600.0f);
			ui::UIElement::set_projection(proj);
			ui_layer->draw();

			// TODO: make a focus function for changing viewports
			// glViewport(50, 50, 1500, 950);

			// TODO: focus function eeds to take into account the
			// achor location (of 0-0)

			// TODO: need to make a 16:9 ratio thing (IRL)
			glViewport(25, 25, 750, 500);
			ui::UIElement::set_projection(
				glm::ortho(-25.0f, 825.0f, -25.0f, 625.0f)
			);
			viewport_layer->draw();

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

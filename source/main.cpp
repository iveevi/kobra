#include "../include/init.hpp"
#include "../include/text.hpp"

#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

void processInput(GLFWwindow *window);
void RenderText(mercury::Shader &shader, std::string text, float x, float y, float scale, glm::vec3 color);

namespace mercury {

glm::vec2 transform(const glm::vec2 &in)
{
	return glm::vec2 {
		(in.x - win_width/2)/(win_width/2),
		-(in.y - win_height/2)/(win_height/2)
	};
}

class PureRect {
	unsigned int 	_vao;
	unsigned int 	_vbo;
	unsigned int 	_ebo;

	Shader		_shader;

	static const char *default_vs;
	static const char *default_fs;

	void _genbufs(const glm::vec2 p1, const glm::vec2 p2) {
		float vertices[] = {
			p2.x, p1.y, 0.0f,
			p2.x, p2.y, 0.0f,
			p1.x, p2.y, 0.0f,
			p1.x, p1.y, 0.0f,
		};

		unsigned int indices[] = {
			0, 3, 1,
			1, 3, 2
		};

		glGenVertexArrays(1, &_vao);
		glGenBuffers(1, &_vbo);
		glGenBuffers(1, &_ebo);
		glBindVertexArray(_vao);

		glBindBuffer(GL_ARRAY_BUFFER, _vbo);
		glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, _ebo);
		glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW);

		glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
		glEnableVertexAttribArray(0);
		glBindBuffer(GL_ARRAY_BUFFER, 0);
	}
public:
	PureRect(const glm::vec2 &p1, const glm::vec2 &p2) {
		_genbufs(
			transform(p1),
			transform(p2)
		);

		_shader = Shader::from_source(
			default_vs,
			default_fs
		);
	}

	PureRect(float x1, float y1, float x2, float y2) {
		_genbufs(
			transform({x1, y1}),
			transform({x2, y2})
		);

		_shader = Shader::from_source(
			default_vs,
			default_fs
		);
	}

	void set_color(const glm::vec4 &color)
	{
		_shader.use();
		_shader.set_vec4("rect_color", color);
	}

	void draw() {
		_shader.use();

		glBindVertexArray(_vao);
		glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);
	}
};

const char *PureRect::default_vs = R"(
#version 330 core

layout (location = 0) in vec3 apos;

void main()
{
	gl_Position = vec4(apos.x, apos.y, apos.z, 1.0);
}
)";

const char *PureRect::default_fs = R"(
#version 330 core

out vec4 fragment_color;

uniform vec4 rect_color;

void main()
{
	fragment_color = rect_color; // vec4(1.0);
}
)";

}

int main()
{
	// Initialize mercury
	GLFWwindow *window = mercury::init();

	// Setup the shader
	// TODO: put 2d project into win struct...
	glm::mat4 projection = glm::ortho(0.0f, mercury::win_width, 0.0f, mercury::win_height);
	mercury::Char::shader.use();
	mercury::Char::shader.set_mat4("projection", projection);

	mercury::Text t1("This is sample text", 25.0f, 25.0f, 1.0f, glm::vec3(0.5, 0.8f, 0.2f));
	mercury::Text t2("(C) LearnOpenGL.com", 540.0f, 570.0f, 0.5f, glm::vec3(0.3, 0.7f, 0.9f));

	mercury::PureRect pr(
		{0.0, 0.0},
		{400.0, 300.0}
	);

	pr.set_color({0.2, 0.5, 0.1, 1.0});

	float time = 0.0f;
	while (!glfwWindowShouldClose(window)) {
		processInput(window);

		glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
		glClear(GL_COLOR_BUFFER_BIT);

		pr.draw();
		t1.draw();
		t2.draw();

		t1.set_position(200 + 100 * cos(time), 200 + 100 * sin(time));

		glfwSwapBuffers(window);
		glfwPollEvents();

		time += 0.001f;
	}

	glfwTerminate();
	return 0;
}

void processInput(GLFWwindow *window)
{
	if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
		glfwSetWindowShouldClose(window, true);
}
